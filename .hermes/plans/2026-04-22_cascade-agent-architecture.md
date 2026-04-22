# Cascade Agent — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a robust, asynchronous LLM agentic engine in Rust — a CLI tool backed by a reusable library crate — leveraging `llm-cascade` for inference with tool execution, context management, dynamic skills, vector knowledge base, and orchestrator-ready bidirectional communication.

**Architecture:** Trait-based OOP with Tokio async runtime. The core is an interruptible agentic loop that sends conversations to `llm-cascade`, executes tool calls, manages memory via token-aware compaction, discovers/invokes dynamic skills, and communicates through an abstract orchestrator channel. WebSocket is the first transport; gRPC can be slotted in behind the same trait.

**Tech Stack:** Rust, Tokio, llm-cascade, LanceDB, fastembed (multilingual-e5-base), gray\_matter, tokio-tungstenite, reqwest, serde, anyhow/thiserror, tokenizers, clap, tracing

---

## Directory Structure

```
cascade-agent/
├── Cargo.toml
├── config.example.toml
├── SOUL.md                          # Default system prompt
├── data/
│   ├── skills/                      # Dynamic skill directories
│   │   └── example-skill/
│   │       ├── SKILL.md             # Metadata + instructions
│   │       └── run.sh              # Optional executable
│   ├── plans/                       # Task plans (markdown)
│   ├── outputs/                     # Task-specific output dirs
│   └── lancedb/                     # Local vector DB storage
├── src/
│   ├── lib.rs                       # Public API re-exports
│   ├── main.rs                      # CLI entry (clap)
│   ├── error.rs                     # Library error types (thiserror)
│   ├── config.rs                    # TOML config + loading
│   ├── agent/
│   │   ├── mod.rs                   # AgentLoop struct, builder, public API
│   │   ├── state.rs                 # ConversationState, turn tracking
│   │   └── loop.rs                  # Core agentic loop (tokio::select!)
│   ├── memory/
│   │   ├── mod.rs                   # MemoryManager (token tracking + compaction trigger)
│   │   ├── compaction.rs            # Summarization-based context compaction
│   │   └── tokenizer.rs             # Token counting via tokenizers crate
│   ├── skills/
│   │   ├── mod.rs                   # SkillManager (discovery, registry)
│   │   ├── parser.rs                # SKILL.md frontmatter parsing (gray_matter)
│   │   ├── executor.rs              # Skill executable invocation (stdin JSON → stdout JSON)
│   │   └── types.rs                 # Skill, SkillMetadata, SkillResult structs
│   ├── tools/
│   │   ├── mod.rs                   # Tool trait + ToolRegistry
│   │   ├── search.rs                # Tavily + Brave Search API tools
│   │   ├── knowledge_tool.rs        # Vector DB query tool (exposed to LLM)
│   │   └── builtin.rs               # Plan, memory, skill-management tools
│   ├── knowledge/
│   │   ├── mod.rs                   # KnowledgeBase facade
│   │   ├── vectordb.rs              # LanceDB wrapper (create collections, insert, search)
│   │   └── embeddings.rs            # fastembed wrapper (sync → spawn_blocking)
│   ├── orchestrator/
│   │   ├── mod.rs                   # OrchestratorConnection trait + Router
│   │   ├── websocket.rs             # WebSocket transport implementation
│   │   ├── server.rs                # Localhost WS server (for future orchestrator)
│   │   └── types.rs                 # OrchestratorMessage enum (JSON serde)
│   └── planning/
│       ├── mod.rs                   # PlanManager (create, update, persist plans)
│       └── types.rs                 # Plan, PlanStep, PlanStatus structs
└── tests/
    ├── integration/
    │   ├── agent_loop_test.rs
    │   ├── skill_execution_test.rs
    │   └── knowledge_base_test.rs
    └── common/
        └── fixtures.rs
```

---

## Cargo.toml Dependencies

```toml
[package]
name = "cascade-agent"
version = "0.1.0"
edition = "2021"
description = "Async LLM agentic engine with tool execution, memory management, and dynamic skills"
license = "MIT"

[dependencies]
# Async runtime
tokio = { version = "1", features = ["full", "sync"] }
futures = "0.3"

# LLM inference (cascading failover)
llm-cascade = "0.1"

# Token counting
tokenizers = "0.21"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"

# Error handling
anyhow = "1"
thiserror = "2"

# Vector DB & Embeddings
lancedb = "0.27"
arrow-array = "55"
arrow-schema = "55"
fastembed = "5"

# HTTP (search APIs)
reqwest = { version = "0.12", features = ["json"] }

# Orchestrator communication
tokio-tungstenite = { version = "0.26", features = ["connect"] }
futures-util = "0.3"

# Markdown / Frontmatter parsing
gray-matter = { version = "0.3", features = ["yaml"] }

# CLI
clap = { version = "4", features = ["derive"] }

# Logging / Tracing
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Utilities
uuid = { version = "1", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
dirs = "6"

[dev-dependencies]
tempfile = "3"
tokio-test = "0.4"
```

---

## Core Types & Traits

### 1. Error Types (`src/error.rs`)

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Cascade inference failed: {0}")]
    InferenceFailed(String),

    #[error("Tool execution failed for '{tool}': {reason}")]
    ToolFailed { tool: String, reason: String },

    #[error("Context limit exceeded: {current} tokens (limit: {max})")]
    ContextOverflow { current: usize, max: usize },

    #[error("Skill error: {0}")]
    SkillError(String),

    #[error("Knowledge base error: {0}")]
    KnowledgeError(String),

    #[error("Orchestrator error: {0}")]
    OrchestratorError(String),

    #[error("Config error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, AgentError>;
```

### 2. Configuration (`src/config.rs`)

```rust
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct AgentConfig {
    pub agent: AgentSettings,
    pub memory: MemorySettings,
    pub knowledge: KnowledgeSettings,
    pub orchestrator: OrchestratorSettings,
    pub search: SearchSettings,
    pub paths: PathSettings,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AgentSettings {
    pub cascade_name: String,          // llm-cascade cascade key
    pub cascade_config_path: String,   // path to llm-cascade's config.toml
    pub max_tool_rounds: usize,        // safety limit for tool-call loops (default: 25)
    pub soul_md_path: String,          // path to SOUL.md system prompt
}

#[derive(Debug, Deserialize, Clone)]
pub struct MemorySettings {
    pub context_token_limit: usize,     // compact when exceeded (e.g., 120000)
    pub compaction_target_ratio: f64,   // compress to this fraction (e.g., 0.6)
    pub summarization_cascade: String,  // llm-cascade key for summarization calls
    pub tokenizer_model: String,        // HuggingFace tokenizer identifier
}

#[derive(Debug, Deserialize, Clone)]
pub struct KnowledgeSettings {
    pub db_path: String,                // LanceDB storage path
    pub embedding_model: String,        // fastembed model name (default: multilingual-e5-base)
    pub default_collection: String,     // default collection name
    pub similarity_threshold: f32,      // min similarity score (0.0-1.0)
    pub max_results: usize,             // max results per query
}

#[derive(Debug, Deserialize, Clone)]
pub struct OrchestratorSettings {
    pub enabled: bool,
    pub transport: String,              // "websocket" | "grpc" (future)
    pub bind_address: String,           // e.g., "127.0.0.1:9876"
    pub connect_url: Option<String>,    // client mode URL
}

#[derive(Debug, Deserialize, Clone)]
pub struct SearchSettings {
    pub tavily_api_key_env: String,     // env var name for Tavily key
    pub brave_api_key_env: String,      // env var name for Brave key
    pub max_results: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PathSettings {
    pub skills_dir: String,
    pub plans_dir: String,
    pub outputs_dir: String,
}
```

### 3. Agent Loop (`src/agent/`)

```rust
// src/agent/mod.rs
use tokio::sync::mpsc;

/// User-facing message injected mid-loop via the interrupt channel
#[derive(Debug, Clone)]
pub enum UserInterrupt {
    NewMessage(String),
    Cancel,
    EditPlan(String),
}

/// Outcome of a single agent turn
#[derive(Debug)]
pub enum TurnOutcome {
    Text(String),
    ToolCalls(Vec<PendingToolCall>),
    Error(String),
}

/// A tool call received from the LLM, awaiting execution
#[derive(Debug, Clone)]
pub struct PendingToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// The main agent struct — holds all subsystems and drives the loop
pub struct AgentLoop {
    config: AgentConfig,
    state: ConversationState,
    memory: MemoryManager,
    skill_manager: SkillManager,
    tool_registry: ToolRegistry,
    knowledge: KnowledgeBase,
    orchestrator: Box<dyn OrchestratorConnection>,
    interrupt_rx: mpsc::Receiver<UserInterrupt>,
    interrupt_tx: mpsc::Sender<UserInterrupt>,
}

impl AgentLoop {
    /// Create new agent with all subsystems initialized
    pub async fn new(config: AgentConfig) -> Result<Self>;

    /// Run the agent loop until completion, error, or cancellation
    pub async fn run(&mut self, initial_prompt: String) -> Result<String>;

    /// Get a handle to send interrupts into the running loop
    pub fn interrupt_sender(&self) -> mpsc::Sender<UserInterrupt>;
}
```

```rust
// src/agent/state.rs
use llm_cascade::{Message, MessageRole, Conversation};

/// Wraps llm-cascade Conversation with agent-specific metadata
pub struct ConversationState {
    pub conversation: Conversation,
    pub system_prompt: String,
    pub turn_count: usize,
    pub task_id: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl ConversationState {
    pub fn new(system_prompt: String, task_id: String) -> Self;

    /// Add a user message, check for pending interrupts
    pub fn add_user_message(&mut self, content: String);

    /// Add assistant text response
    pub fn add_assistant_text(&mut self, content: String);

    /// Add a tool result message
    pub fn add_tool_result(&mut self, tool_call_id: &str, result: &str);

    /// Serialize current state for persistence
    pub fn to_json(&self) -> Result<String>;
}
```

```rust
// src/agent/loop.rs (core loop pseudocode)

impl AgentLoop {
    pub async fn run(&mut self, initial_prompt: String) -> Result<String> {
        self.state.add_user_message(initial_prompt);

        loop {
            // 1. Check for pending user interrupts
            if let Ok(interrupt) = self.interrupt_rx.try_recv() {
                match interrupt {
                    UserInterrupt::NewMessage(msg) => {
                        self.state.add_user_message(msg);
                    }
                    UserInterrupt::Cancel => {
                        self.orchestrator.push(OrchestratorMessage::TaskCancelled).await;
                        return Ok("Task cancelled by user.".into());
                    }
                    UserInterrupt::EditPlan(content) => {
                        // Delegate to PlanManager
                    }
                }
            }

            // 2. Check memory budget, compact if needed
            let token_count = self.memory.count_tokens(&self.state)?;
            if token_count > self.config.memory.context_token_limit {
                self.memory.compact(&mut self.state).await?;
                self.orchestrator.push(OrchestratorMessage::ContextCompacted {
                    before: token_count,
                    after: self.memory.count_tokens(&self.state)?,
                }).await;
            }

            // 3. Build tool definitions from skills + built-in tools
            let tool_defs = self.tool_registry.all_definitions();

            // 4. Send to llm-cascade
            let mut convo = self.state.conversation.clone();
            convo = convo.with_tools(tool_defs);

            let response = match llm_cascade::run_cascade(
                &self.config.agent.cascade_name,
                &convo,
                &self.cascade_config,
                &self.db_conn,
            ).await {
                Ok(r) => r,
                Err(cascade_err) => {
                    // Save state, exponential backoff, notify
                    self.state.to_json_file(&self.config.paths.outputs_dir)?;
                    self.orchestrator.push(
                        OrchestratorMessage::Error(cascade_err.message.clone())
                    ).await;
                    return Err(AgentError::InferenceFailed(cascade_err.message));
                }
            };

            // 5. Process response
            let mut has_tool_calls = false;
            for block in &response.content {
                match block {
                    ContentBlock::Text { text } => {
                        self.state.add_assistant_text(text.clone());
                        self.orchestrator.push(
                            OrchestratorMessage::AssistantText(text.clone())
                        ).await;
                    }
                    ContentBlock::ToolCall { id, name, arguments } => {
                        has_tool_calls = true;
                        // 6. Execute tool
                        let result = self.execute_tool(id, name, arguments).await;
                        self.state.add_tool_result(id, &result);
                    }
                }
            }

            // 7. Safety: max tool rounds
            if !has_tool_calls {
                break; // Agent is done
            }
            if self.state.turn_count >= self.config.agent.max_tool_rounds {
                self.orchestrator.push(
                    OrchestratorMessage::Warning("Max tool rounds reached".into())
                ).await;
                break;
            }
        }

        Ok(self.state.last_assistant_text().unwrap_or_default())
    }
}
```

### 4. Tool System (`src/tools/`)

```rust
// src/tools/mod.rs

/// Trait that all tools (built-in and skill-derived) implement
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (matches LLM function call name)
    fn name(&self) -> &str;

    /// Description for LLM tool definition
    fn description(&self) -> &str;

    /// JSON Schema for tool parameters
    fn parameters_schema(&self) -> serde_json::Value;

    /// Execute the tool with given arguments
    async fn execute(&self, args: serde_json::Value) -> ToolResult;

    /// Convert to llm-cascade ToolDefinition
    fn to_definition(&self) -> llm_cascade::ToolDefinition {
        llm_cascade::ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters_schema(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolResult {
    pub status: ToolStatus,
    pub data: serde_json::Value,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub enum ToolStatus {
    Success,
    Error,
}

/// Registry that holds all available tools
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, tool: Box<dyn Tool>);
    pub fn get(&self, name: &str) -> Option<&dyn Tool>;
    pub fn all_definitions(&self) -> Vec<llm_cascade::ToolDefinition>;
}
```

### 5. Skill System (`src/skills/`)

```rust
// src/skills/types.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMetadata {
    pub name: String,
    pub description: String,
    pub version: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub input_format: Option<InputFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputFormat {
    pub content_type: String,    // "json"
    pub schema: serde_json::Value, // JSON Schema
}

#[derive(Debug, Clone)]
pub struct Skill {
    pub metadata: SkillMetadata,
    pub instructions: String,      // Body of SKILL.md (after frontmatter)
    pub executable_path: Option<PathBuf>,
    pub directory: PathBuf,
}
```

```rust
// src/skills/parser.rs
use gray_matter::{Matter, ParsedEntity};
use gray_matter::engine::YAML;

pub fn parse_skill_file(path: &Path) -> Result<Skill> {
    let content = std::fs::read_to_string(path)?;
    let matter = Matter::<YAML>::new();
    let parsed: ParsedEntity<SkillMetadata> = matter.parse(&content)?;

    let metadata = parsed.data.ok_or(AgentError::SkillError("Missing frontmatter".into()))?;
    let instructions = parsed.content;

    // Check for executable in same directory
    let dir = path.parent().unwrap();
    let executable_path = find_executable(dir).ok();  // e.g., run.sh, run.py, or binary

    Ok(Skill { metadata, instructions, executable_path, directory: dir.to_path_buf() })
}
```

```rust
// src/skills/executor.rs
use tokio::process::Command;
use tokio::io::AsyncWriteExt;

pub struct SkillExecutor;

impl SkillExecutor {
    /// Execute a skill's binary/script with JSON args via stdin
    pub async fn execute(
        executable: &Path,
        args: serde_json::Value,
        timeout: Duration,
    ) -> ToolResult {
        let mut child = Command::new(executable)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| ToolResult::error(&format!("Failed to spawn: {e}")))?;

        // Write JSON args to stdin, then close
        if let Some(mut stdin) = child.stdin.take() {
            let payload = serde_json::to_string(&args).unwrap_or_default();
            stdin.write_all(payload.as_bytes()).await.ok();
        }

        // Wait with timeout
        match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                // stderr goes to tracing
                if !stderr.is_empty() {
                    tracing::info!(target: "skill_output", "[{}] {}", executable.display(), stderr);
                }
                // Parse stdout as JSON
                serde_json::from_str::<ToolResult>(&stdout)
                    .unwrap_or_else(|_| ToolResult::raw_success(stdout.into()))
            }
            Ok(Err(e)) => ToolResult::error(&format!("Execution error: {e}")),
            Err(_) => ToolResult::error("Skill execution timed out"),
        }
    }
}
```

```rust
// src/skills/mod.rs
pub struct SkillManager {
    skills_dir: PathBuf,
    skills: HashMap<String, Skill>,
}

impl SkillManager {
    pub async fn new(skills_dir: PathBuf) -> Result<Self>;
    pub fn discover(&mut self) -> Result<Vec<String>>;     // Scan dir, parse all SKILL.md
    pub fn get(&self, name: &str) -> Option<&Skill>;
    pub fn create_skill(&self, name: &str, metadata: SkillMetadata, body: &str) -> Result<()>;  // Write new SKILL.md
    pub fn update_skill(&self, name: &str, body: &str) -> Result<()>;  // Edit existing SKILL.md body
    pub fn all_tools(&self) -> Vec<Box<dyn Tool>>;  // Convert skills to Tool impls
}
```

### 6. Memory & Context Management (`src/memory/`)

```rust
// src/memory/tokenizer.rs
use tokenizers::Tokenizer;

pub struct TokenCounter {
    tokenizer: Tokenizer,
}

impl TokenCounter {
    pub fn new(model_identifier: &str) -> Result<Self>;
    /// Count tokens for a slice of messages
    pub fn count_messages(&self, messages: &[llm_cascade::Message]) -> usize;
    /// Count tokens for a single string
    pub fn count_text(&self, text: &str) -> usize;
}
```

```rust
// src/memory/compaction.rs

pub struct Compactor {
    cascade_config: llm_cascade::AppConfig,
    db_conn: rusqlite::Connection,
    summarization_cascade: String,
    target_ratio: f64,
}

impl Compactor {
    /// Summarize the oldest N messages into a single system-context message.
    /// Replaces those messages in the conversation.
    pub async fn compact(
        &self,
        state: &mut ConversationState,
        token_counter: &TokenCounter,
        token_limit: usize,
    ) -> Result<CompactionReport> {
        // 1. Find how many messages from the start can be compacted
        //    (keep system prompt + last N messages, summarize the middle)
        // 2. Build a summarization prompt
        // 3. Call llm-cascade with the summarization cascade
        // 4. Replace compacted messages with a single "past_context" system message
        // 5. Return report { messages_before, messages_after, tokens_before, tokens_after }
    }
}

#[derive(Debug, Serialize)]
pub struct CompactionReport {
    pub messages_before: usize,
    pub messages_after: usize,
    pub tokens_before: usize,
    pub tokens_after: usize,
}
```

```rust
// src/memory/mod.rs
pub struct MemoryManager {
    token_counter: TokenCounter,
    compactor: Compactor,
    config: MemorySettings,
}

impl MemoryManager {
    pub fn new(config: &MemorySettings, cascade_config: ...) -> Result<Self>;
    pub fn count_tokens(&self, state: &ConversationState) -> Result<usize>;
    pub async fn compact(&self, state: &mut ConversationState) -> Result<CompactionReport>;
}
```

### 7. Knowledge Base (`src/knowledge/`)

```rust
// src/knowledge/embeddings.rs
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use std::sync::Arc;

pub struct Embedder {
    model: Arc<TextEmbedding>,
}

impl Embedder {
    pub fn new(model_name: &str) -> Result<Self>;
    /// Embed a single text (with query/passage prefix)
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
    pub fn embed_passage(&self, text: &str) -> Result<Vec<f32>>;
    /// Embed multiple texts in batch
    pub fn embed_batch(&self, texts: &[String], is_query: bool) -> Result<Vec<Vec<f32>>>;
    /// Dimension of the embedding vectors
    pub fn dimension(&self) -> usize;
}

// NOTE: fastembed is sync → use tokio::task::spawn_blocking in async contexts
```

```rust
// src/knowledge/vectordb.rs
use lancedb::Connection;
use arrow_array::{RecordBatch, StringArray, FixedSizeListArray, Float32Array, UInt64Array};
use arrow_schema::{Schema, Field, DataType, ArrowError};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    pub text: String,
    pub source: String,           // "tavily", "brave", "agent_generated", "user"
    pub metadata: serde_json::Value,
    pub timestamp: i64,           // Unix epoch
}

pub struct VectorStore {
    conn: Connection,
    embedder: Arc<Embedder>,
    default_collection: String,
}

impl VectorStore {
    pub async fn new(db_path: &str, embedder: Arc<Embedder>, default_collection: &str) -> Result<Self>;
    pub async fn create_collection(&self, name: &str) -> Result<()>;
    pub async fn insert(&self, collection: &str, entries: Vec<KnowledgeEntry>) -> Result<()>;
    pub async fn search(&self, collection: &str, query: &str, limit: usize) -> Result<Vec<SearchResult>>;
    pub async fn list_collections(&self) -> Result<Vec<String>>;
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub text: String,
    pub source: String,
    pub score: f32,
    pub metadata: serde_json::Value,
    pub timestamp: i64,
}
```

```rust
// src/knowledge/mod.rs
pub struct KnowledgeBase {
    store: VectorStore,
    config: KnowledgeSettings,
}

impl KnowledgeBase {
    pub async fn new(config: &KnowledgeSettings) -> Result<Self>;

    /// Check if we have relevant existing knowledge before searching externally
    pub async fn query_existing(&self, query: &str) -> Result<Vec<SearchResult>>;

    /// Store search results + metadata from external APIs
    pub async fn store_results(&self, query: &str, entries: Vec<KnowledgeEntry>) -> Result<()>;

    /// Create a new collection at runtime
    pub async fn create_collection(&self, name: &str) -> Result<()>;
}
```

### 8. Orchestrator Communication (`src/orchestrator/`)

```rust
// src/orchestrator/types.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum OrchestratorMessage {
    // Agent → Orchestrator
    TaskStarted { task_id: String, description: String },
    StepUpdate { task_id: String, step: usize, status: String },
    AssistantText(String),
    ToolExecuted { tool: String, status: String, duration_ms: u64 },
    ContextCompacted { before: usize, after: usize },
    Error(String),
    Warning(String),
    TaskCompleted { task_id: String, output_path: Option<String> },
    TaskCancelled,
    UserQuestion { question: String },      // Agent needs user input

    // Orchestrator → Agent
    UserReply { content: String },          // Answer to agent's question
    PlanApproval { approved: bool, feedback: Option<String> },
    CancelTask,
}
```

```rust
// src/orchestrator/mod.rs

/// Trait for bidirectional orchestrator communication.
/// Implementations: WebSocket, (future: gRPC).
#[async_trait::async_trait]
pub trait OrchestratorConnection: Send + Sync {
    /// Push a message from agent to orchestrator
    async fn push(&self, message: OrchestratorMessage) -> ();

    /// Stream of incoming messages from orchestrator
    async fn recv(&mut self) -> Option<OrchestratorMessage>;

    /// Check if connected
    fn is_connected(&self) -> bool;
}

/// No-op implementation for when orchestrator is disabled
pub struct NoopOrchestrator;

/// Factory: creates the right transport based on config
pub fn create_orchestrator(config: &OrchestratorSettings) -> Result<Box<dyn OrchestratorConnection>>;
```

```rust
// src/orchestrator/websocket.rs
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use futures_util::{SinkExt, StreamExt};

pub struct WebSocketOrchestrator {
    outbound: mpsc::Sender<OrchestratorMessage>,  // agent pushes here
    inbound: mpsc::Receiver<OrchestratorMessage>, // agent reads from here
}

impl WebSocketOrchestrator {
    /// Connect as a client to a WS server
    pub async fn connect(url: &str) -> Result<Self> {
        // Spawn two tasks:
        // 1. outbound_rx → serialize → ws_sink.send()
        // 2. ws_stream.next() → deserialize → inbound_tx.send()
    }
}
```

### 9. Planning (`src/planning/`)

```rust
// src/planning/types.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub id: String,
    pub task_id: String,
    pub title: String,
    pub steps: Vec<PlanStep>,
    pub status: PlanStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub file_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub number: usize,
    pub description: String,
    pub status: StepStatus,
    pub result: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanStatus { Draft, PendingApproval, Approved, InProgress, Completed, Cancelled }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepStatus { Pending, InProgress, Completed, Failed, Skipped }
```

```rust
// src/planning/mod.rs
pub struct PlanManager {
    plans_dir: PathBuf,
}

impl PlanManager {
    pub fn new(plans_dir: PathBuf) -> Self;
    pub fn create_plan(&self, task_id: &str, title: &str, steps: Vec<String>) -> Result<Plan>;
    pub fn update_step(&self, plan_id: &str, step_num: usize, status: StepStatus, result: Option<&str>) -> Result<()>;
    pub fn load_plan(&self, plan_id: &str) -> Result<Plan>;
    pub fn render_markdown(&self, plan: &Plan) -> String;  // For displaying to user
}
```

---

## Config Example (`config.example.toml`)

```toml
[agent]
cascade_name = "primary"
cascade_config_path = "~/.config/llm-cascade/config.toml"
max_tool_rounds = 25
soul_md_path = "./SOUL.md"

[memory]
context_token_limit = 120000
compaction_target_ratio = 0.6
summarization_cascade = "summarizer"
tokenizer_model = "Xenova/gpt-4o"  # or any HF tokenizer

[knowledge]
db_path = "./data/lancedb"
embedding_model = "multilingual-e5-base"
default_collection = "general"
similarity_threshold = 0.65
max_results = 5

[orchestrator]
enabled = false
transport = "websocket"
bind_address = "127.0.0.1:9876"

[search]
tavily_api_key_env = "TAVILY_API_KEY"
brave_api_key_env = "BRAVE_API_KEY"
max_results = 5

[paths]
skills_dir = "./data/skills"
plans_dir = "./data/plans"
outputs_dir = "./data/outputs"
```

---

## Implementation Phases

### Phase 1: Foundation (Core Loop + Config + Errors)
1. Scaffold project with `cargo init`, add all dependencies to `Cargo.toml`
2. Implement `error.rs` — all `AgentError` variants
3. Implement `config.rs` — TOML loading, defaults, validation
4. Implement `agent/state.rs` — ConversationState wrapping llm-cascade types
5. Implement `agent/loop.rs` — core loop with tokio::select!, interrupt channel, tool call processing
6. Implement `agent/mod.rs` — AgentLoop builder, public API
7. Implement `main.rs` — CLI with clap (init, run, config subcommands)
8. Write integration test: send a prompt, get a text response, verify loop exits

### Phase 2: Tool System + Skills
9. Implement `tools/mod.rs` — Tool trait + ToolRegistry
10. Implement `skills/types.rs` — Skill, SkillMetadata structs
11. Implement `skills/parser.rs` — gray_matter SKILL.md parsing
12. Implement `skills/executor.rs` — stdin JSON → stdout JSON execution
13. Implement `skills/mod.rs` — SkillManager (discovery, CRUD)
14. Wire skills into ToolRegistry (each skill becomes a Tool impl)
15. Implement `tools/builtin.rs` — plan management, memory query built-in tools
16. Write integration test: discover skill, execute via LLM tool call

### Phase 3: Memory & Context Management
17. Implement `memory/tokenizer.rs` — TokenCounter via tokenizers crate
18. Implement `memory/compaction.rs` — summarization-based compaction
19. Implement `memory/mod.rs` — MemoryManager orchestration
20. Write integration test: fill context past limit, verify compaction triggers

### Phase 4: Knowledge Base + Search
21. Implement `knowledge/embeddings.rs` — fastembed wrapper with spawn_blocking
22. Implement `knowledge/vectordb.rs` — LanceDB wrapper (Arrow schema, insert, search)
23. Implement `knowledge/mod.rs` — KnowledgeBase facade
24. Implement `tools/search.rs` — Tavily + Brave Search API tools
25. Implement `tools/knowledge_tool.rs` — vector DB query as LLM tool
26. Write integration test: insert entries, query, verify relevance

### Phase 5: Orchestrator Communication
27. Implement `orchestrator/types.rs` — OrchestratorMessage enum
28. Implement `orchestrator/mod.rs` — OrchestratorConnection trait + NoopOrchestrator
29. Implement `orchestrator/websocket.rs` — WebSocket client transport
30. Implement `orchestrator/server.rs` — Localhost WS server for future orchestrator
31. Wire orchestrator into AgentLoop (push updates, recv user replies)
32. Write integration test: start server, push message, verify receipt

### Phase 6: Planning + Polish
33. Implement `planning/types.rs` — Plan, PlanStep, statuses
34. Implement `planning/mod.rs` — PlanManager (create, update, render)
35. Wire planning into AgentLoop (auto-plan for complex tasks, plan approval flow)
36. Add SOUL.md loading + dynamic editing
37. Comprehensive CLI polish, help text, shell completion
38. Final integration test: end-to-end run with tools, compaction, skills

---

## Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Orchestrator transport | **WebSocket first** | Simpler protocol, easier to extend, sufficient for localhost. gRPC slot-in via trait later. |
| fastembed threading | **spawn_blocking** | fastembed is sync; wrap in tokio::task::spawn_blocking to avoid blocking the async runtime. |
| State persistence | **JSON files** per task | Simple, debuggable, human-readable. No extra DB dependency beyond llm-cascade's SQLite. |
| Skill → Tool bridge | **Adapter pattern** | Each discovered skill is wrapped in a struct implementing `Tool` trait. Executor delegates to the skill's binary. |
| Compaction strategy | **Middle-out summarization** | Keep system prompt + recent N messages, summarize everything in between. Preserves fresh context. |
| Arrow vectors | **FixedSizeList\<Float32\>** | Required by LanceDB for vector columns. Dimension = 768 for multilingual-e5-base. |

---

## Risks & Open Questions

1. **llm-cascade SQLite** — `run_cascade()` takes `&rusqlite::Connection` (sync). We need to manage this in the tokio context carefully (likely `spawn_blocking` for the cascade call or keeping the connection on a dedicated thread).

2. **fastembed cold start** — First run downloads the ONNX model (~400MB for e5-base). Need graceful handling (progress indication, pre-download step in CLI).

3. **LanceDB Arrow version alignment** — LanceDB 0.27 pins specific Arrow versions. Must ensure arrow-array/arrow-schema versions match.

4. **Tool call JSON argument parsing** — llm-cascade returns arguments as `String`. Need robust serde_json parsing with good error messages back to the LLM.

5. **Graceful shutdown** — Need a clean shutdown path: finish current tool, save state, close WS connections.

---

## Verification Commands

```bash
# After each phase, run:
cargo fmt
cargo clippy -- -D warnings
cargo test

# Full build check:
cargo build --release

# CLI smoke test:
cargo run -- run --config config.example.toml "Hello, what tools do you have?"
```
