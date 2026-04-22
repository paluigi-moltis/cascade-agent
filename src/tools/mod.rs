//! Tool system: trait definition, registry, and built-in tools.

pub mod builtin;
pub mod knowledge_tool;
pub mod search;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{AgentError, Result};

// ---------------------------------------------------------------------------
// ToolStatus / ToolResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolStatus {
    Success,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub status: ToolStatus,
    pub data: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ToolResult {
    /// Convenience constructor for a successful result with arbitrary data.
    pub fn ok(data: serde_json::Value) -> Self {
        Self {
            status: ToolStatus::Success,
            data,
            error: None,
        }
    }

    /// Convenience constructor for a successful result whose data is a string.
    pub fn ok_string(s: impl Into<String>) -> Self {
        Self::ok(serde_json::Value::String(s.into()))
    }

    /// Convenience constructor for an error result.
    pub fn err(msg: impl Into<String>) -> Self {
        Self {
            status: ToolStatus::Error,
            data: serde_json::Value::Null,
            error: Some(msg.into()),
        }
    }

    /// Serialize to a compact JSON string suitable for feeding back to the LLM.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| format!("{:?}", self))
    }

    /// Return the status as a lowercase string ("success" or "error").
    pub fn status_str(&self) -> &str {
        match self.status {
            ToolStatus::Success => "success",
            ToolStatus::Error => "error",
        }
    }
}

// ---------------------------------------------------------------------------
// Tool trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait Tool: Send + Sync {
    /// Machine-readable tool name (e.g. "echo").
    fn name(&self) -> &str;

    /// Human-readable description the LLM uses to decide when to call this tool.
    fn description(&self) -> &str;

    /// JSON Schema describing the parameters this tool accepts.
    fn parameters_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given JSON arguments.
    async fn execute(&self, args: serde_json::Value) -> ToolResult;

    /// Build a `llm_cascade::ToolDefinition` that can be passed to the model.
    fn to_definition(&self) -> llm_cascade::ToolDefinition {
        llm_cascade::ToolDefinition {
            name: self.name().to_owned(),
            description: self.description().to_owned(),
            parameters: self.parameters_schema(),
        }
    }
}

// ---------------------------------------------------------------------------
// ToolRegistry
// ---------------------------------------------------------------------------

/// A thread-safe registry that maps tool names to their implementations.
pub struct ToolRegistry {
    tools: Arc<std::sync::Mutex<HashMap<String, Arc<dyn Tool>>>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tools: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Register a tool. Overwrites any previously registered tool with the same name.
    pub fn register<T: Tool + 'static>(&self, tool: T) {
        self.tools
            .lock()
            .unwrap()
            .insert(tool.name().to_owned(), Arc::new(tool));
    }

    /// Register a tool that is already boxed as `Arc<dyn Tool>`.
    pub fn register_arc(&self, tool: Arc<dyn Tool>) {
        self.tools
            .lock()
            .unwrap()
            .insert(tool.name().to_owned(), tool);
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.lock().unwrap().get(name).cloned()
    }

    /// Return all registered tool names.
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.lock().unwrap().keys().cloned().collect()
    }

    /// Return all registered tools as `Arc<dyn Tool>`.
    pub fn all_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.lock().unwrap().values().cloned().collect()
    }

    /// Build a list of `ToolDefinition` for every registered tool.
    pub fn tool_definitions(&self) -> Vec<llm_cascade::ToolDefinition> {
        self.tools
            .lock()
            .unwrap()
            .values()
            .map(|t| t.to_definition())
            .collect()
    }

    /// Convenience: execute a tool by name with the given JSON args.
    pub async fn execute(&self, name: &str, args: serde_json::Value) -> Result<ToolResult> {
        let tool = self.get(name).ok_or_else(|| AgentError::ToolFailed {
            tool: name.to_owned(),
            reason: format!("Tool '{}' is not registered", name),
        })?;
        Ok(tool.execute(args).await)
    }

    /// Create a `ListToolsTool` that references this registry and register it.
    pub fn register_list_tools(&self) {
        let names = self.tool_names();
        let tools = self.all_tools();
        self.register(builtin::ListToolsTool::new(names, tools));
    }
}
