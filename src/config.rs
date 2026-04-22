use serde::Deserialize;
use std::path::Path;

/// Top-level agent configuration loaded from TOML.
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
    pub cascade_name: String,
    pub cascade_config_path: String,
    #[serde(default = "default_max_tool_rounds")]
    pub max_tool_rounds: usize,
    pub soul_md_path: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MemorySettings {
    pub context_token_limit: usize,
    #[serde(default = "default_compaction_ratio")]
    pub compaction_target_ratio: f64,
    pub summarization_cascade: String,
    pub tokenizer_model: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct KnowledgeSettings {
    pub db_path: String,
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
    #[serde(default = "default_collection")]
    pub default_collection: String,
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
    #[serde(default = "default_max_results")]
    pub max_results: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OrchestratorSettings {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_transport")]
    pub transport: String,
    #[serde(default = "default_bind_address")]
    pub bind_address: String,
    pub connect_url: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SearchSettings {
    #[serde(default = "default_tavily_env")]
    pub tavily_api_key_env: String,
    #[serde(default = "default_brave_env")]
    pub brave_api_key_env: String,
    #[serde(default = "default_max_results")]
    pub max_results: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PathSettings {
    pub skills_dir: String,
    pub plans_dir: String,
    pub outputs_dir: String,
}

fn default_max_tool_rounds() -> usize {
    25
}
fn default_compaction_ratio() -> f64 {
    0.6
}
fn default_embedding_model() -> String {
    "multilingual-e5-base".into()
}
fn default_collection() -> String {
    "general".into()
}
fn default_similarity_threshold() -> f32 {
    0.65
}
fn default_max_results() -> usize {
    5
}
fn default_transport() -> String {
    "websocket".into()
}
fn default_bind_address() -> String {
    "127.0.0.1:9876".into()
}
fn default_tavily_env() -> String {
    "TAVILY_API_KEY".into()
}
fn default_brave_env() -> String {
    "BRAVE_API_KEY".into()
}

impl AgentConfig {
    /// Load configuration from a TOML file path.
    /// Expands `~` in paths.
    pub fn load(path: &Path) -> crate::error::Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            crate::error::AgentError::ConfigError(format!(
                "Cannot read config file {}: {}",
                path.display(),
                e
            ))
        })?;

        let mut config: AgentConfig = toml::from_str(&content).map_err(|e| {
            crate::error::AgentError::ConfigError(format!("Failed to parse config: {}", e))
        })?;

        // Expand ~ in paths
        expand_home(&mut config.agent.cascade_config_path);
        expand_home(&mut config.agent.soul_md_path);
        expand_home(&mut config.knowledge.db_path);
        expand_home(&mut config.paths.skills_dir);
        expand_home(&mut config.paths.plans_dir);
        expand_home(&mut config.paths.outputs_dir);

        Ok(config)
    }
}

fn expand_home(s: &mut String) {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            *s = format!("{}/{}", home.display(), rest);
        }
    }
}
