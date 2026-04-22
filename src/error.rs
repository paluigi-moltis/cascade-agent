use thiserror::Error;

/// Library-level error types for cascade-agent.
/// Application code should use anyhow; this module is for the public API surface.

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

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Toml error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("Tokenization error: {0}")]
    TokenizerError(String),
}

pub type Result<T> = std::result::Result<T, AgentError>;
