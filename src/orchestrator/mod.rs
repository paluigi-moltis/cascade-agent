pub mod server;
pub mod types;
pub mod websocket;

use crate::config::OrchestratorSettings;
use crate::error::Result;
use async_trait::async_trait;
use types::OrchestratorMessage;

/// Trait for bidirectional orchestrator communication.
#[async_trait]
pub trait OrchestratorConnection: Send + Sync {
    /// Push a message from agent to orchestrator (fire-and-forget)
    async fn push(&self, message: OrchestratorMessage);

    /// Receive the next message from orchestrator (blocking await)
    async fn recv(&mut self) -> Option<OrchestratorMessage>;

    /// Check if connected
    fn is_connected(&self) -> bool;
}

/// No-op implementation for when orchestrator is disabled.
pub struct NoopOrchestrator;

#[async_trait]
impl OrchestratorConnection for NoopOrchestrator {
    async fn push(&self, message: OrchestratorMessage) {
        tracing::debug!(target: "orchestrator", "Noop push: {:?}", message);
    }
    async fn recv(&mut self) -> Option<OrchestratorMessage> {
        // Never receives anything — return None immediately
        std::future::pending().await
    }
    fn is_connected(&self) -> bool {
        false
    }
}

/// Factory function to create the right orchestrator transport.
pub fn create_orchestrator(
    config: &OrchestratorSettings,
) -> Result<Box<dyn OrchestratorConnection>> {
    if !config.enabled {
        return Ok(Box::new(NoopOrchestrator));
    }
    match config.transport.as_str() {
        "websocket" => {
            // Return a WebSocket orchestrator
            // For now, return Noop — the WebSocket impl needs async init
            // The actual creation will happen in AgentLoop::new() which is async
            Ok(Box::new(NoopOrchestrator))
        }
        other => Err(crate::error::AgentError::OrchestratorError(format!(
            "Unknown transport: {}",
            other
        ))),
    }
}

/// Async factory for transports that need initialization (like WebSocket).
pub async fn create_orchestrator_async(
    config: &OrchestratorSettings,
) -> Result<Box<dyn OrchestratorConnection>> {
    if !config.enabled {
        return Ok(Box::new(NoopOrchestrator));
    }
    match config.transport.as_str() {
        "websocket" => {
            let url = config.connect_url.as_deref().ok_or_else(|| {
                crate::error::AgentError::OrchestratorError(
                    "WebSocket connect_url required when orchestrator enabled".into(),
                )
            })?;
            let ws = websocket::WebSocketOrchestrator::connect(url).await?;
            Ok(Box::new(ws))
        }
        other => Err(crate::error::AgentError::OrchestratorError(format!(
            "Unknown transport: {}",
            other
        ))),
    }
}
