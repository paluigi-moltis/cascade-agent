use super::types::OrchestratorMessage;
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio::sync::{broadcast, mpsc};
use tokio_tungstenite::accept_async;

/// A simple WebSocket server for the orchestrator to connect to.
/// The agent runs this server; a future orchestrator UI connects as a client.
pub struct OrchestratorServer {
    bind_address: String,
    // broadcast channel: server sends to all connected orchestrator clients
    broadcast_tx: broadcast::Sender<OrchestratorMessage>,
    // channel for server to receive messages from orchestrator clients
    inbound_tx: mpsc::Sender<OrchestratorMessage>,
}

impl OrchestratorServer {
    pub fn new(
        bind_address: &str,
    ) -> (
        Self,
        broadcast::Sender<OrchestratorMessage>,
        mpsc::Receiver<OrchestratorMessage>,
    ) {
        let (broadcast_tx, _) = broadcast::channel(256);
        let (inbound_tx, inbound_rx) = mpsc::channel(256);
        let server = Self {
            bind_address: bind_address.to_string(),
            broadcast_tx: broadcast_tx.clone(),
            inbound_tx,
        };
        (server, broadcast_tx, inbound_rx)
    }

    pub async fn run(self) -> crate::error::Result<()> {
        let listener = TcpListener::bind(&self.bind_address).await.map_err(|e| {
            crate::error::AgentError::OrchestratorError(format!("Failed to bind: {}", e))
        })?;

        tracing::info!(target: "orchestrator", "Server listening on {}", self.bind_address);

        while let Ok((stream, addr)) = listener.accept().await {
            let broadcast_tx = self.broadcast_tx.clone();
            let inbound_tx = self.inbound_tx.clone();

            tokio::spawn(async move {
                let ws_stream = match accept_async(stream).await {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::error!(target: "orchestrator", "Accept failed from {}: {}", addr, e);
                        return;
                    }
                };

                tracing::info!(target: "orchestrator", "Orchestrator connected from {}", addr);
                let (mut ws_sink, mut ws_stream) = ws_stream.split();

                // Subscribe to broadcast
                let mut rx = broadcast_tx.subscribe();

                // Forward broadcast → WebSocket
                let ws_to_client = async {
                    while let Ok(msg) = rx.recv().await {
                        let json = serde_json::to_string(&msg).unwrap_or_default();
                        if ws_sink
                            .send(tokio_tungstenite::tungstenite::protocol::Message::Text(
                                json.into(),
                            ))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                };

                // Forward WebSocket → inbound
                let client_to_server = async {
                    while let Some(Ok(msg)) = ws_stream.next().await {
                        if msg.is_text() {
                            let text = msg.to_text().unwrap_or_default();
                            if let Ok(parsed) = serde_json::from_str::<OrchestratorMessage>(text) {
                                if inbound_tx.send(parsed).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                };

                tokio::select! {
                    _ = ws_to_client => {},
                    _ = client_to_server => {},
                }

                tracing::info!(target: "orchestrator", "Orchestrator disconnected from {}", addr);
            });
        }

        Ok(())
    }
}
