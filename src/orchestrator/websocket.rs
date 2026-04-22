use super::{types::OrchestratorMessage, OrchestratorConnection};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

pub struct WebSocketOrchestrator {
    outbound_tx: mpsc::Sender<OrchestratorMessage>,
    inbound_rx: mpsc::Receiver<OrchestratorMessage>,
}

impl WebSocketOrchestrator {
    pub async fn connect(url: &str) -> crate::error::Result<Self> {
        let (ws_stream, _response) = connect_async(url).await.map_err(|e| {
            crate::error::AgentError::OrchestratorError(format!("WebSocket connect failed: {}", e))
        })?;

        let (mut ws_sink, mut ws_stream) = ws_stream.split();

        let (outbound_tx, mut outbound_rx) = mpsc::channel::<OrchestratorMessage>(256);
        let (inbound_tx, inbound_rx) = mpsc::channel::<OrchestratorMessage>(256);

        // Task 1: Forward outbound messages to WebSocket
        tokio::spawn(async move {
            while let Some(msg) = outbound_rx.recv().await {
                let json = serde_json::to_string(&msg).unwrap_or_default();
                if ws_sink.send(Message::Text(json.into())).await.is_err() {
                    tracing::error!(target: "orchestrator", "Failed to send message to WebSocket");
                    break;
                }
            }
        });

        // Task 2: Forward WebSocket messages to inbound channel
        tokio::spawn(async move {
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
        });

        Ok(Self {
            outbound_tx,
            inbound_rx,
        })
    }
}

#[async_trait]
impl OrchestratorConnection for WebSocketOrchestrator {
    async fn push(&self, message: OrchestratorMessage) {
        let _ = self.outbound_tx.send(message).await;
    }

    async fn recv(&mut self) -> Option<OrchestratorMessage> {
        self.inbound_rx.recv().await
    }

    fn is_connected(&self) -> bool {
        !self.outbound_tx.is_closed()
    }
}
