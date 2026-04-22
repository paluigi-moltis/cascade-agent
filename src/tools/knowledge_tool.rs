//! Knowledge query tool — queries a local vector DB for relevant knowledge.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{Tool, ToolResult};

// ---------------------------------------------------------------------------
// KnowledgeProvider trait
// ---------------------------------------------------------------------------
// The knowledge module hasn't been built yet, so we define a minimal trait
// that the real KnowledgeBase will implement.  This keeps this tool decoupled
// from the (future) knowledge implementation.

/// A single knowledge result returned from a vector search.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KnowledgeHit {
    pub text: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, Value>>,
}

/// Trait that any knowledge backend must implement.
#[async_trait]
pub trait KnowledgeProvider: Send + Sync {
    /// Search the knowledge base for chunks similar to `query`.
    async fn query(
        &self,
        query: &str,
        collection: &str,
        limit: usize,
    ) -> Result<Vec<KnowledgeHit>, String>;
}

// ---------------------------------------------------------------------------
// KnowledgeQueryTool
// ---------------------------------------------------------------------------

/// Queries the local vector DB for relevant knowledge.
///
/// Parameters:
///   - `query`      (string, required): the query text.
///   - `collection` (string, optional): the collection to search (default "general").
///   - `limit`      (int, optional):    max results (default 5).
pub struct KnowledgeQueryTool {
    /// The knowledge backend. `None` when no knowledge base is configured.
    provider: Option<Arc<dyn KnowledgeProvider>>,
    default_collection: String,
    default_limit: usize,
}

impl KnowledgeQueryTool {
    /// Create with a knowledge provider backend.
    pub fn new(provider: Arc<dyn KnowledgeProvider>) -> Self {
        Self {
            provider: Some(provider),
            default_collection: "general".to_owned(),
            default_limit: 5,
        }
    }

    /// Create with a knowledge provider and explicit defaults.
    pub fn with_defaults(
        provider: Arc<dyn KnowledgeProvider>,
        default_collection: String,
        default_limit: usize,
    ) -> Self {
        Self {
            provider: Some(provider),
            default_collection,
            default_limit,
        }
    }

    /// Create a *disabled* knowledge tool (no backend configured).
    pub fn disabled() -> Self {
        Self {
            provider: None,
            default_collection: "general".to_owned(),
            default_limit: 5,
        }
    }
}

#[async_trait]
impl Tool for KnowledgeQueryTool {
    fn name(&self) -> &str {
        "knowledge_query"
    }

    fn description(&self) -> &str {
        "Query the local knowledge base for relevant text snippets. Returns passages ranked by semantic similarity."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query text to search for."
                },
                "collection": {
                    "type": "string",
                    "description": "The knowledge collection to search.",
                    "default": self.default_collection
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": self.default_limit
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return ToolResult::err("Missing required parameter 'query'"),
        };

        let collection = args
            .get("collection")
            .and_then(|v| v.as_str())
            .unwrap_or(&self.default_collection);

        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.default_limit as u64) as usize;

        let provider = match &self.provider {
            Some(p) => p,
            None => {
                return ToolResult::err(
                    "Knowledge base is not configured. Set up the knowledge module first.",
                );
            }
        };

        match provider.query(query, collection, limit).await {
            Ok(hits) => {
                let results: Vec<Value> = hits
                    .iter()
                    .map(|h| {
                        let mut obj = json!({
                            "text": h.text,
                            "score": h.score,
                        });
                        if let Some(ref meta) = h.metadata {
                            obj.as_object_mut()
                                .unwrap()
                                .insert("metadata".into(), Value::Object(meta.clone()));
                        }
                        obj
                    })
                    .collect();

                ToolResult::ok(json!({
                    "query": query,
                    "collection": collection,
                    "count": results.len(),
                    "results": results,
                }))
            }
            Err(e) => ToolResult::err(format!("Knowledge query failed: {}", e)),
        }
    }
}

// ---------------------------------------------------------------------------
// No-op knowledge provider for testing
// ---------------------------------------------------------------------------

#[cfg(test)]
struct MockKnowledgeProvider;

#[cfg(test)]
#[async_trait]
impl KnowledgeProvider for MockKnowledgeProvider {
    async fn query(
        &self,
        query: &str,
        _collection: &str,
        limit: usize,
    ) -> Result<Vec<KnowledgeHit>, String> {
        Ok(vec![KnowledgeHit {
            text: format!("Mock result for: {}", query),
            score: 0.99,
            metadata: None,
        }]
        .into_iter()
        .cycle()
        .take(limit)
        .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn knowledge_query_works() {
        let provider = Arc::new(MockKnowledgeProvider);
        let tool = KnowledgeQueryTool::new(provider);
        let result = tool.execute(json!({"query": "rust async"})).await;
        assert_eq!(result.status, super::super::ToolStatus::Success);
        let data = result.data.as_object().unwrap();
        assert_eq!(data["count"].as_u64().unwrap(), 5);
        assert!(data["results"].as_array().unwrap()[0]["text"]
            .as_str()
            .unwrap()
            .contains("Mock result for: rust async"));
    }

    #[tokio::test]
    async fn knowledge_query_disabled() {
        let tool = KnowledgeQueryTool::disabled();
        let result = tool.execute(json!({"query": "test"})).await;
        assert_eq!(result.status, super::super::ToolStatus::Error);
        assert!(result.error.unwrap().contains("not configured"));
    }

    #[tokio::test]
    async fn knowledge_query_custom_params() {
        let provider = Arc::new(MockKnowledgeProvider);
        let tool = KnowledgeQueryTool::with_defaults(provider, "custom".to_owned(), 3);
        let result = tool
            .execute(json!({
                "query": "test",
                "collection": "special",
                "limit": 2
            }))
            .await;
        assert_eq!(result.status, super::super::ToolStatus::Success);
        assert_eq!(result.data["count"].as_u64().unwrap(), 2);
    }
}
