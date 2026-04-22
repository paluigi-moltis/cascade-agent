//! Knowledge base facade — embeds, stores, and retrieves knowledge.

pub mod embeddings;
pub mod vectordb;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::config::KnowledgeSettings;
use crate::error::Result;
use crate::tools::knowledge_tool::{KnowledgeHit, KnowledgeProvider};

use embeddings::Embedder;
use vectordb::{KnowledgeEntry, SearchResult, VectorStore};

// ---------------------------------------------------------------------------
// KnowledgeBase
// ---------------------------------------------------------------------------

/// High-level knowledge base that combines embedding and vector storage.
pub struct KnowledgeBase {
    store: VectorStore,
    config: KnowledgeSettings,
}

impl KnowledgeBase {
    /// Create a new knowledge base from configuration.
    ///
    /// Downloads the embedding model (if needed) and connects to LanceDB.
    pub async fn new(config: &KnowledgeSettings) -> Result<Self> {
        // Create embedder (blocking model download + load).
        let model_name = config.embedding_model.clone();
        let embedder = tokio::task::spawn_blocking(move || Embedder::new(&model_name))
            .await
            .map_err(|e| {
                crate::error::AgentError::KnowledgeError(format!("Embedder init panicked: {}", e))
            })??;

        let embedder = Arc::new(embedder);
        let store = VectorStore::new(&config.db_path, embedder, &config.default_collection).await?;

        // Ensure the default collection exists.
        store.create_collection(&config.default_collection).await?;

        Ok(Self {
            store,
            config: config.clone(),
        })
    }

    /// Query the default collection.
    pub async fn query_existing(&self, query: &str) -> Result<Vec<SearchResult>> {
        let limit = self.config.max_results;
        self.store
            .search(&self.config.default_collection, query, limit)
            .await
    }

    /// Store entries into a collection.
    pub async fn store_results(
        &self,
        collection: &str,
        entries: Vec<KnowledgeEntry>,
    ) -> Result<()> {
        self.store.insert(collection, entries).await
    }

    /// Create a new collection.
    pub async fn create_collection(&self, name: &str) -> Result<()> {
        self.store.create_collection(name).await
    }

    /// List all collections.
    pub async fn list_collections(&self) -> Result<Vec<String>> {
        self.store.list_collections().await
    }
}

// ---------------------------------------------------------------------------
// KnowledgeProvider implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl KnowledgeProvider for KnowledgeBase {
    async fn query(
        &self,
        query: &str,
        collection: &str,
        limit: usize,
    ) -> std::result::Result<Vec<KnowledgeHit>, String> {
        let results = self
            .store
            .search(collection, query, limit)
            .await
            .map_err(|e| format!("Knowledge query failed: {}", e))?;

        let hits: Vec<KnowledgeHit> = results
            .into_iter()
            .filter(|r| {
                // Apply similarity threshold from config.
                r.score >= self.config.similarity_threshold
            })
            .map(|r| KnowledgeHit {
                text: r.text,
                score: r.score,
                metadata: if r.metadata.is_null() {
                    None
                } else if let Value::Object(map) = r.metadata {
                    Some(map)
                } else {
                    let mut map = Map::new();
                    map.insert("value".into(), r.metadata);
                    Some(map)
                },
            })
            .collect();

        Ok(hits)
    }
}
