//! LanceDB-backed vector store for knowledge retrieval.

use std::sync::Arc;

use arrow_array::{
    types::Float32Type, Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::connect;
use lancedb::database::CreateTableMode;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};

use super::embeddings::Embedder;
use crate::error::{AgentError, Result};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single entry to store in the vector database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    pub text: String,
    pub source: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
    /// Unix timestamp (seconds since epoch). Default: now.
    #[serde(default = "default_timestamp")]
    pub timestamp: i64,
}

fn default_timestamp() -> i64 {
    chrono::Utc::now().timestamp()
}

/// A single result returned from a vector search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub text: String,
    pub source: String,
    pub score: f32,
    pub metadata: serde_json::Value,
    pub timestamp: i64,
}

// ---------------------------------------------------------------------------
// VectorStore
// ---------------------------------------------------------------------------

/// LanceDB-backed vector store.
pub struct VectorStore {
    conn: lancedb::Connection,
    embedder: Arc<Embedder>,
    default_collection: String,
    dim: usize,
}

impl VectorStore {
    /// Connect to (or create) a LanceDB database at `db_path`.
    pub async fn new(
        db_path: &str,
        embedder: Arc<Embedder>,
        default_collection: &str,
    ) -> Result<Self> {
        let conn = connect(db_path).execute().await.map_err(|e| {
            AgentError::KnowledgeError(format!(
                "Failed to connect to LanceDB at '{}': {}",
                db_path, e
            ))
        })?;
        let dim = embedder.dimension();
        Ok(Self {
            conn,
            embedder,
            default_collection: default_collection.to_owned(),
            dim,
        })
    }

    /// Build the Arrow schema used for knowledge tables.
    fn schema(dim: i32) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
            Field::new("text", DataType::Utf8, false),
            Field::new("source", DataType::Utf8, false),
            Field::new("metadata_json", DataType::Utf8, true),
            Field::new("timestamp", DataType::UInt64, false),
        ]))
    }

    /// Create a new collection (table). Uses `exist_ok` mode.
    pub async fn create_collection(&self, name: &str) -> Result<()> {
        let schema = Self::schema(self.dim as i32);
        self.conn
            .create_empty_table(name, schema)
            .mode(CreateTableMode::exist_ok(|req| req))
            .execute()
            .await
            .map_err(|e| {
                AgentError::KnowledgeError(format!("Failed to create collection '{}': {}", name, e))
            })?;
        Ok(())
    }

    /// Insert entries into a collection.
    pub async fn insert(&self, collection: &str, entries: Vec<KnowledgeEntry>) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        // Ensure the collection exists.
        self.create_collection(collection).await?;

        // Embed passages (blocking, offloaded to blocking thread pool).
        let texts: Vec<String> = entries.iter().map(|e| e.text.clone()).collect();
        let embedder = self.embedder.clone();
        let embeddings = tokio::task::spawn_blocking(move || embedder.embed_batch_passages(&texts))
            .await
            .map_err(|e| AgentError::KnowledgeError(format!("Embedding task panicked: {}", e)))?
            .map_err(|e| AgentError::KnowledgeError(format!("Embedding failed: {}", e)))?;

        // Build Arrow columns.
        let n = entries.len();
        let dim = self.dim as i32;

        let ids: Vec<String> = (0..n).map(|_| uuid::Uuid::new_v4().to_string()).collect();
        let id_array = StringArray::from(ids);

        let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            embeddings
                .iter()
                .map(|emb| Some(emb.iter().map(|&v| Some(v)))),
            dim,
        );

        let text_array =
            StringArray::from(entries.iter().map(|e| e.text.as_str()).collect::<Vec<_>>());
        let source_array = StringArray::from(
            entries
                .iter()
                .map(|e| e.source.as_str())
                .collect::<Vec<_>>(),
        );
        let metadata_json_array = StringArray::from(
            entries
                .iter()
                .map(|e| {
                    if e.metadata.is_null() {
                        None
                    } else {
                        Some(serde_json::to_string(&e.metadata).unwrap_or_default())
                    }
                })
                .collect::<Vec<_>>(),
        );
        let timestamp_array = UInt64Array::from(
            entries
                .iter()
                .map(|e| e.timestamp as u64)
                .collect::<Vec<_>>(),
        );

        let schema = Self::schema(dim);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(id_array),
                Arc::new(vector_array),
                Arc::new(text_array),
                Arc::new(source_array),
                Arc::new(metadata_json_array),
                Arc::new(timestamp_array),
            ],
        )
        .map_err(|e| AgentError::KnowledgeError(format!("Failed to build RecordBatch: {}", e)))?;

        // Open table and add.
        let table = self
            .conn
            .open_table(collection)
            .execute()
            .await
            .map_err(|e| {
                AgentError::KnowledgeError(format!(
                    "Failed to open collection '{}': {}",
                    collection, e
                ))
            })?;

        table.add(batch).execute().await.map_err(|e| {
            AgentError::KnowledgeError(format!(
                "Failed to insert into collection '{}': {}",
                collection, e
            ))
        })?;

        Ok(())
    }

    /// Search a collection for entries similar to `query`.
    pub async fn search(
        &self,
        collection: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        // Embed the query (blocking, offloaded).
        let embedder = self.embedder.clone();
        let query_owned = query.to_owned();
        let query_vec = tokio::task::spawn_blocking(move || embedder.embed_query(&query_owned))
            .await
            .map_err(|e| AgentError::KnowledgeError(format!("Query embed panicked: {}", e)))?
            .map_err(|e| AgentError::KnowledgeError(format!("Query embed failed: {}", e)))?;

        // Open table (graceful if missing).
        let table = match self.conn.open_table(collection).execute().await {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()),
        };

        // Execute the vector search.
        let results: Vec<RecordBatch> = table
            .query()
            .nearest_to(query_vec.as_slice())
            .map_err(|e| AgentError::KnowledgeError(format!("Failed to build query: {}", e)))?
            .limit(limit)
            .execute()
            .await
            .map_err(|e| AgentError::KnowledgeError(format!("Search failed: {}", e)))?
            .try_collect()
            .await
            .map_err(|e| AgentError::KnowledgeError(format!("Failed to collect results: {}", e)))?;

        // Parse results.
        let mut search_results = Vec::new();
        for batch in &results {
            let schema = batch.schema();
            let text_idx = schema.index_of("text").unwrap_or(0);
            let source_idx = schema.index_of("source").unwrap_or(1);
            let metadata_idx = schema.index_of("metadata_json").ok();
            let timestamp_idx = schema.index_of("timestamp").unwrap_or(5);

            // The distance column is _distance.
            let distance_idx = schema.index_of("_distance").ok();

            for row in 0..batch.num_rows() {
                let text_col = batch
                    .column(text_idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let source_col = batch
                    .column(source_idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let ts_col = batch
                    .column(timestamp_idx)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();

                let text = text_col.value(row).to_owned();
                let source = source_col.value(row).to_owned();
                let timestamp = ts_col.value(row) as i64;

                let metadata = if let Some(mi) = metadata_idx {
                    let meta_col = batch
                        .column(mi)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap();
                    if meta_col.is_null(row) {
                        serde_json::Value::Null
                    } else {
                        serde_json::from_str(meta_col.value(row)).unwrap_or(serde_json::Value::Null)
                    }
                } else {
                    serde_json::Value::Null
                };

                let score = if let Some(di) = distance_idx {
                    let dist_col = batch
                        .column(di)
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .unwrap();
                    // Convert distance to a similarity-like score (lower distance = higher score).
                    // Use 1.0 / (1.0 + distance) for a simple mapping.
                    let dist = dist_col.value(row);
                    1.0 / (1.0 + dist)
                } else {
                    1.0
                };

                search_results.push(SearchResult {
                    text,
                    source,
                    score,
                    metadata,
                    timestamp,
                });
            }
        }

        Ok(search_results)
    }

    /// List all collection (table) names.
    pub async fn list_collections(&self) -> Result<Vec<String>> {
        let names = self.conn.table_names().execute().await.map_err(|e| {
            AgentError::KnowledgeError(format!("Failed to list collections: {}", e))
        })?;
        Ok(names)
    }

    /// Returns the default collection name.
    pub fn default_collection(&self) -> &str {
        &self.default_collection
    }
}
