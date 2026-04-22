//! Fastembed-based text embedding with thread-safe Mutex wrapper.

use std::sync::Mutex;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

use crate::error::{AgentError, Result};

/// Maps a human-readable model name to the fastembed enum.
fn model_from_name(name: &str) -> Result<EmbeddingModel> {
    match name {
        "all-MiniLM-L6-v2" => Ok(EmbeddingModel::AllMiniLML6V2),
        "all-MiniLM-L6-v2-q" => Ok(EmbeddingModel::AllMiniLML6V2Q),
        "all-MiniLM-L12-v2" => Ok(EmbeddingModel::AllMiniLML12V2),
        "all-mpnet-base-v2" => Ok(EmbeddingModel::AllMpnetBaseV2),
        "bge-base-en-v1.5" => Ok(EmbeddingModel::BGEBaseENV15),
        "bge-base-en-v1.5-q" => Ok(EmbeddingModel::BGEBaseENV15Q),
        "bge-large-en-v1.5" => Ok(EmbeddingModel::BGELargeENV15),
        "bge-large-en-v1.5-q" => Ok(EmbeddingModel::BGELargeENV15Q),
        "bge-small-en-v1.5" => Ok(EmbeddingModel::BGESmallENV15),
        "bge-small-en-v1.5-q" => Ok(EmbeddingModel::BGESmallENV15Q),
        "nomic-embed-text-v1" => Ok(EmbeddingModel::NomicEmbedTextV1),
        "nomic-embed-text-v1.5" => Ok(EmbeddingModel::NomicEmbedTextV15),
        "nomic-embed-text-v1.5-q" => Ok(EmbeddingModel::NomicEmbedTextV15Q),
        "paraphrase-multilingual-MiniLM-L12-v2" => Ok(EmbeddingModel::ParaphraseMLMiniLML12V2),
        "paraphrase-multilingual-mpnet-base-v2" => Ok(EmbeddingModel::ParaphraseMLMpnetBaseV2),
        "bgem3" | "BAAI/bgem3" => Ok(EmbeddingModel::BGEM3),
        "multilingual-e5-small" | "intfloat/multilingual-e5-small" => {
            Ok(EmbeddingModel::MultilingualE5Small)
        }
        "multilingual-e5-base" | "intfloat/multilingual-e5-base" => {
            Ok(EmbeddingModel::MultilingualE5Base)
        }
        "multilingual-e5-large" | "intfloat/multilingual-e5-large" => {
            Ok(EmbeddingModel::MultilingualE5Large)
        }
        "mxbai-embed-large-v1" => Ok(EmbeddingModel::MxbaiEmbedLargeV1),
        "mxbai-embed-large-v1-q" => Ok(EmbeddingModel::MxbaiEmbedLargeV1Q),
        "gte-base-en-v1.5" => Ok(EmbeddingModel::GTEBaseENV15),
        _ => Err(AgentError::EmbeddingError(format!(
            "Unknown embedding model: '{}'. \
             See docs for supported models.",
            name
        ))),
    }
}

/// Thread-safe embedding wrapper using fastembed.
///
/// The inner `TextEmbedding` is wrapped in a `Mutex` because `embed(&mut self)`
/// requires mutable access.
pub struct Embedder {
    model: Mutex<TextEmbedding>,
    dimension: usize,
}

impl Embedder {
    /// Create a new embedder, downloading the model if necessary.
    pub fn new(model_name: &str) -> Result<Self> {
        let model_enum = model_from_name(model_name)?;
        let opts = TextInitOptions::new(model_enum).with_show_download_progress(true);

        let mut text_embedding = TextEmbedding::try_new(opts)
            .map_err(|e| AgentError::EmbeddingError(format!("Failed to load model: {}", e)))?;

        // Determine dimension by embedding a dummy text.
        let dimension = {
            let mut te = text_embedding;
            let result = te
                .embed(["dim_probe"], None)
                .map_err(|e| AgentError::EmbeddingError(format!("Probe embed failed: {}", e)))?;
            let dim = result.into_iter().next().unwrap().len();
            text_embedding = te;
            dim
        };

        Ok(Self {
            model: Mutex::new(text_embedding),
            dimension,
        })
    }

    /// Embed a single query string (prepends "query: " prefix for E5 models).
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let prefixed = format!("query: {}", text);
        let mut guard = self
            .model
            .lock()
            .map_err(|e| AgentError::EmbeddingError(format!("Embedder lock poisoned: {}", e)))?;
        let results = guard
            .embed([&prefixed], None)
            .map_err(|e| AgentError::EmbeddingError(format!("Query embed failed: {}", e)))?;
        Ok(results.into_iter().next().unwrap())
    }

    /// Embed a single passage string (prepends "passage: " prefix for E5 models).
    pub fn embed_passage(&self, text: &str) -> Result<Vec<f32>> {
        let prefixed = format!("passage: {}", text);
        let mut guard = self
            .model
            .lock()
            .map_err(|e| AgentError::EmbeddingError(format!("Embedder lock poisoned: {}", e)))?;
        let results = guard
            .embed([&prefixed], None)
            .map_err(|e| AgentError::EmbeddingError(format!("Passage embed failed: {}", e)))?;
        Ok(results.into_iter().next().unwrap())
    }

    /// Embed a batch of passage strings (prepends "passage: " prefix).
    pub fn embed_batch_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let prefixed: Vec<String> = texts.iter().map(|t| format!("passage: {}", t)).collect();
        let mut guard = self
            .model
            .lock()
            .map_err(|e| AgentError::EmbeddingError(format!("Embedder lock poisoned: {}", e)))?;
        guard
            .embed(&prefixed, None)
            .map_err(|e| AgentError::EmbeddingError(format!("Batch embed failed: {}", e)))
    }

    /// Returns the embedding dimensionality.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}
