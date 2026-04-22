//! Memory & Context Management.
//!
//! Provides token counting, context compaction (summarization), and orchestration
//! via [`MemoryManager`].

pub mod compaction;
pub mod state;
pub mod tokenizer;
pub mod types;

use std::sync::Arc;

use crate::config::MemorySettings;
use crate::error::Result;
use crate::memory::compaction::Compactor;
use crate::memory::state::ConversationStateAccess;
use crate::memory::tokenizer::TokenCounter;
use crate::memory::types::CompactionReport;

/// Orchestrates token counting and context compaction for conversation management.
///
/// `MemoryManager` is the main entry point for memory operations. It checks whether
/// compaction is needed and delegates to the [`Compactor`] when the conversation
/// exceeds configured token limits.
pub struct MemoryManager {
    token_counter: TokenCounter,
    compactor: Compactor,
    context_token_limit: usize,
}

impl MemoryManager {
    /// Create a new memory manager.
    ///
    /// # Arguments
    /// * `settings` - Memory configuration (token limits, compaction ratio, etc.)
    /// * `cascade_config` - llm-cascade `AppConfig` for summarization inference
    /// * `db_conn` - SQLite connection for llm-cascade cooldown/logging
    pub fn new(
        settings: &MemorySettings,
        cascade_config: Arc<llm_cascade::AppConfig>,
        db_conn: rusqlite::Connection,
    ) -> Result<Self> {
        let token_counter = TokenCounter::new(&settings.tokenizer_model)?;
        let compactor = Compactor::new(settings, cascade_config, db_conn);

        Ok(Self {
            token_counter,
            compactor,
            context_token_limit: settings.context_token_limit,
        })
    }

    /// Count the total tokens in a conversation state.
    pub fn count_tokens(&self, state: &impl ConversationStateAccess) -> usize {
        self.token_counter.count_messages(state.messages())
    }

    /// Check whether compaction should be triggered based on current token count.
    ///
    /// Compaction is recommended when tokens exceed the configured limit.
    pub fn should_compact(&self, token_count: usize) -> bool {
        token_count > self.context_token_limit
    }

    /// Attempt to compact the conversation if it exceeds the token limit.
    ///
    /// If compaction is not needed, returns a no-op report.
    /// If compaction fails (e.g., summarization LLM error), returns the error
    /// without modifying the conversation state.
    pub async fn compact(
        &self,
        state: &mut impl ConversationStateAccess,
    ) -> Result<CompactionReport> {
        let tokens = self.count_tokens(state);

        if !self.should_compact(tokens) {
            return Ok(CompactionReport {
                messages_before: state.messages().len(),
                messages_after: state.messages().len(),
                tokens_before: tokens,
                tokens_after: tokens,
            });
        }

        self.compactor
            .compact(state, &self.token_counter, self.context_token_limit)
            .await
    }

    /// Returns the configured context token limit.
    pub fn token_limit(&self) -> usize {
        self.context_token_limit
    }

    /// Returns a reference to the underlying token counter.
    pub fn token_counter(&self) -> &TokenCounter {
        &self.token_counter
    }
}
