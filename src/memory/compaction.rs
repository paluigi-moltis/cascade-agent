use std::sync::{Arc, Mutex};

use llm_cascade::{Conversation, Message, MessageRole};

use crate::config::MemorySettings;
use crate::error::{AgentError, Result};
use crate::memory::state::ConversationStateAccess;
use crate::memory::tokenizer::TokenCounter;
use crate::memory::types::CompactionReport;

/// Context compactor that summarizes older messages to stay within token limits.
///
/// Strategy:
/// 1. Always keep the system prompt (first message).
/// 2. Keep the most recent N messages that fit within `token_limit * target_ratio`.
/// 3. Summarize all messages in between into a single system message.
/// 4. If summarization fails, the original messages are preserved (no data loss).
pub struct Compactor {
    cascade_name: String,
    cascade_config: Arc<llm_cascade::AppConfig>,
    db_conn: Mutex<rusqlite::Connection>,
    target_ratio: f64,
}

impl Compactor {
    /// Create a new compactor.
    ///
    /// # Arguments
    /// * `settings` - Memory settings containing the summarization cascade name and target ratio
    /// * `cascade_config` - The llm-cascade `AppConfig` for LLM inference
    /// * `db_conn` - SQLite connection for llm-cascade's cooldown/logging
    pub fn new(
        settings: &MemorySettings,
        cascade_config: Arc<llm_cascade::AppConfig>,
        db_conn: rusqlite::Connection,
    ) -> Self {
        Self {
            cascade_name: settings.summarization_cascade.clone(),
            cascade_config,
            db_conn: Mutex::new(db_conn),
            target_ratio: settings.compaction_target_ratio,
        }
    }

    /// Compact conversation state if it exceeds the token limit.
    ///
    /// Returns a `CompactionReport` with before/after statistics.
    /// If compaction is not needed, returns a report with no changes.
    /// If summarization fails, returns an error and leaves messages unchanged.
    pub async fn compact(
        &self,
        state: &mut impl ConversationStateAccess,
        token_counter: &TokenCounter,
        token_limit: usize,
    ) -> Result<CompactionReport> {
        let messages = state.messages();
        let messages_before = messages.len();
        let tokens_before = token_counter.count_messages(messages);

        let target_tokens = (token_limit as f64 * self.target_ratio) as usize;

        // If we're already under target, no compaction needed
        if tokens_before <= target_tokens {
            return Ok(CompactionReport {
                messages_before,
                messages_after: messages_before,
                tokens_before,
                tokens_after: tokens_before,
            });
        }

        // Need at least 3 messages to compact: system prompt + at least 1 middle + at least 1 recent
        if messages.len() < 3 {
            return Ok(CompactionReport {
                messages_before,
                messages_after: messages_before,
                tokens_before,
                tokens_after: tokens_before,
            });
        }

        // Strategy: keep system prompt (index 0), find how many recent messages fit,
        // summarize everything in between.
        let system_prompt = state.system_prompt().to_string();
        // Estimate tokens for system prompt: content + role overhead ("system: " = 2) + separator (1) + framing (3)
        let system_tokens = token_counter.count_text(&system_prompt) + 2 + 1 + 3;

        // Start from the end and count backwards to find how many recent messages we can keep
        let available_for_recent = target_tokens.saturating_sub(system_tokens);
        let mut recent_count = 0;
        let mut recent_tokens = 0;

        for msg in messages.iter().rev() {
            let msg_tokens = token_counter.count_text(&msg.content) + 1; // +1 for separator
            if recent_tokens + msg_tokens > available_for_recent && recent_count > 0 {
                break;
            }
            recent_tokens += msg_tokens;
            recent_count += 1;
        }

        // Ensure we keep at least 1 recent message
        if recent_count == 0 {
            recent_count = 1;
        }

        // Messages to summarize: everything after system prompt up to the recent tail
        let summarize_start = 1; // skip system prompt
        let summarize_end = messages.len().saturating_sub(recent_count);

        if summarize_start >= summarize_end {
            // Nothing to summarize
            return Ok(CompactionReport {
                messages_before,
                messages_after: messages_before,
                tokens_before,
                tokens_after: tokens_before,
            });
        }

        let messages_to_summarize = &messages[summarize_start..summarize_end];

        // Build the summarization prompt
        let conversation_text = format_messages_for_summarization(messages_to_summarize);

        let summarization_prompt = Message::system(format!(
            "Summarize the following conversation excerpt concisely. \
             Preserve:\n\
             - Key facts and information discussed\n\
             - Decisions that were made\n\
             - User preferences and requirements\n\
             - Any critical context that would be needed to continue the conversation\n\
             - File paths, code snippets, or tool results that are referenced\n\n\
             Write the summary in a clear, structured format. Be concise but complete.\n\n\
             [Conversation to summarize]:\n\
             {}",
            conversation_text
        ));

        // Call the LLM to summarize
        let summary = self.call_summarization(summarization_prompt).await?;

        let compacted_message = Message {
            role: MessageRole::System,
            content: format!(
                "[Compacted context from earlier in the conversation]:\n{}",
                summary
            ),
            tool_call_id: None,
        };

        // Rebuild messages: system prompt + summary + recent messages
        let mut new_messages = Vec::with_capacity(2 + recent_count);
        new_messages.push(messages[0].clone()); // original system prompt
        new_messages.push(compacted_message);
        new_messages.extend(messages[summarize_end..].iter().cloned());

        // Calculate tokens after compaction
        let tokens_after = token_counter.count_messages(&new_messages);

        // Replace messages in state
        *state.messages_mut() = new_messages;

        tracing::info!(
            "Compacted conversation: {} messages -> {}, {} tokens -> {}",
            messages_before,
            state.messages().len(),
            tokens_before,
            tokens_after,
        );

        Ok(CompactionReport {
            messages_before,
            messages_after: state.messages().len(),
            tokens_before,
            tokens_after,
        })
    }

    /// Call the summarization cascade via llm-cascade.
    #[allow(clippy::await_holding_lock)]
    async fn call_summarization(&self, prompt: Message) -> Result<String> {
        let conversation = Conversation::new(vec![prompt]);
        let cascade_name = self.cascade_name.clone();
        let config = Arc::clone(&self.cascade_config);

        let conn_guard = self.db_conn.lock().map_err(|e| {
            AgentError::InferenceFailed(format!("Failed to acquire db lock for compaction: {}", e))
        })?;

        // Note: std::sync::Mutex is held across the await because run_cascade is async
        // but takes &Connection (sync). The lock duration is the HTTP call time.
        // This is acceptable because compaction is infrequent and we don't want to
        // move the Connection between threads (SQLite is not Send).
        let result =
            llm_cascade::run_cascade(&cascade_name, &conversation, &config, &conn_guard).await;

        drop(conn_guard);

        match result {
            Ok(response) => Ok(response.text_only()),
            Err(e) => Err(AgentError::InferenceFailed(format!(
                "Summarization cascade '{}' failed: {}",
                cascade_name, e
            ))),
        }
    }
}

/// Format messages into a readable text representation for summarization.
fn format_messages_for_summarization(messages: &[Message]) -> String {
    let mut output = String::new();
    for msg in messages {
        let role_str = match msg.role {
            MessageRole::System => "System",
            MessageRole::User => "User",
            MessageRole::Assistant => "Assistant",
            MessageRole::Tool => "Tool",
        };
        output.push_str(&format!("--- {} ---\n{}\n\n", role_str, msg.content));
    }
    output
}
