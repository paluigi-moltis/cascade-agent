//! Conversation state management for the agent loop.

use llm_cascade::{Conversation, Message};
use serde::Serialize;

use crate::error::Result;
use crate::memory::state::ConversationStateAccess;

/// Wraps llm-cascade Conversation with agent-specific metadata.
///
/// Tracks turn count, task ID, and timestamps for persistence and observability.
/// Implements [`ConversationStateAccess`] so the memory subsystem can operate on it.
#[derive(Debug, Serialize)]
pub struct ConversationState {
    /// The ordered list of conversation messages (user, assistant, tool results).
    pub messages: Vec<Message>,
    /// System prompt prepended to every LLM call.
    pub system_prompt: String,
    /// Number of user turns processed so far.
    pub turn_count: usize,
    /// Unique identifier for the current task.
    pub task_id: String,
    /// Timestamp when this state was created.
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl ConversationState {
    /// Create a new empty conversation state.
    pub fn new(system_prompt: String, task_id: String) -> Self {
        Self {
            messages: Vec::new(),
            system_prompt,
            turn_count: 0,
            task_id,
            created_at: chrono::Utc::now(),
        }
    }

    /// Append a user message and increment the turn counter.
    pub fn add_user_message(&mut self, content: String) {
        self.messages.push(Message::user(content));
        self.turn_count += 1;
    }

    /// Append an assistant text response.
    pub fn add_assistant_text(&mut self, content: String) {
        self.messages.push(Message::assistant(content));
    }

    /// Append a tool result message associated with a tool call ID.
    pub fn add_tool_result(&mut self, tool_call_id: &str, result: &str) {
        self.messages
            .push(Message::tool(result.to_string(), tool_call_id.to_string()));
    }

    /// Build an `llm_cascade::Conversation` from current state.
    ///
    /// Prepends the system prompt as the first message and attaches no tools
    /// (tools are added via `with_tools` by the caller when needed).
    pub fn to_conversation(&self) -> Conversation {
        let mut all_msgs = vec![Message::system(&self.system_prompt)];
        all_msgs.extend(self.messages.clone());
        Conversation::new(all_msgs)
    }

    /// Get the last assistant text message (excluding tool-call stubs).
    pub fn last_assistant_text(&self) -> Option<String> {
        self.messages
            .iter()
            .rev()
            .find(|m| matches!(m.role, llm_cascade::MessageRole::Assistant))
            .map(|m| m.content.clone())
    }

    /// Serialize state to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Into::into)
    }

    /// Save state to a JSON file in the given directory.
    pub fn to_json_file(&self, dir: &str) -> Result<std::path::PathBuf> {
        std::fs::create_dir_all(dir)?;
        let filename = format!(
            "state_{}_{}.json",
            self.created_at.format("%Y%m%d_%H%M%S"),
            &self.task_id[..8.min(self.task_id.len())]
        );
        let path = std::path::PathBuf::from(dir).join(filename);
        std::fs::write(&path, self.to_json()?)?;
        Ok(path)
    }
}

impl ConversationStateAccess for ConversationState {
    fn messages(&self) -> &[Message] {
        &self.messages
    }

    fn messages_mut(&mut self) -> &mut Vec<Message> {
        &mut self.messages
    }

    fn system_prompt(&self) -> &str {
        &self.system_prompt
    }
}
