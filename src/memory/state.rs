use llm_cascade::Message;

/// Trait for conversation state that memory management can operate on.
///
/// This decouples the memory module from any specific agent state representation.
/// The agent's `ConversationState` (e.g., in `src/agent/state.rs`) implements this trait.
pub trait ConversationStateAccess {
    /// Returns the current conversation messages as a slice.
    fn messages(&self) -> &[Message];

    /// Returns a mutable reference to the messages vector for in-place compaction.
    fn messages_mut(&mut self) -> &mut Vec<Message>;

    /// Returns the system prompt string (first system message content).
    fn system_prompt(&self) -> &str;
}
