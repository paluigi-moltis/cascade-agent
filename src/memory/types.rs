use serde::Serialize;

/// Report from a context compaction operation.
#[derive(Debug, Clone, Serialize)]
pub struct CompactionReport {
    /// Number of messages before compaction.
    pub messages_before: usize,
    /// Number of messages after compaction.
    pub messages_after: usize,
    /// Total tokens before compaction.
    pub tokens_before: usize,
    /// Total tokens after compaction.
    pub tokens_after: usize,
}
