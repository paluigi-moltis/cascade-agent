use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum OrchestratorMessage {
    // Agent → Orchestrator
    TaskStarted {
        task_id: String,
        description: String,
    },
    StepUpdate {
        task_id: String,
        step: usize,
        status: String,
    },
    AssistantText(String),
    ToolExecuted {
        tool: String,
        status: String,
        duration_ms: u64,
    },
    ContextCompacted {
        before: usize,
        after: usize,
    },
    Error(String),
    Warning(String),
    TaskCompleted {
        task_id: String,
        output_path: Option<String>,
    },
    TaskCancelled,
    UserQuestion {
        question: String,
    },

    // Orchestrator → Agent
    UserReply {
        content: String,
    },
    PlanApproval {
        approved: bool,
        feedback: Option<String>,
    },
    CancelTask,
}
