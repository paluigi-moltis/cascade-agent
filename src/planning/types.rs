use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub id: String,
    pub task_id: String,
    pub title: String,
    pub steps: Vec<PlanStep>,
    pub status: PlanStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub file_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub number: usize,
    pub description: String,
    pub status: StepStatus,
    pub result: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PlanStatus {
    Draft,
    PendingApproval,
    Approved,
    InProgress,
    Completed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StepStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl Plan {
    /// Create a new plan with the given steps
    pub fn new(task_id: &str, title: &str, step_descriptions: Vec<String>) -> Self {
        let steps: Vec<PlanStep> = step_descriptions
            .into_iter()
            .enumerate()
            .map(|(i, desc)| PlanStep {
                number: i + 1,
                description: desc,
                status: StepStatus::Pending,
                result: None,
            })
            .collect();

        Plan {
            id: uuid::Uuid::new_v4().to_string(),
            task_id: task_id.to_string(),
            title: title.to_string(),
            steps,
            status: PlanStatus::Draft,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            file_path: PathBuf::new(), // Will be set by PlanManager
        }
    }

    pub fn mark_step_in_progress(&mut self, step_num: usize) -> bool {
        if let Some(step) = self.steps.iter_mut().find(|s| s.number == step_num) {
            step.status = StepStatus::InProgress;
            self.updated_at = Utc::now();
            true
        } else {
            false
        }
    }

    pub fn mark_step_completed(&mut self, step_num: usize, result: Option<String>) -> bool {
        if let Some(step) = self.steps.iter_mut().find(|s| s.number == step_num) {
            step.status = StepStatus::Completed;
            step.result = result;
            self.updated_at = Utc::now();
            // If all steps completed, mark plan as completed
            if self
                .steps
                .iter()
                .all(|s| s.status == StepStatus::Completed || s.status == StepStatus::Skipped)
            {
                self.status = PlanStatus::Completed;
            }
            true
        } else {
            false
        }
    }

    pub fn mark_step_failed(&mut self, step_num: usize, error: String) -> bool {
        if let Some(step) = self.steps.iter_mut().find(|s| s.number == step_num) {
            step.status = StepStatus::Failed;
            step.result = Some(error);
            self.updated_at = Utc::now();
            true
        } else {
            false
        }
    }
}
