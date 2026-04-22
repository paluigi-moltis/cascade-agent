pub mod types;

use crate::error::Result;
use std::path::{Path, PathBuf};
use types::{Plan, PlanStatus, StepStatus};

pub struct PlanManager {
    plans_dir: PathBuf,
}

impl PlanManager {
    pub fn new(plans_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&plans_dir)?;
        Ok(Self { plans_dir })
    }

    /// Create a new plan, save it to disk, return it
    pub fn create_plan(&self, task_id: &str, title: &str, steps: Vec<String>) -> Result<Plan> {
        let mut plan = Plan::new(task_id, title, steps);
        let filename = format!(
            "{}_{}.md",
            plan.created_at.format("%Y%m%d_%H%M%S"),
            &plan.id[..8]
        );
        plan.file_path = self.plans_dir.join(filename);
        self.save_plan(&plan)?;
        Ok(plan)
    }

    /// Save plan to markdown file
    pub fn save_plan(&self, plan: &Plan) -> Result<()> {
        let content = self.render_markdown(plan);
        std::fs::write(&plan.file_path, content)?;
        Ok(())
    }

    /// Load a plan from its file path
    pub fn load_plan(&self, file_path: &Path) -> Result<Plan> {
        let content = std::fs::read_to_string(file_path)?;
        self.parse_markdown(&content, file_path)
    }

    /// List all plan files in the plans directory
    pub fn list_plans(&self) -> Result<Vec<PathBuf>> {
        let mut plans: Vec<PathBuf> = std::fs::read_dir(&self.plans_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "md"))
            .collect();
        plans.sort();
        Ok(plans)
    }

    /// Update a step's status and re-save
    pub fn update_step(
        &self,
        plan: &mut Plan,
        step_num: usize,
        status: StepStatus,
        result: Option<&str>,
    ) -> Result<()> {
        if let Some(step) = plan.steps.iter_mut().find(|s| s.number == step_num) {
            step.status = status;
            step.result = result.map(String::from);
            plan.updated_at = chrono::Utc::now();
            self.save_plan(plan)?;
            Ok(())
        } else {
            Err(crate::error::AgentError::ConfigError(format!(
                "Step {} not found in plan",
                step_num
            )))
        }
    }

    /// Render a Plan to markdown
    pub fn render_markdown(&self, plan: &Plan) -> String {
        let status_emoji = match &plan.status {
            PlanStatus::Draft => "📝",
            PlanStatus::PendingApproval => "⏳",
            PlanStatus::Approved => "✅",
            PlanStatus::InProgress => "🔄",
            PlanStatus::Completed => "🏁",
            PlanStatus::Cancelled => "❌",
        };

        let step_emoji = |s: &StepStatus| match s {
            StepStatus::Pending => "⬜",
            StepStatus::InProgress => "🔄",
            StepStatus::Completed => "✅",
            StepStatus::Failed => "❌",
            StepStatus::Skipped => "⏭️",
        };

        let mut md = format!("# {}\n\n", plan.title);
        md.push_str(&format!("**Status:** {} {:?}\n", status_emoji, plan.status));
        md.push_str(&format!("**Task ID:** {}\n", plan.task_id));
        md.push_str(&format!("**Plan ID:** {}\n", plan.id));
        md.push_str(&format!(
            "**Created:** {}\n\n",
            plan.created_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        md.push_str("---\n\n## Steps\n\n");

        for step in &plan.steps {
            md.push_str(&format!(
                "{} **Step {}:** {}\n",
                step_emoji(&step.status),
                step.number,
                step.description
            ));
            if let Some(result) = &step.result {
                md.push_str(&format!("   - Result: {}\n", result));
            }
            md.push('\n');
        }

        md
    }

    /// Parse a plan from markdown (basic parser — reconstructs Plan from rendered markdown)
    fn parse_markdown(&self, content: &str, file_path: &Path) -> Result<Plan> {
        // Simple parser: extract title from first # heading, parse steps
        let mut title = String::new();
        let mut steps = Vec::new();
        let mut step_num = 0;

        for line in content.lines() {
            if title.is_empty() && line.starts_with("# ") {
                title = line[2..].trim().to_string();
            } else if line.contains("Step ") && line.contains(':') {
                step_num += 1;
                if let Some(desc_start) = line.find(": ") {
                    let desc = line[desc_start + 2..].trim().to_string();
                    // Remove emoji prefix if present
                    let desc = desc.trim_start_matches(|c: char| !c.is_alphanumeric() && c != '_');
                    let desc = desc.trim_start_matches(|c: char| !c.is_alphanumeric());
                    steps.push(types::PlanStep {
                        number: step_num,
                        description: desc.to_string(),
                        status: StepStatus::Pending, // Lost in markdown serialization
                        result: None,
                    });
                }
            }
        }

        Ok(Plan {
            id: uuid::Uuid::new_v4().to_string(), // Regenerate since not stored in markdown
            task_id: String::new(),
            title,
            steps,
            status: PlanStatus::Draft,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            file_path: file_path.to_path_buf(),
        })
    }
}
