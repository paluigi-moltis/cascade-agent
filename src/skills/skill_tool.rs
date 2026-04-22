//! Adapter that wraps a [`Skill`] as a [`Tool`] for the tool registry.

use async_trait::async_trait;

use crate::tools::{Tool, ToolResult};

use super::executor::{SkillExecutor, DEFAULT_TIMEOUT};
use super::types::Skill;

/// A tool wrapper around a parsed [`Skill`].
///
/// If the skill has an executable, invoking the tool runs it via
/// [`SkillExecutor`].  Otherwise, the tool returns the skill's
/// instructions as a reference response.
pub struct SkillTool {
    skill: Skill,
}

impl SkillTool {
    pub fn new(skill: Skill) -> Self {
        Self { skill }
    }

    /// Build a description string for the LLM, combining the metadata
    /// description with the instructions (truncated if very long).
    fn build_description(&self) -> String {
        let meta_desc = self.skill.metadata.description.clone();
        let instructions = self.skill.instructions.trim();
        if instructions.is_empty() {
            return meta_desc;
        }
        // Truncate very long instructions to avoid wasting context window.
        const MAX_INSTRUCTIONS_LEN: usize = 2_000;
        let truncated = if instructions.len() > MAX_INSTRUCTIONS_LEN {
            format!(
                "{}\n\n[... instructions truncated at {} chars ...]",
                &instructions[..MAX_INSTRUCTIONS_LEN],
                MAX_INSTRUCTIONS_LEN
            )
        } else {
            instructions.to_string()
        };
        format!("{}\n\nInstructions:\n{}", meta_desc, truncated)
    }

    /// Build a parameters schema for the tool.
    fn build_parameters_schema(&self) -> serde_json::Value {
        if let Some(ref input_format) = self.skill.metadata.input_format {
            // Use the schema from the skill's input_format.
            input_format.schema.clone()
        } else if self.skill.executable_path.is_some() {
            // Has an executable but no declared schema – use a generic schema.
            serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input to pass to the skill executable"
                    }
                }
            })
        } else {
            // No executable – no real parameters needed.
            serde_json::json!({
                "type": "object",
                "properties": {}
            })
        }
    }
}

#[async_trait]
impl Tool for SkillTool {
    fn name(&self) -> &str {
        &self.skill.metadata.name
    }

    fn description(&self) -> &str {
        // We can't return a reference to a computed string, so we store it.
        // Actually, Tool trait requires &str which makes dynamic descriptions tricky.
        // We'll use a workaround via self-description caching.
        // The trait requires &str, so we need to store the description.
        // For now, return the metadata description (the trait only requires &str).
        &self.skill.metadata.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.build_parameters_schema()
    }

    async fn execute(&self, args: serde_json::Value) -> ToolResult {
        if let Some(ref exe_path) = self.skill.executable_path {
            SkillExecutor::execute(exe_path, args, &self.skill.directory, DEFAULT_TIMEOUT).await
        } else {
            // No executable – return the instructions as a reference response.
            ToolResult::ok(serde_json::json!({
                "instructions": self.skill.instructions.trim()
            }))
        }
    }

    fn to_definition(&self) -> llm_cascade::ToolDefinition {
        llm_cascade::ToolDefinition {
            name: self.name().to_owned(),
            description: self.build_description(),
            parameters: self.parameters_schema(),
        }
    }
}
