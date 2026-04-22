//! Core types for the skill system.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// JSON Schema describing the expected input format for a skill's executable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputFormat {
    pub content_type: String,
    pub schema: serde_json::Value,
}

/// Metadata extracted from the YAML frontmatter of a SKILL.md file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMetadata {
    /// Machine-readable skill name (used as tool name).
    pub name: String,
    /// Human-readable description for LLM decision-making.
    pub description: String,
    /// Optional version string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Tags for categorisation / discovery.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Optional structured input format for the executable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_format: Option<InputFormat>,
}

/// A fully resolved skill, ready for execution or registration as a tool.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Parsed frontmatter metadata.
    pub metadata: SkillMetadata,
    /// The markdown body after the frontmatter (instructions for the LLM).
    pub instructions: String,
    /// Path to an executable script/binary inside the skill directory, if found.
    pub executable_path: Option<PathBuf>,
    /// The directory containing the skill's SKILL.md and any executables.
    pub directory: PathBuf,
}
