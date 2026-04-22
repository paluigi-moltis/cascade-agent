//! Skill system: discovery, parsing, CRUD, and Tool adapter.

pub mod executor;
pub mod parser;
pub mod skill_tool;
pub mod types;

use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

use crate::error::{AgentError, Result};
use crate::tools::Tool;

use parser::{parse_skill_file, read_skill_parts, serialize_skill_md, skill_file_name};
use skill_tool::SkillTool;
use types::SkillMetadata;

/// Manages a collection of skills discovered from a directory on disk.
///
/// Each skill lives in its own subdirectory of `skills_dir` and must contain
/// a `SKILL.md` file with YAML frontmatter.
pub struct SkillManager {
    /// Root directory under which skill subdirectories live.
    skills_dir: std::path::PathBuf,
    /// Parsed skills indexed by their metadata name.
    skills: HashMap<String, types::Skill>,
}

impl SkillManager {
    /// Create a new `SkillManager`, ensuring the skills directory exists.
    pub fn new(skills_dir: std::path::PathBuf) -> Result<Self> {
        if !skills_dir.exists() {
            std::fs::create_dir_all(&skills_dir).map_err(|e| {
                AgentError::SkillError(format!(
                    "Cannot create skills directory '{}': {}",
                    skills_dir.display(),
                    e
                ))
            })?;
            info!(dir = %skills_dir.display(), "Created skills directory");
        }

        Ok(Self {
            skills_dir,
            skills: HashMap::new(),
        })
    }

    /// Scan the skills directory for subdirectories containing `SKILL.md`,
    /// parse each, and insert into the internal map.
    ///
    /// Returns the names of all successfully discovered skills.
    pub fn discover(&mut self) -> Result<Vec<String>> {
        let mut discovered = Vec::new();

        let entries = match std::fs::read_dir(&self.skills_dir) {
            Ok(rd) => rd,
            Err(e) => {
                return Err(AgentError::SkillError(format!(
                    "Cannot read skills directory '{}': {}",
                    self.skills_dir.display(),
                    e
                )));
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let skill_md = path.join(skill_file_name());
            if !skill_md.exists() {
                continue;
            }

            match parse_skill_file(&skill_md) {
                Ok(skill) => {
                    let name = skill.metadata.name.clone();
                    debug!(skill = %name, dir = %path.display(), "Discovered skill");
                    self.skills.insert(name.clone(), skill);
                    discovered.push(name);
                }
                Err(e) => {
                    warn!(
                        path = %skill_md.display(),
                        error = %e,
                        "Skipping invalid skill directory"
                    );
                }
            }
        }

        info!(count = discovered.len(), "Skill discovery complete");
        Ok(discovered)
    }

    /// Get a skill by name.
    pub fn get(&self, name: &str) -> Option<&types::Skill> {
        self.skills.get(name)
    }

    /// Create a new skill directory and `SKILL.md` file.
    pub fn create_skill(&self, name: &str, metadata: &SkillMetadata, body: &str) -> Result<()> {
        let skill_dir = self.skills_dir.join(name);

        if skill_dir.exists() {
            return Err(AgentError::SkillError(format!(
                "Skill directory '{}' already exists",
                skill_dir.display()
            )));
        }

        std::fs::create_dir_all(&skill_dir).map_err(|e| {
            AgentError::SkillError(format!(
                "Cannot create skill directory '{}': {}",
                skill_dir.display(),
                e
            ))
        })?;

        let content = serialize_skill_md(metadata, body);
        let skill_md = skill_dir.join(skill_file_name());
        std::fs::write(&skill_md, &content).map_err(|e| {
            AgentError::SkillError(format!("Cannot write '{}': {}", skill_md.display(), e))
        })?;

        info!(skill = name, dir = %skill_dir.display(), "Created skill");
        Ok(())
    }

    /// Update the body (instructions) of an existing skill, preserving its
    /// frontmatter.
    pub fn update_skill(&self, name: &str, body: &str) -> Result<()> {
        let skill_dir = self.skills_dir.join(name);
        let skill_md = skill_dir.join(skill_file_name());

        if !skill_md.exists() {
            return Err(AgentError::SkillError(format!(
                "Skill '{}' not found (no SKILL.md at {})",
                name,
                skill_md.display()
            )));
        }

        let (frontmatter, _old_body) = read_skill_parts(&skill_md)?;
        let content = format!("{}{}", frontmatter, body);
        std::fs::write(&skill_md, &content).map_err(|e| {
            AgentError::SkillError(format!("Cannot write '{}': {}", skill_md.display(), e))
        })?;

        debug!(skill = name, "Updated skill instructions");
        Ok(())
    }

    /// Remove a skill directory entirely.
    pub fn remove_skill(&self, name: &str) -> Result<()> {
        let skill_dir = self.skills_dir.join(name);

        if !skill_dir.exists() {
            return Err(AgentError::SkillError(format!(
                "Skill '{}' not found at '{}'",
                name,
                skill_dir.display()
            )));
        }

        std::fs::remove_dir_all(&skill_dir).map_err(|e| {
            AgentError::SkillError(format!(
                "Cannot remove skill directory '{}': {}",
                skill_dir.display(),
                e
            ))
        })?;

        info!(skill = name, "Removed skill");
        Ok(())
    }

    /// Convert all loaded skills into `SkillTool` instances implementing
    /// the [`Tool`] trait.
    pub fn all_tools(&self) -> Vec<Box<dyn Tool>> {
        self.skills
            .values()
            .map(|skill| Box::new(SkillTool::new(skill.clone())) as Box<dyn Tool>)
            .collect()
    }

    /// Return the path to the skills directory.
    pub fn skills_dir(&self) -> &Path {
        &self.skills_dir
    }
}
