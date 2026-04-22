//! SKILL.md frontmatter parser.

use std::path::Path;

use gray_matter::{engine::YAML, Matter};
use tracing::{debug, warn};

use crate::error::{AgentError, Result};

use super::types::{Skill, SkillMetadata};

const SKILL_FILE: &str = "SKILL.md";

/// Well-known executable names searched in priority order.
const EXECUTABLE_NAMES: &[&str] = &["run.sh", "run.py", "run", "bin"];

/// Parse a `SKILL.md` file into a fully resolved [`Skill`].
///
/// The frontmatter (YAML between `---` delimiters) is deserialised into
/// [`SkillMetadata`].  Everything after the closing delimiter becomes the
/// `instructions` field.  The skill directory is then scanned for an executable.
pub fn parse_skill_file(path: &Path) -> Result<Skill> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| AgentError::SkillError(format!("Cannot read {}: {}", path.display(), e)))?;

    let matter: Matter<YAML> = Matter::new();
    let parsed = matter.parse::<SkillMetadata>(&content).map_err(|e| {
        AgentError::SkillError(format!(
            "Frontmatter parse error in {}: {}",
            path.display(),
            e
        ))
    })?;

    let metadata = parsed.data.ok_or_else(|| {
        AgentError::SkillError(format!(
            "No valid YAML frontmatter found in {}",
            path.display()
        ))
    })?;

    if metadata.name.is_empty() {
        return Err(AgentError::SkillError(format!(
            "Skill metadata in {} is missing a 'name' field",
            path.display()
        )));
    }

    let instructions = parsed.content;
    let directory = path.parent().unwrap_or(path).to_path_buf();

    // Scan the skill directory for an executable.
    let executable_path = find_executable(&directory);

    if let Some(ref exe) = executable_path {
        debug!(
            skill = %metadata.name,
            "Found executable for skill: {}",
            exe.display()
        );
    }

    Ok(Skill {
        metadata,
        instructions,
        executable_path,
        directory,
    })
}

/// Search the skill directory for an executable in priority order:
/// 1. Well-known names (`run.sh`, `run.py`, `run`, `bin`)
/// 2. Any regular file with the execute permission bit set
fn find_executable(dir: &Path) -> Option<std::path::PathBuf> {
    if !dir.is_dir() {
        return None;
    }

    let read_dir = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) => {
            warn!("Cannot read skill directory {}: {}", dir.display(), e);
            return None;
        }
    };

    // First pass: look for well-known names.
    for name in EXECUTABLE_NAMES {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    // Second pass: look for any file with execute permission.
    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.is_file() {
            use std::os::unix::fs::PermissionsExt;
            let mode = entry.metadata().ok()?.permissions().mode();
            if mode & 0o111 != 0 {
                debug!("Found executable by permission bit: {}", path.display());
                return Some(path);
            }
        }
    }

    None
}

/// Re-serialize the metadata as YAML frontmatter for writing SKILL.md.
pub fn serialize_skill_md(metadata: &SkillMetadata, body: &str) -> String {
    let yaml = serde_yaml::to_string(metadata).unwrap_or_default();
    format!("---\n{yaml}---\n{body}")
}

/// Read a SKILL.md file and split it into (frontmatter_yaml_string, body).
pub fn read_skill_parts(path: &Path) -> Result<(String, String)> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| AgentError::SkillError(format!("Cannot read {}: {}", path.display(), e)))?;

    let matter: Matter<YAML> = Matter::new();
    let parsed = matter.parse::<serde_yaml::Value>(&content).map_err(|e| {
        AgentError::SkillError(format!(
            "Frontmatter parse error in {}: {}",
            path.display(),
            e
        ))
    })?;

    // Reconstruct the original frontmatter text from parsed.matter
    let frontmatter = if parsed.matter.is_empty() {
        String::new()
    } else {
        format!("---\n{}\n---\n", parsed.matter)
    };

    Ok((frontmatter, parsed.content))
}

/// The expected SKILL.md filename (exposed for use by SkillManager).
pub fn skill_file_name() -> &'static str {
    SKILL_FILE
}
