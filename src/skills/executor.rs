//! Executor for skill executables (stdin/stdout JSON protocol).

use std::path::Path;
use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tracing::{debug, error, warn};

use crate::tools::ToolResult;

/// Default timeout for skill executable invocations.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Executes skill executables via the stdin/stdout JSON protocol.
///
/// Protocol:
/// 1. The executable is spawned with its cwd set to the skill directory.
/// 2. The tool arguments (`serde_json::Value`) are serialised as JSON and
///    written to the process's stdin, followed by EOF.
/// 3. The process's stdout is read in full and parsed as a JSON
///    [`ToolResult`].
/// 4. Stderr is captured and logged via `tracing`.
pub struct SkillExecutor;

impl SkillExecutor {
    /// Run a skill executable with the given arguments.
    ///
    /// Returns a [`ToolResult`] derived from the process's stdout.
    /// Errors from spawn, timeout, or invalid JSON on stdout produce
    /// `ToolResult::Error`.
    pub async fn execute(
        executable: &Path,
        args: serde_json::Value,
        skill_dir: &Path,
        timeout: Duration,
    ) -> ToolResult {
        debug!(
            exe = %executable.display(),
            skill_dir = %skill_dir.display(),
            "Spawning skill executable"
        );

        let mut child = match Command::new(executable)
            .current_dir(skill_dir)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                error!(exe = %executable.display(), "Failed to spawn: {e}");
                return ToolResult::err(format!(
                    "Failed to spawn skill executable '{}': {}",
                    executable.display(),
                    e
                ));
            }
        };

        // Write JSON args to stdin.
        if let Some(mut stdin) = child.stdin.take() {
            let json_bytes = match serde_json::to_vec(&args) {
                Ok(b) => b,
                Err(e) => {
                    return ToolResult::err(format!("Failed to serialise skill arguments: {}", e));
                }
            };

            if let Err(e) = stdin.write_all(&json_bytes).await {
                return ToolResult::err(format!("Failed to write to skill stdin: {}", e));
            }
            if let Err(e) = stdin.shutdown().await {
                warn!("Failed to close skill stdin: {e}");
            }
        }

        // Wait for the process with a timeout.
        let result = match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                error!(exe = %executable.display(), "Process error: {e}");
                return ToolResult::err(format!("Skill process error: {}", e));
            }
            Err(_elapsed) => {
                // Timeout – the child was moved into wait_with_output and dropped,
                // which implicitly kills the process on tokio::process::Child drop.
                return ToolResult::err(format!(
                    "Skill '{}' timed out after {:?}",
                    executable.display(),
                    timeout
                ));
            }
        };

        // Log stderr.
        if !result.stderr.is_empty() {
            let stderr_str = String::from_utf8_lossy(&result.stderr);
            if !result.status.success() {
                warn!(
                    skill = %executable.display(),
                    "stderr (exit {:?}): {}",
                    result.status.code(),
                    stderr_str
                );
            } else {
                debug!(
                    skill = %executable.display(),
                    "stderr: {}",
                    stderr_str
                );
            }
        }

        // Parse stdout as ToolResult JSON.
        let stdout_str = String::from_utf8_lossy(&result.stdout);

        if stdout_str.trim().is_empty() {
            return if result.status.success() {
                ToolResult::ok(serde_json::Value::Null)
            } else {
                ToolResult::err(format!(
                    "Skill exited with {:?} and produced no output",
                    result.status.code()
                ))
            };
        }

        match serde_json::from_str::<ToolResult>(&stdout_str) {
            Ok(tool_result) => tool_result,
            Err(e) => {
                warn!(
                    skill = %executable.display(),
                    "Stdout is not valid ToolResult JSON: {e}. Raw stdout: {stdout_str}"
                );
                // Return raw output as data so the LLM can still see it.
                ToolResult::ok(serde_json::json!({
                    "raw_output": stdout_str.trim().to_string(),
                    "parse_warning": "Executable output was not valid ToolResult JSON"
                }))
            }
        }
    }
}
