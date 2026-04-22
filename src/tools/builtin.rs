//! Built-in tools: echo, list_tools, read_file, write_file, ask_user.

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{Tool, ToolResult};

// ---------------------------------------------------------------------------
// EchoTool
// ---------------------------------------------------------------------------

/// A simple echo tool useful for testing.
///
/// Parameters:
///   - `message` (string, required): the text to echo back.
#[derive(Debug)]
pub struct EchoTool;

impl EchoTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EchoTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echoes back the provided message. Useful for testing and debugging."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo back."
                }
            },
            "required": ["message"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let message = match args.get("message").and_then(|v| v.as_str()) {
            Some(m) => m.to_owned(),
            None => return ToolResult::err("Missing required parameter 'message'"),
        };
        ToolResult::ok_string(message)
    }
}

// ---------------------------------------------------------------------------
// ListToolsTool
// ---------------------------------------------------------------------------

/// Lists all registered tools and their descriptions.
///
/// Takes no parameters. Returns a JSON array of objects with `name` and `description`.
#[derive(Debug)]
pub struct ListToolsTool {
    tools: Vec<(String, String)>,
}

impl ListToolsTool {
    /// Create from a list of tool names.
    pub fn new(names: Vec<String>, tools: Vec<std::sync::Arc<dyn Tool>>) -> Self {
        let mut entries: Vec<(String, String)> =
            names.iter().map(|n| (n.clone(), String::new())).collect();

        // Fill in descriptions from the actual tool objects.
        for tool in &tools {
            if let Some(entry) = entries.iter_mut().find(|(name, _)| name == tool.name()) {
                entry.1 = tool.description().to_owned();
            }
        }

        Self { tools: entries }
    }
}

#[async_trait]
impl Tool for ListToolsTool {
    fn name(&self) -> &str {
        "list_tools"
    }

    fn description(&self) -> &str {
        "Lists all available tools and their descriptions."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, _args: Value) -> ToolResult {
        let list: Vec<Value> = self
            .tools
            .iter()
            .map(|(name, desc)| {
                json!({
                    "name": name,
                    "description": desc
                })
            })
            .collect();
        ToolResult::ok(json!(list))
    }
}

// ---------------------------------------------------------------------------
// ReadFileTool
// ---------------------------------------------------------------------------

/// Reads a file from disk and returns its contents.
///
/// Parameters:
///   - `path` (string, required): filesystem path to the file.
#[derive(Debug)]
pub struct ReadFileTool;

impl ReadFileTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Reads the contents of a file at the given path and returns the text."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The filesystem path of the file to read."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::err("Missing required parameter 'path'"),
        };

        match tokio::fs::read_to_string(path).await {
            Ok(content) => ToolResult::ok_string(content),
            Err(e) => ToolResult::err(format!("Failed to read file '{}': {}", path, e)),
        }
    }
}

// ---------------------------------------------------------------------------
// WriteFileTool
// ---------------------------------------------------------------------------

/// Writes content to a file on disk.
///
/// Parameters:
///   - `path`    (string, required): filesystem path to the file.
///   - `content` (string, required): the text to write.
#[derive(Debug)]
pub struct WriteFileTool;

impl WriteFileTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WriteFileTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Writes the provided content to a file at the given path. Creates parent directories if needed. Overwrites existing files."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The filesystem path of the file to write."
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file."
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_owned(),
            None => return ToolResult::err("Missing required parameter 'path'"),
        };
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_owned(),
            None => return ToolResult::err("Missing required parameter 'content'"),
        };

        // Ensure parent directories exist.
        if let Some(parent) = std::path::Path::new(&path).parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = tokio::fs::create_dir_all(parent).await {
                    return ToolResult::err(format!(
                        "Failed to create directories for '{}': {}",
                        path, e
                    ));
                }
            }
        }

        match tokio::fs::write(&path, &content).await {
            Ok(()) => ToolResult::ok_string(format!("Successfully wrote to '{}'", path)),
            Err(e) => ToolResult::err(format!("Failed to write file '{}': {}", path, e)),
        }
    }
}

// ---------------------------------------------------------------------------
// AskUserTool
// ---------------------------------------------------------------------------

/// Signals that the agent needs to ask the user a question.
///
/// In practice the orchestrator will intercept this tool call and route the
/// question to the user interface, then inject the user's reply as a tool
/// result.  When executed directly it returns a placeholder response.
///
/// Parameters:
///   - `question` (string, required): the question to ask the user.
#[derive(Debug)]
pub struct AskUserTool;

impl AskUserTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AskUserTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for AskUserTool {
    fn name(&self) -> &str {
        "ask_user"
    }

    fn description(&self) -> &str {
        "Ask the user a question. The orchestrator will route this to the user interface and the user's reply will be provided as the tool result."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user."
                }
            },
            "required": ["question"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let question = match args.get("question").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return ToolResult::err("Missing required parameter 'question'"),
        };

        // Placeholder: the orchestrator should intercept this tool call and
        // replace the result with the actual user reply.
        ToolResult::ok(json!({
            "status": "pending_user_response",
            "question": question,
            "answer": null
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn echo_works() {
        let tool = EchoTool::new();
        let result = tool.execute(json!({"message": "hello"})).await;
        assert_eq!(result.status, super::super::ToolStatus::Success);
        assert_eq!(result.data.as_str().unwrap(), "hello");
    }

    #[tokio::test]
    async fn echo_missing_param() {
        let tool = EchoTool::new();
        let result = tool.execute(json!({})).await;
        assert_eq!(result.status, super::super::ToolStatus::Error);
    }

    #[tokio::test]
    async fn read_file_works() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello world").unwrap();

        let tool = ReadFileTool::new();
        let result = tool.execute(json!({"path": path.to_str().unwrap()})).await;
        assert_eq!(result.status, super::super::ToolStatus::Success);
        assert_eq!(result.data.as_str().unwrap(), "hello world");
    }

    #[tokio::test]
    async fn write_file_works() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("subdir").join("out.txt");

        let tool = WriteFileTool::new();
        let result = tool
            .execute(json!({"path": path.to_str().unwrap(), "content": "data"}))
            .await;
        assert_eq!(result.status, super::super::ToolStatus::Success);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "data");
    }
}
