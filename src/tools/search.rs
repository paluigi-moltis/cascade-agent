//! Web search tools: Tavily and Brave.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{Tool, ToolResult};

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

// ---------------------------------------------------------------------------
// TavilySearchTool
// ---------------------------------------------------------------------------

/// Calls the [Tavily Search API](https://tavily.com).
///
/// Parameters:
///   - `query`       (string, required): the search query.
///   - `max_results` (int, optional):    max number of results (default 5).
#[derive(Debug)]
pub struct TavilySearchTool {
    api_key: String,
    client: reqwest::Client,
    default_max_results: usize,
}

impl TavilySearchTool {
    /// Create from a raw API key.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
            default_max_results: 5,
        }
    }

    /// Create by reading the API key from an environment variable.
    pub fn from_env(env_var: &str, default_max_results: usize) -> Option<Self> {
        let api_key = std::env::var(env_var).ok()?;
        Some(Self {
            api_key,
            client: reqwest::Client::new(),
            default_max_results,
        })
    }

    async fn do_search(&self, query: &str, max_results: usize) -> ToolResult {
        let body = json!({
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": true,
        });

        let resp = match self
            .client
            .post("https://api.tavily.com/search")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return ToolResult::err(format!("Tavily request failed: {}", e)),
        };

        let status = resp.status();
        let text = match resp.text().await {
            Ok(t) => t,
            Err(e) => return ToolResult::err(format!("Failed to read Tavily response: {}", e)),
        };

        if !status.is_success() {
            return ToolResult::err(format!("Tavily API returned HTTP {}: {}", status, text));
        }

        // Parse the response
        let parsed: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => return ToolResult::err(format!("Failed to parse Tavily response: {}", e)),
        };

        // Extract results into a uniform format
        let results = parse_tavily_results(&parsed);
        let answer = parsed.get("answer").and_then(|v| v.as_str()).unwrap_or("");

        ToolResult::ok(json!({
            "query": query,
            "answer": answer,
            "results": results,
        }))
    }
}

fn parse_tavily_results(parsed: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(arr) = parsed.get("results").and_then(|v| v.as_array()) {
        for item in arr {
            out.push(json!({
                "title": item.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                "url": item.get("url").and_then(|v| v.as_str()).unwrap_or(""),
                "snippet": item.get("content").and_then(|v| v.as_str()).unwrap_or(""),
                "score": item.get("score").and_then(|v| v.as_f64()),
            }));
        }
    }
    out
}

#[async_trait]
impl Tool for TavilySearchTool {
    fn name(&self) -> &str {
        "tavily_search"
    }

    fn description(&self) -> &str {
        "Search the web using the Tavily Search API. Returns relevant results with titles, URLs, and snippets."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5).",
                    "default": self.default_max_results
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return ToolResult::err("Missing required parameter 'query'"),
        };
        let max_results = args
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.default_max_results as u64) as usize;

        self.do_search(query, max_results).await
    }
}

// ---------------------------------------------------------------------------
// BraveSearchTool
// ---------------------------------------------------------------------------

/// Calls the [Brave Web Search API](https://brave.com/search/api/).
///
/// Parameters:
///   - `query`       (string, required): the search query.
///   - `max_results` (int, optional):    max number of results (default 5).
#[derive(Debug)]
pub struct BraveSearchTool {
    api_key: String,
    client: reqwest::Client,
    default_max_results: usize,
}

impl BraveSearchTool {
    /// Create from a raw API key.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
            default_max_results: 5,
        }
    }

    /// Create by reading the API key from an environment variable.
    pub fn from_env(env_var: &str, default_max_results: usize) -> Option<Self> {
        let api_key = std::env::var(env_var).ok()?;
        Some(Self {
            api_key,
            client: reqwest::Client::new(),
            default_max_results,
        })
    }

    async fn do_search(&self, query: &str, max_results: usize) -> ToolResult {
        let url = format!(
            "https://api.search.brave.com/res/v1/web/search?q={}&count={}",
            urlencoding::encode(query),
            max_results
        );

        let resp = match self
            .client
            .get(&url)
            .header("Accept", "application/json")
            .header("Accept-Encoding", "gzip")
            .header("X-Subscription-Token", &self.api_key)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return ToolResult::err(format!("Brave request failed: {}", e)),
        };

        let status = resp.status();
        let text = match resp.text().await {
            Ok(t) => t,
            Err(e) => return ToolResult::err(format!("Failed to read Brave response: {}", e)),
        };

        if !status.is_success() {
            return ToolResult::err(format!("Brave API returned HTTP {}: {}", status, text));
        }

        let parsed: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => return ToolResult::err(format!("Failed to parse Brave response: {}", e)),
        };

        let results = parse_brave_results(&parsed);

        ToolResult::ok(json!({
            "query": query,
            "results": results,
        }))
    }
}

fn parse_brave_results(parsed: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    let web = parsed.get("web").and_then(|v| v.get("results"));
    if let Some(arr) = web.and_then(|v| v.as_array()) {
        for item in arr {
            out.push(json!({
                "title": item.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                "url": item.get("url").and_then(|v| v.as_str()).unwrap_or(""),
                "snippet": item.get("description").and_then(|v| v.as_str()).unwrap_or(""),
            }));
        }
    }
    out
}

#[async_trait]
impl Tool for BraveSearchTool {
    fn name(&self) -> &str {
        "brave_search"
    }

    fn description(&self) -> &str {
        "Search the web using the Brave Web Search API. Returns relevant results with titles, URLs, and snippets."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5).",
                    "default": self.default_max_results
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, args: Value) -> ToolResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return ToolResult::err("Missing required parameter 'query'"),
        };
        let max_results = args
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.default_max_results as u64) as usize;

        self.do_search(query, max_results).await
    }
}

// ---------------------------------------------------------------------------
// Minimal URL-encoding helper (avoid adding a dependency just for this)
// ---------------------------------------------------------------------------
mod urlencoding {
    pub fn encode(s: &str) -> String {
        let mut out = String::with_capacity(s.len() * 2);
        for byte in s.as_bytes() {
            match *byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    out.push(*byte as char);
                }
                _ => {
                    out.push_str(&format!("%{:02X}", byte));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tavily_schema_is_valid() {
        let tool = TavilySearchTool::new("test-key".into());
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .contains(&json!("query")));
    }

    #[test]
    fn brave_schema_is_valid() {
        let tool = BraveSearchTool::new("test-key".into());
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .contains(&json!("query")));
    }

    #[test]
    fn urlencode_works() {
        assert_eq!(urlencoding::encode("hello world"), "hello%20world");
        assert_eq!(urlencoding::encode("a+b=c"), "a%2Bb%3Dc");
    }
}
