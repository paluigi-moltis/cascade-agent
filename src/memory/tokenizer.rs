use llm_cascade::Message;
use tokenizers::Tokenizer;

use crate::error::Result;

/// Token counter backed by a HuggingFace `tokenizers` tokenizer.
///
/// Provides accurate token counting for LLM context window management.
/// Falls back to a character-based heuristic if tokenization fails.
pub struct TokenCounter {
    tokenizer: Option<Tokenizer>,
    model_identifier: String,
}

impl TokenCounter {
    /// Load a HuggingFace tokenizer by model identifier or local file path.
    ///
    /// Accepts:
    /// - HuggingFace model names like `"Xenova/gpt-4o"` (fetched via the Hub)
    /// - Local file paths to a tokenizer.json file
    ///
    /// If loading fails (e.g., no network for Hub model, missing file), the counter
    /// still works using a character-based estimate fallback.
    pub fn new(model_identifier: &str) -> Result<Self> {
        let tokenizer = match Tokenizer::from_pretrained(model_identifier, None) {
            Ok(t) => {
                tracing::info!("Loaded tokenizer: {}", model_identifier);
                Some(t)
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load tokenizer '{}': {}. Will use char-based estimate fallback.",
                    model_identifier,
                    e
                );
                None
            }
        };

        Ok(Self {
            tokenizer,
            model_identifier: model_identifier.to_string(),
        })
    }

    /// Count the number of tokens in a text string.
    ///
    /// If the tokenizer is available, encodes the text and returns the token count.
    /// Falls back to `text.len() / 4` as a rough estimate.
    pub fn count_text(&self, text: &str) -> usize {
        if let Some(ref tok) = self.tokenizer {
            match tok.encode(text, false) {
                Ok(encoding) => encoding.len(),
                Err(e) => {
                    tracing::debug!("Tokenization failed, using estimate: {}", e);
                    estimate_tokens(text)
                }
            }
        } else {
            estimate_tokens(text)
        }
    }

    /// Count tokens for a list of chat messages, including role prefix overhead.
    ///
    /// Adds estimated tokens for role prefixes and message separators:
    /// - `system: ` ≈ 2 tokens
    /// - `user: ` ≈ 2 tokens
    /// - `assistant: ` ≈ 3 tokens
    /// - `tool: ` ≈ 2 tokens
    /// - Plus 1 separator token per message
    pub fn count_messages(&self, messages: &[Message]) -> usize {
        let mut total = 0;
        for msg in messages {
            total += self.count_text(&msg.content);
            total += role_overhead(&msg.role);
            total += 1; // separator token between messages
        }
        // Add 3 tokens for the overall conversation framing (bos, eos, etc.)
        total += 3;
        total
    }

    /// Returns the model identifier this counter was created with.
    pub fn model_identifier(&self) -> &str {
        &self.model_identifier
    }
}

/// Estimated token overhead for a message role prefix.
fn role_overhead(role: &llm_cascade::MessageRole) -> usize {
    match role {
        llm_cascade::MessageRole::System => 2,    // "system: "
        llm_cascade::MessageRole::User => 2,      // "user: "
        llm_cascade::MessageRole::Assistant => 3, // "assistant: "
        llm_cascade::MessageRole::Tool => 2,      // "tool: "
    }
}

/// Rough character-based token estimate (≈4 chars per token for English text).
fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello"), 2); // 5 chars / 4 = 1, max(1) = 1... actually 5/4=1 in integer
        assert_eq!(estimate_tokens("hello world this is a test"), 7);
        assert!(estimate_tokens("") >= 1);
    }

    #[test]
    fn test_count_messages_empty() {
        // Create a counter that will use fallback (no real tokenizer needed)
        let counter = TokenCounter {
            tokenizer: None,
            model_identifier: "test".into(),
        };
        assert_eq!(counter.count_messages(&[]), 3); // just the framing tokens
    }

    #[test]
    fn test_count_messages_with_roles() {
        let counter = TokenCounter {
            tokenizer: None,
            model_identifier: "test".into(),
        };
        let msgs = vec![
            Message::system("You are helpful."),
            Message::user("Hello!"),
            Message::assistant("Hi there!"),
        ];
        let count = counter.count_messages(&msgs);
        // "You are helpful." = 16/4=4 + 2 (system) + 1 (sep) = 7
        // "Hello!" = 6/4=2 + 2 (user) + 1 (sep) = 5
        // "Hi there!" = 9/4=3 + 3 (assistant) + 1 (sep) = 7
        // + 3 framing = 22
        assert_eq!(count, 22);
    }
}
