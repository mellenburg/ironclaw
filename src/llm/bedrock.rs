//! Amazon Bedrock LLM provider using the Converse API.
//!
//! Uses the AWS SDK for Rust to call Bedrock's Converse API, which provides
//! a unified interface across model families. Authentication is handled by
//! the standard AWS credential chain (env vars, profile, IMDS, etc.).

use std::sync::RwLock;

use async_trait::async_trait;
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ConversationRole, Message, StopReason, SystemContentBlock, Tool,
    ToolConfiguration, ToolInputSchema, ToolResultBlock, ToolResultContentBlock, ToolSpecification,
    ToolUseBlock,
};
use rust_decimal::Decimal;

use aws_smithy_types::Document;

use crate::config::BedrockConfig;
use crate::error::LlmError;
use crate::llm::costs;
use crate::llm::provider::{
    ChatMessage, CompletionRequest, CompletionResponse, FinishReason, LlmProvider, ModelMetadata,
    Role, ToolCall, ToolCompletionRequest, ToolCompletionResponse, ToolDefinition,
};

/// LLM provider backed by Amazon Bedrock's Converse API.
pub struct BedrockProvider {
    client: aws_sdk_bedrockruntime::Client,
    model_id: String,
    active_model: RwLock<String>,
}

impl BedrockProvider {
    /// Create a new Bedrock provider from config.
    ///
    /// Loads AWS credentials from the standard credential chain
    /// (env vars → profile → IMDS) with optional region/profile overrides.
    pub async fn new(config: &BedrockConfig) -> Result<Self, LlmError> {
        let mut aws_config = aws_config::from_env();

        if let Some(ref region) = config.region {
            aws_config =
                aws_config.region(aws_sdk_bedrockruntime::config::Region::new(region.clone()));
        }

        if let Some(ref profile) = config.profile {
            aws_config = aws_config.profile_name(profile);
        }

        let sdk_config = aws_config.load().await;
        let client = aws_sdk_bedrockruntime::Client::new(&sdk_config);

        tracing::info!(
            model = %config.model,
            region = ?config.region,
            profile = ?config.profile,
            "Initialized Amazon Bedrock provider"
        );

        Ok(Self {
            client,
            model_id: config.model.clone(),
            active_model: RwLock::new(config.model.clone()),
        })
    }

    /// Get the currently active model ID.
    fn current_model(&self) -> String {
        self.active_model
            .read()
            .map(|m| m.clone())
            .unwrap_or_else(|_| self.model_id.clone())
    }

    /// Convert IronClaw messages into Bedrock Message list + system prompts.
    ///
    /// Bedrock Converse takes system prompts as a separate parameter,
    /// and tool results must be wrapped inside User messages.
    fn convert_messages(
        messages: &[ChatMessage],
    ) -> Result<(Vec<SystemContentBlock>, Vec<Message>), LlmError> {
        let mut system_prompts = Vec::new();
        let mut bedrock_messages: Vec<Message> = Vec::new();

        // Accumulator for tool results that need to be grouped into a single User message
        let mut pending_tool_results: Vec<ContentBlock> = Vec::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    system_prompts.push(SystemContentBlock::Text(msg.content.clone()));
                }
                Role::User => {
                    // Flush any pending tool results first
                    if !pending_tool_results.is_empty() {
                        let tool_msg = Message::builder()
                            .role(ConversationRole::User)
                            .set_content(Some(std::mem::take(&mut pending_tool_results)))
                            .build()
                            .map_err(|e| LlmError::RequestFailed {
                                provider: "bedrock".to_string(),
                                reason: format!("Failed to build tool result message: {e}"),
                            })?;
                        bedrock_messages.push(tool_msg);
                    }

                    let user_msg = Message::builder()
                        .role(ConversationRole::User)
                        .content(ContentBlock::Text(msg.content.clone()))
                        .build()
                        .map_err(|e| LlmError::RequestFailed {
                            provider: "bedrock".to_string(),
                            reason: format!("Failed to build user message: {e}"),
                        })?;
                    bedrock_messages.push(user_msg);
                }
                Role::Assistant => {
                    // Flush any pending tool results first
                    if !pending_tool_results.is_empty() {
                        let tool_msg = Message::builder()
                            .role(ConversationRole::User)
                            .set_content(Some(std::mem::take(&mut pending_tool_results)))
                            .build()
                            .map_err(|e| LlmError::RequestFailed {
                                provider: "bedrock".to_string(),
                                reason: format!("Failed to build tool result message: {e}"),
                            })?;
                        bedrock_messages.push(tool_msg);
                    }

                    let mut content_blocks = Vec::new();

                    // Add text content if non-empty
                    if !msg.content.is_empty() {
                        content_blocks.push(ContentBlock::Text(msg.content.clone()));
                    }

                    // Add tool use blocks from assistant tool_calls
                    if let Some(ref tool_calls) = msg.tool_calls {
                        for tc in tool_calls {
                            let input_doc = json_to_document(&tc.arguments);
                            let tool_use = ToolUseBlock::builder()
                                .tool_use_id(&tc.id)
                                .name(&tc.name)
                                .input(input_doc)
                                .build()
                                .map_err(|e| LlmError::RequestFailed {
                                    provider: "bedrock".to_string(),
                                    reason: format!("Failed to build tool use block: {e}"),
                                })?;
                            content_blocks.push(ContentBlock::ToolUse(tool_use));
                        }
                    }

                    // If no content at all, add empty text to satisfy builder
                    if content_blocks.is_empty() {
                        content_blocks.push(ContentBlock::Text(String::new()));
                    }

                    let assistant_msg = Message::builder()
                        .role(ConversationRole::Assistant)
                        .set_content(Some(content_blocks))
                        .build()
                        .map_err(|e| LlmError::RequestFailed {
                            provider: "bedrock".to_string(),
                            reason: format!("Failed to build assistant message: {e}"),
                        })?;
                    bedrock_messages.push(assistant_msg);
                }
                Role::Tool => {
                    // Tool results are ContentBlock::ToolResult inside a User message.
                    // Accumulate them so consecutive tool results can be grouped.
                    let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("unknown");

                    let result_block = ToolResultBlock::builder()
                        .tool_use_id(tool_call_id)
                        .content(ToolResultContentBlock::Text(msg.content.clone()))
                        .build()
                        .map_err(|e| LlmError::RequestFailed {
                            provider: "bedrock".to_string(),
                            reason: format!("Failed to build tool result block: {e}"),
                        })?;
                    pending_tool_results.push(ContentBlock::ToolResult(result_block));
                }
            }
        }

        // Flush any remaining tool results
        if !pending_tool_results.is_empty() {
            let tool_msg = Message::builder()
                .role(ConversationRole::User)
                .set_content(Some(pending_tool_results))
                .build()
                .map_err(|e| LlmError::RequestFailed {
                    provider: "bedrock".to_string(),
                    reason: format!("Failed to build final tool result message: {e}"),
                })?;
            bedrock_messages.push(tool_msg);
        }

        // Bedrock requires alternating user/assistant roles and must start with user.
        // Merge consecutive same-role messages.
        bedrock_messages = merge_consecutive_roles(bedrock_messages);

        Ok((system_prompts, bedrock_messages))
    }

    /// Convert IronClaw tool definitions into Bedrock ToolConfiguration.
    fn convert_tools(tools: &[ToolDefinition]) -> Result<ToolConfiguration, LlmError> {
        let mut tool_specs = Vec::new();

        for tool in tools {
            let input_schema = ToolInputSchema::Json(json_to_document(&tool.parameters));

            let spec = ToolSpecification::builder()
                .name(&tool.name)
                .description(&tool.description)
                .input_schema(input_schema)
                .build()
                .map_err(|e| LlmError::RequestFailed {
                    provider: "bedrock".to_string(),
                    reason: format!("Failed to build tool spec for '{}': {e}", tool.name),
                })?;
            tool_specs.push(Tool::ToolSpec(spec));
        }

        ToolConfiguration::builder()
            .set_tools(Some(tool_specs))
            .build()
            .map_err(|e| LlmError::RequestFailed {
                provider: "bedrock".to_string(),
                reason: format!("Failed to build tool configuration: {e}"),
            })
    }

    /// Extract text content, tool calls, and usage from a Converse response.
    fn extract_response(
        output: &aws_sdk_bedrockruntime::operation::converse::ConverseOutput,
    ) -> Result<(String, Vec<ToolCall>, u32, u32, FinishReason), LlmError> {
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();

        if let Some(aws_sdk_bedrockruntime::types::ConverseOutput::Message(msg)) =
            output.output.as_ref()
        {
            for block in msg.content() {
                match block {
                    ContentBlock::Text(t) => {
                        text_parts.push(t.clone());
                    }
                    ContentBlock::ToolUse(tu) => {
                        let arguments = document_to_json(tu.input());
                        tool_calls.push(ToolCall {
                            id: tu.tool_use_id().to_string(),
                            name: tu.name().to_string(),
                            arguments,
                        });
                    }
                    _ => {}
                }
            }
        }

        let (input_tokens, output_tokens) = if let Some(ref usage) = output.usage {
            (usage.input_tokens() as u32, usage.output_tokens() as u32)
        } else {
            (0, 0)
        };

        let finish_reason = match output.stop_reason() {
            StopReason::EndTurn | StopReason::StopSequence => FinishReason::Stop,
            StopReason::MaxTokens => FinishReason::Length,
            StopReason::ToolUse => FinishReason::ToolUse,
            StopReason::ContentFiltered | StopReason::GuardrailIntervened => {
                FinishReason::ContentFilter
            }
            _ => FinishReason::Unknown,
        };

        Ok((
            text_parts.join(""),
            tool_calls,
            input_tokens,
            output_tokens,
            finish_reason,
        ))
    }

    /// Extract the canonical model name from a Bedrock model ID for cost lookup.
    ///
    /// Bedrock model IDs look like "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    /// or "anthropic.claude-sonnet-4-5-20250929-v1:0". We extract the part after
    /// the provider prefix and strip the version suffix.
    fn canonical_model_name(model_id: &str) -> String {
        // Strip region prefix (e.g., "us." in "us.anthropic.claude-...")
        let without_region = if model_id.starts_with("us.")
            || model_id.starts_with("eu.")
            || model_id.starts_with("ap.")
        {
            &model_id[3..]
        } else {
            model_id
        };

        // Strip provider prefix (e.g., "anthropic." -> "claude-...")
        let without_provider = if let Some(rest) = without_region.strip_prefix("anthropic.") {
            rest
        } else if let Some(rest) = without_region.strip_prefix("meta.") {
            rest
        } else if let Some(rest) = without_region.strip_prefix("amazon.") {
            rest
        } else if let Some(rest) = without_region.strip_prefix("cohere.") {
            rest
        } else if let Some(rest) = without_region.strip_prefix("mistral.") {
            rest
        } else if let Some(rest) = without_region.strip_prefix("ai21.") {
            rest
        } else {
            without_region
        };

        // Strip version suffix (e.g., "-v1:0" or "-v2:0")
        if let Some(idx) = without_provider.rfind("-v") {
            let suffix = &without_provider[idx..];
            // Check it matches pattern -v<digits>:<digits>
            if suffix.len() > 2
                && suffix[2..].contains(':')
                && suffix[2..].chars().all(|c| c.is_ascii_digit() || c == ':')
            {
                return without_provider[..idx].to_string();
            }
        }

        without_provider.to_string()
    }
}

#[async_trait]
impl LlmProvider for BedrockProvider {
    fn model_name(&self) -> &str {
        &self.model_id
    }

    fn active_model_name(&self) -> String {
        self.current_model()
    }

    fn set_model(&self, model: &str) -> Result<(), LlmError> {
        match self.active_model.write() {
            Ok(mut m) => {
                *m = model.to_string();
                Ok(())
            }
            Err(_) => Err(LlmError::RequestFailed {
                provider: "bedrock".to_string(),
                reason: "Failed to acquire model lock".to_string(),
            }),
        }
    }

    fn cost_per_token(&self) -> (Decimal, Decimal) {
        let canonical = Self::canonical_model_name(&self.current_model());
        costs::model_cost(&canonical).unwrap_or_else(costs::default_cost)
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let current = self.current_model();
        let model = request.model.as_deref().unwrap_or(&current);
        let (system_prompts, messages) = Self::convert_messages(&request.messages)?;

        let mut req = self
            .client
            .converse()
            .model_id(model)
            .set_system(if system_prompts.is_empty() {
                None
            } else {
                Some(system_prompts)
            })
            .set_messages(Some(messages));

        if let Some(max_tokens) = request.max_tokens {
            req = req.inference_config(
                aws_sdk_bedrockruntime::types::InferenceConfiguration::builder()
                    .max_tokens(max_tokens as i32)
                    .set_temperature(request.temperature)
                    .build(),
            );
        } else if let Some(temp) = request.temperature {
            req = req.inference_config(
                aws_sdk_bedrockruntime::types::InferenceConfiguration::builder()
                    .max_tokens(4096)
                    .temperature(temp)
                    .build(),
            );
        } else {
            req = req.inference_config(
                aws_sdk_bedrockruntime::types::InferenceConfiguration::builder()
                    .max_tokens(4096)
                    .build(),
            );
        }

        let response = req
            .send()
            .await
            .map_err(|e| converse_sdk_error(e.into_service_error(), model))?;

        let (content, _tool_calls, input_tokens, output_tokens, finish_reason) =
            Self::extract_response(&response)?;

        Ok(CompletionResponse {
            content,
            input_tokens,
            output_tokens,
            finish_reason,
        })
    }

    async fn complete_with_tools(
        &self,
        request: ToolCompletionRequest,
    ) -> Result<ToolCompletionResponse, LlmError> {
        let current = self.current_model();
        let model = request.model.as_deref().unwrap_or(&current);
        let (system_prompts, messages) = Self::convert_messages(&request.messages)?;
        let tool_config = Self::convert_tools(&request.tools)?;

        let mut req = self
            .client
            .converse()
            .model_id(model)
            .set_system(if system_prompts.is_empty() {
                None
            } else {
                Some(system_prompts)
            })
            .set_messages(Some(messages))
            .tool_config(tool_config);

        if let Some(max_tokens) = request.max_tokens {
            req = req.inference_config(
                aws_sdk_bedrockruntime::types::InferenceConfiguration::builder()
                    .max_tokens(max_tokens as i32)
                    .set_temperature(request.temperature)
                    .build(),
            );
        } else if let Some(temp) = request.temperature {
            req = req.inference_config(
                aws_sdk_bedrockruntime::types::InferenceConfiguration::builder()
                    .max_tokens(4096)
                    .temperature(temp)
                    .build(),
            );
        } else {
            req = req.inference_config(
                aws_sdk_bedrockruntime::types::InferenceConfiguration::builder()
                    .max_tokens(4096)
                    .build(),
            );
        }

        let response = req
            .send()
            .await
            .map_err(|e| converse_sdk_error(e.into_service_error(), model))?;

        let (content, tool_calls, input_tokens, output_tokens, finish_reason) =
            Self::extract_response(&response)?;

        Ok(ToolCompletionResponse {
            content: if content.is_empty() {
                None
            } else {
                Some(content)
            },
            tool_calls,
            input_tokens,
            output_tokens,
            finish_reason,
        })
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        // Listing models requires the aws-sdk-bedrock (management API) crate,
        // which is a separate dependency. Return empty to avoid the extra dep.
        Ok(Vec::new())
    }

    async fn model_metadata(&self) -> Result<ModelMetadata, LlmError> {
        Ok(ModelMetadata {
            id: self.current_model(),
            context_length: None,
        })
    }
}

/// Convert an AWS SDK error from the Converse API into an `LlmError`.
///
/// `SdkError::Display` only prints "service error" without detail, so we
/// unwrap the service error to get the actual message from the inner exception.
fn converse_sdk_error(
    e: aws_sdk_bedrockruntime::operation::converse::ConverseError,
    model: &str,
) -> LlmError {
    use aws_sdk_bedrockruntime::operation::converse::ConverseError;
    use aws_smithy_types::error::metadata::ProvideErrorMetadata;

    match &e {
        ConverseError::ThrottlingException(_) => LlmError::RateLimited {
            provider: "bedrock".to_string(),
            retry_after: None,
        },
        ConverseError::AccessDeniedException(_) => LlmError::AuthFailed {
            provider: "bedrock".to_string(),
        },
        ConverseError::ResourceNotFoundException(_) => LlmError::ModelNotAvailable {
            provider: "bedrock".to_string(),
            model: model.to_string(),
        },
        _ => {
            // ConverseError::Unhandled Display just says "unhandled error".
            // Extract the code + message from error metadata for useful diagnostics.
            let meta = ProvideErrorMetadata::meta(&e);
            let code = meta.code().unwrap_or("unknown");
            let message = meta
                .message()
                .map(|m| m.to_string())
                .unwrap_or_else(|| format!("{e}"));
            LlmError::RequestFailed {
                provider: "bedrock".to_string(),
                reason: format!("[{code}] {message}"),
            }
        }
    }
}

/// Convert a `serde_json::Value` to an `aws_smithy_types::Document`.
fn json_to_document(value: &serde_json::Value) -> Document {
    match value {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(b) => Document::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Document::Number(aws_smithy_types::Number::PosInt(u))
            } else if let Some(i) = n.as_i64() {
                Document::Number(aws_smithy_types::Number::NegInt(i))
            } else if let Some(f) = n.as_f64() {
                Document::Number(aws_smithy_types::Number::Float(f))
            } else {
                Document::Null
            }
        }
        serde_json::Value::String(s) => Document::String(s.clone()),
        serde_json::Value::Array(arr) => {
            Document::Array(arr.iter().map(json_to_document).collect())
        }
        serde_json::Value::Object(map) => Document::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), json_to_document(v)))
                .collect(),
        ),
    }
}

/// Convert an `aws_smithy_types::Document` to a `serde_json::Value`.
fn document_to_json(doc: &Document) -> serde_json::Value {
    match doc {
        Document::Null => serde_json::Value::Null,
        Document::Bool(b) => serde_json::Value::Bool(*b),
        Document::Number(n) => match *n {
            aws_smithy_types::Number::PosInt(i) => serde_json::json!(i),
            aws_smithy_types::Number::NegInt(i) => serde_json::json!(i),
            aws_smithy_types::Number::Float(f) => serde_json::Value::Number(
                serde_json::Number::from_f64(f).unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        },
        Document::String(s) => serde_json::Value::String(s.clone()),
        Document::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(document_to_json).collect())
        }
        Document::Object(map) => {
            let obj: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), document_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
    }
}

/// Merge consecutive messages with the same role.
///
/// Bedrock Converse API requires alternating user/assistant roles.
/// When we have consecutive same-role messages (e.g., two User messages),
/// we merge their content blocks into one message.
fn merge_consecutive_roles(messages: Vec<Message>) -> Vec<Message> {
    if messages.is_empty() {
        return messages;
    }

    let mut merged: Vec<Message> = Vec::new();

    for msg in messages {
        let should_merge = merged.last().is_some_and(|last| last.role() == msg.role());

        if should_merge {
            let last = merged.pop().unwrap(); // safe: checked above
            let mut combined_content: Vec<ContentBlock> = last.content().to_vec();
            combined_content.extend(msg.content().to_vec());
            let merged_msg = Message::builder()
                .role(last.role().clone())
                .set_content(Some(combined_content))
                .build();
            match merged_msg {
                Ok(m) => merged.push(m),
                Err(_) => {
                    // If merge fails, keep both separate (shouldn't happen)
                    merged.push(last);
                    merged.push(msg);
                }
            }
        } else {
            merged.push(msg);
        }
    }

    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_model_name_full_bedrock_id() {
        assert_eq!(
            BedrockProvider::canonical_model_name("us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
            "claude-sonnet-4-5-20250929"
        );
    }

    #[test]
    fn test_canonical_model_name_without_region() {
        assert_eq!(
            BedrockProvider::canonical_model_name("anthropic.claude-sonnet-4-5-20250929-v1:0"),
            "claude-sonnet-4-5-20250929"
        );
    }

    #[test]
    fn test_canonical_model_name_short_id() {
        // Some models use short IDs like "anthropic.claude-sonnet-4-6"
        assert_eq!(
            BedrockProvider::canonical_model_name("anthropic.claude-sonnet-4-6"),
            "claude-sonnet-4-6"
        );
    }

    #[test]
    fn test_canonical_model_name_plain() {
        // Already canonical
        assert_eq!(
            BedrockProvider::canonical_model_name("claude-sonnet-4-6"),
            "claude-sonnet-4-6"
        );
    }

    #[test]
    fn test_canonical_model_name_eu_region() {
        assert_eq!(
            BedrockProvider::canonical_model_name("eu.anthropic.claude-opus-4-6-v1:0"),
            "claude-opus-4-6"
        );
    }

    #[test]
    fn test_json_to_document_roundtrip() {
        let original = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "count": 42,
                "active": true,
                "tags": ["a", "b"]
            }
        });
        let doc = json_to_document(&original);
        let back = document_to_json(&doc);
        assert_eq!(original, back);
    }

    #[test]
    fn test_convert_messages_separates_system() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
        ];
        let (system, bedrock_msgs) = BedrockProvider::convert_messages(&messages).unwrap();
        assert_eq!(system.len(), 1);
        assert_eq!(bedrock_msgs.len(), 1);
        assert_eq!(bedrock_msgs[0].role(), &ConversationRole::User);
    }

    #[test]
    fn test_convert_messages_tool_results_grouped() {
        let tc1 = ToolCall {
            id: "call_1".to_string(),
            name: "echo".to_string(),
            arguments: serde_json::json!({}),
        };
        let tc2 = ToolCall {
            id: "call_2".to_string(),
            name: "time".to_string(),
            arguments: serde_json::json!({}),
        };
        let messages = vec![
            ChatMessage::user("do stuff"),
            ChatMessage::assistant_with_tool_calls(None, vec![tc1, tc2]),
            ChatMessage::tool_result("call_1", "echo", "result1"),
            ChatMessage::tool_result("call_2", "time", "result2"),
        ];
        let (_system, bedrock_msgs) = BedrockProvider::convert_messages(&messages).unwrap();
        // Should be: User, Assistant, User (with 2 tool results merged)
        assert_eq!(bedrock_msgs.len(), 3);
        assert_eq!(bedrock_msgs[2].role(), &ConversationRole::User);
        assert_eq!(bedrock_msgs[2].content().len(), 2); // 2 tool results in one message
    }
}
