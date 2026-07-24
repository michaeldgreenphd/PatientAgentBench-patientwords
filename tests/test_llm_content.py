# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for utils/llm_content.py — LLM response content normalization.

Validates that flatten_response_content correctly extracts answer text from
the three channels' polymorphic response.content shapes (str, list-of-blocks).
"""

from patient_agent_bench.utils.llm_content import flatten_response_content


class TestFlattenResponseContent:
    """Validates the shared text-extraction helper across all channel shapes."""

    def test_str_passthrough(self):
        """Plain string content is returned unchanged."""
        assert flatten_response_content("hello") == "hello"

    def test_empty_string(self):
        assert flatten_response_content("") == ""

    def test_none_returns_empty(self):
        assert flatten_response_content(None) == ""

    # -- ChatAnthropic (adaptive thinking) --

    def test_anthropic_thinking_blocks(self):
        """ChatAnthropic with adaptive thinking: [thinking, text] -> text only."""
        content = [
            {"type": "thinking", "thinking": "Let me reason...", "signature": "sig"},
            {"type": "text", "text": '{"answer": 42}'},
        ]
        assert flatten_response_content(content) == '{"answer": 42}'

    def test_anthropic_redacted_thinking(self):
        """Redacted thinking block (no text key) is correctly skipped."""
        content = [
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "result"},
        ]
        assert flatten_response_content(content) == "result"

    # -- ChatBedrockConverse (extended thinking) --

    def test_bedrock_reasoning_content_blocks(self):
        """Bedrock Converse with reasoning: [reasoning_content, text] -> text only.
        reasoning_content block has nested structure, no top-level 'text' key."""
        content = [
            {
                "type": "reasoning_content",
                "reasoning_content": {"text": "internal thinking", "signature": "s"},
            },
            {"type": "text", "text": '{"offices": []}'},
        ]
        assert flatten_response_content(content) == '{"offices": []}'

    # -- ChatOpenAI Responses API --

    def test_openai_responses_api_blocks(self):
        """ChatOpenAI Responses API: [reasoning (no text key), text] -> text only."""
        content = [
            {
                "type": "reasoning",
                "id": "r1",
                "summary": [{"type": "summary_text", "text": "thinking summary"}],
            },
            {
                "type": "text",
                "text": "The answer is 7.",
                "annotations": [],
            },
        ]
        assert flatten_response_content(content) == "The answer is 7."

    # -- Edge cases --

    def test_multiple_text_blocks_joined(self):
        """Multiple text blocks are joined with newline."""
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert flatten_response_content(content) == "first\nsecond"

    def test_list_of_strings(self):
        """Legacy format: list of bare strings."""
        content = ["hello", "world"]
        assert flatten_response_content(content) == "hello\nworld"

    def test_empty_list(self):
        assert flatten_response_content([]) == ""

    def test_output_text_block_type(self):
        """output_text is also a text-carrying block type."""
        content = [{"type": "output_text", "text": "output here"}]
        assert flatten_response_content(content) == "output here"

    def test_tool_use_blocks_skipped(self):
        """tool_use blocks do not carry answer text and should be skipped."""
        content = [
            {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {}},
            {"type": "text", "text": "answer"},
        ]
        assert flatten_response_content(content) == "answer"
