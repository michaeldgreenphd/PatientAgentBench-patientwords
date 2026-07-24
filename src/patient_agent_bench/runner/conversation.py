# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Conversation data structure for PatientAgentBench.

Wraps LangChain message types with helpers for formatting and extraction.
"""

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)


class Conversation:
    """
    A multi-turn conversation using LangChain message types.

    Stores the full message chain including tool calls and results,
    and provides helpers for extracting formatted views.
    """

    _TYPE_MAP = {
        "human": HumanMessage,
        "ai": AIMessage,
        "tool": ToolMessage,
    }

    def __init__(self, messages: List[BaseMessage] | None = None):
        """Initialize conversation with optional existing messages."""
        self.messages: List[BaseMessage] = messages or []

    @classmethod
    def from_dicts(cls, data: List[Dict[str, Any]]) -> "Conversation":
        """
        Create Conversation from serialized message dicts.

        Used when loading conversations from JSON files (e.g., cmd_evaluate).
        """
        messages: List[BaseMessage] = []
        for item in data:
            msg_type = item.get("type", "")
            msg_class = cls._TYPE_MAP.get(msg_type)
            if msg_class:
                messages.append(msg_class(**item))
        return cls(messages)

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Serialize messages to JSON-serializable list of dicts."""
        return [msg.model_dump() for msg in self.messages]

    def add_message(self, message: BaseMessage) -> None:
        """Add a single message to the conversation."""
        self.messages.append(message)

    def extend_from_agent_result(self, result_messages: List[BaseMessage]) -> None:
        """
        Append new messages from agent result.

        The agent returns the full chain (input + new). This method
        appends only the new messages (those beyond current length).
        Also sanitizes tool_call args to ensure they are dicts, working
        around a langchain-aws bug where string args cause Converse API
        ValidationException.
        """
        current_len = len(self.messages)
        if len(result_messages) > current_len:
            for msg in result_messages[current_len:]:
                self._sanitize_tool_call_args(msg)
                self._ensure_nonempty_content(msg)
            self.messages.extend(result_messages[current_len:])

    @staticmethod
    def _sanitize_tool_call_args(msg: BaseMessage) -> None:
        """Ensure tool_call args are dicts, not strings.

        The Bedrock Converse API requires toolUse.input to be a JSON object.
        Some models return string args which langchain-aws passes through
        without parsing in _upsert_tool_calls_to_bedrock_content.
        """
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return
        for tc in tool_calls:
            args = tc.get("args")
            if isinstance(args, str):
                try:
                    tc["args"] = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    tc["args"] = {}
            elif args is None:
                tc["args"] = {}

    @staticmethod
    def _ensure_nonempty_content(msg: BaseMessage) -> None:
        """Ensure AI messages have non-empty content for Bedrock Converse API.

        Thinking models (e.g., Nova 2 Lite Think) may return only reasoning
        blocks with no text. The Converse API rejects empty content on
        subsequent turns. This adds a minimal text placeholder if needed.
        """
        if not isinstance(msg, AIMessage):
            return
        content = msg.content
        if content:
            # String content or non-empty list — fine
            if isinstance(content, str) and content.strip():
                return
            if isinstance(content, list) and content:
                return
        # Empty content — check if there are tool_calls (those are valid)
        if getattr(msg, "tool_calls", None):
            return
        # Truly empty — add placeholder
        msg.content = "..."

    def set_last_assistant_response_to_user(self, response: str) -> None:
        """
        Store the processed response for the last AIMessage with content.

        Stores in response_metadata['response_shown_to_user_agent'] so it
        persists through serialization. This is the actual text shown to the
        user agent (with thinking content stripped if configured).
        """
        for msg in reversed(self.messages):
            if isinstance(msg, AIMessage) and msg.content:
                msg.response_metadata["response_shown_to_user_agent"] = response
                break

    def set_all_new_assistant_responses(
        self, from_index: int, strip_thinking: bool = False
    ) -> None:
        """
        Set response_shown_to_user_agent on ALL new AIMessages from from_index.

        Each AIMessage with text content gets its own processed text extracted
        and stored. This ensures intermediate messages (text + tool_use) also
        have the field set, not just the final response.

        Args:
            from_index: Index in self.messages to start processing from
            strip_thinking: Whether to strip thinking tags from responses
        """
        for msg in self.messages[from_index:]:
            if not isinstance(msg, AIMessage) or not msg.content:
                continue
            text = self._extract_text_from_content(msg.content)
            if text and strip_thinking:
                text = re.sub(
                    r"<thinking>.*?</thinking>\n?", "", text, flags=re.DOTALL
                )
                text = re.sub(
                    r"<think>.*?</think>\n?", "", text, flags=re.DOTALL
                )
                text = text.strip()
            if text:
                msg.response_metadata["response_shown_to_user_agent"] = text


    def get_all_new_assistant_text(self, from_index: int) -> str:
        """
        Get concatenated text from ALL new AIMessages starting from from_index.

        Combines text from all AI messages in the turn, including intermediate
        messages that contain both text and tool_use blocks. Thinking/reasoning
        blocks are included (caller can strip if needed).

        Args:
            from_index: Index in self.messages to start from

        Returns:
            Concatenated assistant text from all new AI messages
        """
        parts: List[str] = []
        for msg in self.messages[from_index:]:
            if isinstance(msg, AIMessage) and msg.content:
                text = self._extract_text_from_content(msg.content)
                if text:
                    parts.append(text)
        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        """Extract user-visible text from message content (string or list of blocks)."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "text":
                        text_parts.append(block.get("text", ""))
                    elif block_type == "reasoning_content":
                        reasoning = block.get("reasoning_content", {})
                        if isinstance(reasoning, dict) and reasoning.get("text"):
                            text_parts.append(
                                f"<thinking>{reasoning['text']}</thinking>"
                            )
                    elif block_type == "thinking":
                        thinking_text = block.get("thinking", "")
                        if thinking_text:
                            text_parts.append(
                                f"<thinking>{thinking_text}</thinking>"
                            )
            if text_parts:
                return "\n".join(text_parts)
        return ""

    def __len__(self) -> int:
        """Return the number of messages in the conversation."""
        return len(self.messages)
