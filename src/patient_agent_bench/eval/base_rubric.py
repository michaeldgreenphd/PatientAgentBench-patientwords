# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Base Rubric for PatientAgentBench Evaluation.

Provides the abstract base class for all evaluation rubrics.
"""

import json
import re
from abc import ABC
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from patient_agent_bench.config import (
    ModelConfig, create_chat_model, create_bedrock_client_with_role,
)
from patient_agent_bench.eval.constants import (
    PASS_THRESHOLD, MAX_SCORE, MIN_SCORE, SCORE_LABELS, SCORE_RANGE,
)
from patient_agent_bench.logging_config import get_logger
from patient_agent_bench.runner.conversation import Conversation
from patient_agent_bench.utils.retry import retry_sync_with_backoff, LLM_RETRY_CONFIG

logger = get_logger(__name__)


def _flatten_response_content(content: Any) -> str:
    """Flatten an LLM response's ``content`` to plain text.

    Most chat models return a string, but the OpenAI Responses API (used by the
    Mantle GPT-5.x models via ``use_responses_api``) returns a list of content
    blocks. Concatenate the text of those blocks so downstream JSON parsing gets
    a string instead of a list (which otherwise raises "expected string or
    bytes-like object, got 'list'").
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                # Text blocks carry the model's actual answer; skip reasoning /
                # tool blocks (evaluators only need the JSON-bearing text).
                if block.get("type") in ("text", "output_text") or "text" in block:
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return str(content) if content is not None else ""


class BaseRubric(ABC):
    """
    Abstract base class for evaluation rubrics.

    All rubrics follow a 1-5 scoring scale:
    - 1: Fail (clinically significant issue or critical error)
    - 2: Poor (major gaps in performance)
    - 3: Adequate (baseline competence, minor issues)
    - 4: Good (solid performance, only minor gaps)
    - 5: Excellent (comprehensive, all requirements met)

    Pass rate: score >= PASS_THRESHOLD.
    """

    # Subclasses must define these
    RUBRIC_NAME: str = "base"
    EVALUATION_PROMPT: str = ""
    WEIGHT: float = 1.0  # Weight for aggregate score calculation

    def __init__(self, model_config: ModelConfig, role_arn=None):
        """
        Initialize the rubric.

        Args:
            model_config: Model configuration (required).
            role_arn: Optional AWS IAM role ARN for Bedrock credential management.
                     Passed to create_bedrock_client_with_role() to create the
                     client. None uses base credentials.
        """
        self.model_config = model_config
        self._role_arn = role_arn
        self._bedrock_client = self._create_bedrock_client()
        self.llm = self._create_llm()

    def _create_bedrock_client(self):
        """Create a bedrock client if needed for this model config."""
        if not self.model_config.requires_bedrock:
            return None
        return create_bedrock_client_with_role(self._role_arn)

    def _create_llm(self) -> BaseChatModel:
        """Create the LLM instance via the centralized factory."""
        return create_chat_model(
            self.model_config, self._bedrock_client
        )

    def _refresh_llm(self) -> None:
        """Recreate the bedrock client and LLM after credential refresh."""
        if self.model_config.requires_bedrock:
            self._bedrock_client = self._create_bedrock_client()
            self.llm = self._create_llm()
            logger.debug("Refreshed LLM client for rubric %s", self.RUBRIC_NAME)

    def format_conversation(self, conversation_history: Conversation) -> str:
        """
        Format a conversation for inclusion in the evaluation prompt.

        Handles all message types including tool calls and tool results.
        Uses XML tags with message indices for clear structure.
        Includes response_shown_to_user_agent when available to show what
        the user agent actually saw (with thinking content stripped).

        Args:
            conversation_history: Conversation object containing messages

        Returns:
            Formatted conversation string with XML structure
        """
        formatted = []
        for idx, msg in enumerate(conversation_history.messages):
            content = msg.content

            # Handle content that's a list (multi-block: thinking + tool_use)
            # When tool_calls attribute is present, it is the canonical source
            # for tool call data — skip tool_use content blocks to avoid
            # rendering each tool call twice.
            if isinstance(content, list):
                has_tool_calls_attr = bool(getattr(msg, "tool_calls", None))
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use" and not has_tool_calls_attr:
                            tool_name = block.get("name", "unknown")
                            tool_input = block.get("input", {})
                            text_parts.append(
                                f'<tool_call name="{tool_name}">{tool_input}</tool_call>'
                            )
                content = "\n".join(text_parts)

            # Format based on message type with XML tags
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "unknown")
                formatted.append(
                    f'<message index="{idx}" role="tool_result" tool="{tool_name}">'
                    f'\n{content}\n</message>'
                )
            elif isinstance(msg, AIMessage):
                # Check for response_shown_to_user_agent in response_metadata
                response_metadata = getattr(msg, "response_metadata", {})
                shown_to_user = response_metadata.get("response_shown_to_user_agent")

                # Include tool_calls if present
                tool_calls = getattr(msg, "tool_calls", [])
                if tool_calls:
                    tool_xml = "\n".join(
                        f'<tool_call name="{tc.get("name", "unknown")}">'
                        f'{tc.get("args", {})}</tool_call>'
                        for tc in tool_calls
                    )
                    formatted.append(
                        f'<message index="{idx}" role="assistant">'
                        f'\n{content}\n{tool_xml}\n</message>'
                    )
                elif shown_to_user:
                    # Include what was actually shown to user agent
                    formatted.append(
                        f'<message index="{idx}" role="assistant">'
                        f'\n<response_shown_to_user_agent>\n{shown_to_user}\n'
                        f'</response_shown_to_user_agent>\n</message>'
                    )
                else:
                    formatted.append(
                        f'<message index="{idx}" role="assistant">\n{content}\n</message>'
                    )
            elif isinstance(msg, HumanMessage):
                formatted.append(
                    f'<message index="{idx}" role="user">\n{content}\n</message>'
                )
            else:
                # SystemMessage or other types
                role = msg.__class__.__name__.replace("Message", "").lower()
                formatted.append(
                    f'<message index="{idx}" role="{role}">\n{content}\n</message>'
                )

        return "\n\n".join(formatted)

    def prepare_prompt(
        self,
        conversation_history: Conversation,
        user_profile: str,
        scenario: Optional[str] = None,
    ) -> str:
        """
        Prepare the evaluation prompt with conversation and context.

        Args:
            conversation_history: The Conversation to evaluate
            user_profile: Patient profile information
            scenario: Optional scenario description

        Returns:
            Formatted prompt string
        """
        formatted_conversation = self.format_conversation(conversation_history)

        prompt = self.EVALUATION_PROMPT.format(
            conversation=formatted_conversation,
            user_profile=user_profile,
            scenario=scenario or "Not provided",
        )

        # Optional reasoning-mode control for the judge model, mirroring the
        # assistant agents. For Qwen3 models this is "/think" (force reasoning)
        # or "/no_think" (disable reasoning — faster, no <think> block). Appended
        # to the evaluation prompt so it reaches judges that read the suffix.
        if self.model_config.thinking_prompt_suffix:
            prompt = prompt + "\n" + self.model_config.thinking_prompt_suffix

        return prompt

    def call_llm(self, prompt: str) -> str:
        """
        Call the LLM with the evaluation prompt.

        Args:
            prompt: The prepared evaluation prompt

        Returns:
            Raw LLM response string
        """
        messages = [HumanMessage(content=prompt)]

        try:
            response = retry_sync_with_backoff(
                lambda msgs: self.llm.invoke(msgs),
                messages,
                config=LLM_RETRY_CONFIG,
                on_credential_refresh=self._refresh_llm,
            )
            # Flatten list-shaped content (Responses API / Mantle GPT-5.x) to a
            # string so parse_response's JSON regex works across all providers.
            return _flatten_response_content(response.content)
        except Exception as e:
            logger.error(f"Error calling LLM for evaluation: {e}")
            raise

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract structured evaluation results.

        Args:
            response: Raw LLM response string

        Returns:
            Dictionary with score and explanation
        """
        # Try to extract JSON from the response
        try:
            # Look for JSON block in the response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                result = json.loads(json_match.group())
                return self._validate_result(result)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from response: {response[:200]}...")

        # Fallback: try to extract score from text
        score = self._extract_score_from_text(response)
        return {
            "score": score,
            "explanation": response,
        }

    def _validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize the parsed result.

        Args:
            result: Parsed result dictionary

        Returns:
            Validated result with required fields
        """
        # Ensure score is in valid range
        score = result.get("score", PASS_THRESHOLD)
        if isinstance(score, (int, float)):
            score = max(MIN_SCORE, min(MAX_SCORE, int(score)))
        else:
            score = PASS_THRESHOLD

        return {
            "score": score,
            "explanation": result.get("explanation", ""),
        }

    def _extract_score_from_text(self, text: str) -> int:
        """
        Extract a score from text when JSON parsing fails.

        Args:
            text: Response text

        Returns:
            Extracted score (defaults to 2 if not found)
        """
        text_lower = text.lower()

        # Build score patterns dynamically from constants
        score_patterns = []
        # Numeric patterns (check higher scores first)
        for s in reversed(SCORE_RANGE):
            score_patterns.append((rf"score[:\s]+{s}", s))
        # Label patterns
        label_to_score = {v.lower(): k for k, v in SCORE_LABELS.items()}
        for label, s in label_to_score.items():
            score_patterns.append((rf"\b{re.escape(label)}\b", s))

        for pattern, score in score_patterns:
            if re.search(pattern, text_lower):
                return score

        return PASS_THRESHOLD  # Default to pass threshold

    def evaluate(
        self,
        conversation_history: Conversation,
        user_profile: str,
        scenario: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a conversation using this rubric.

        Args:
            conversation_history: The Conversation to evaluate
            user_profile: Patient profile information
            scenario: Optional scenario description

        Returns:
            Evaluation result with score, explanation, and sub_scores
        """
        prompt = self.prepare_prompt(conversation_history, user_profile, scenario)
        response = self.call_llm(prompt)
        result = self.parse_response(response)

        # Add rubric metadata
        result["rubric_name"] = self.RUBRIC_NAME
        result["raw_response"] = response

        return result

    @property
    def name(self) -> str:
        """Get the rubric name."""
        return self.RUBRIC_NAME

    @property
    def weight(self) -> float:
        """Get the rubric weight for aggregate scoring."""
        return self.WEIGHT
