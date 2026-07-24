# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Conversation Runner for PatientAgentBench.

Orchestrates multi-turn conversations between assistant and user agents
with parallel execution support.
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage

from patient_agent_bench.assistant_agent.default_agent import AssistantAgentError
from patient_agent_bench.assistant_agent.registry import create_assistant_agent_from_spec
from patient_agent_bench.config import create_bedrock_client_with_role
from patient_agent_bench.logging_config import get_logger
from patient_agent_bench.benchmark_seed import BenchmarkEntry
from patient_agent_bench.runner.experiment_config import ExperimentConfig
from patient_agent_bench.runner.conversation import Conversation
from patient_agent_bench.sandbox import HealthcareSandbox, create_sandbox_llm, initialize_sandbox
from patient_agent_bench.tools.registry import create_tool_registry
from patient_agent_bench.user_agent.base import BaseUserAgent
from patient_agent_bench.user_agent.registry import create_user_agent_from_spec

logger = get_logger(__name__)


@dataclass
class ConversationResult:
    """Result of a single conversation run."""

    case_id: str
    conversation: Conversation
    user_profile: str
    scenario: str
    num_turns: int
    personality: str = ""
    error: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Only call this at file write boundary - prefer using .conversation
        directly when passing to evaluator or other in-memory operations.
        """
        result: Dict[str, Any] = {
            "case_id": self.case_id,
            "conversation": self.conversation.to_dicts(),
            "user_profile": self.user_profile,
            "scenario": self.scenario,
            "num_turns": self.num_turns,
            "personality": self.personality,
        }
        if self.error is not None:
            result["error"] = self.error
        if self.evaluation is not None:
            result["evaluation"] = self.evaluation
        return result


class ConversationRunner:
    """
    Orchestrates conversations between assistant and user agents.

    Designed for parallel execution with role-based credential isolation.
    """

    def __init__(self, experiment: ExperimentConfig):
        """
        Initialize the conversation runner.

        Args:
            experiment: Experiment configuration with all settings
        """
        self.experiment = experiment

    async def run_conversation_async(
        self,
        entry: BenchmarkEntry,
        assigned_role: Optional[str] = None,
    ) -> ConversationResult:
        """
        Run a conversation asynchronously with role-based credential isolation.

        Args:
            entry: The benchmark entry to run
            assigned_role: AWS ARN role to use for this conversation

        Returns:
            ConversationResult with the conversation and metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(self._run_conversation, entry, assigned_role),
        )

    def _run_conversation(
        self,
        entry: BenchmarkEntry,
        assigned_role: Optional[str] = None,
    ) -> ConversationResult:
        """
        Internal: Run a conversation with role-based credentials.

        Args:
            entry: The benchmark entry to run
            assigned_role: AWS ARN role for credential management

        Returns:
            ConversationResult with the conversation and metadata
        """
        max_turns = self.experiment.max_turns

        # Capture current datetime once for consistent temporal context across agents
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

        # Create bedrock client for sandbox (one-shot, doesn't need refresh)
        sandbox_bedrock_client = None
        if self.experiment.sandbox_model.requires_bedrock:
            sandbox_bedrock_client = create_bedrock_client_with_role(assigned_role)

        # Create and initialize sandbox with current_datetime for temporal grounding
        sandbox = HealthcareSandbox()
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            llm_client = create_sandbox_llm(self.experiment.sandbox_model, sandbox_bedrock_client)
            loop.run_until_complete(
                initialize_sandbox(
                    sandbox=sandbox,
                    patient_profile=entry.patient_profile,
                    llm_client=llm_client,
                    current_datetime=current_datetime,
                )
            )
        finally:
            loop.close()

        # Create tool registry with sandbox
        tool_registry = create_tool_registry(sandbox)

        # Create assistant agent with shared datetime
        assistant_agent = create_assistant_agent_from_spec(
            spec=self.experiment.assistant_agent,
            current_datetime=current_datetime,
            tool_registry=tool_registry,
            role_arn=assigned_role,
        )

        # Create user agent with shared datetime
        user_agent = create_user_agent_from_spec(
            spec=self.experiment.user_agent,
            scenario=entry.scenario,
            user_profile=entry.patient_profile_xml,
            current_datetime=current_datetime,
            personality=entry.personality,
            role_arn=assigned_role,
        )

        conversation = Conversation()
        error_message: Optional[str] = None

        # Start conversation with initial user message
        user_message = user_agent.start_conversation()
        conversation.add_message(HumanMessage(content=user_message))

        # Run conversation turns
        for turn in range(max_turns):
            logger.info("Turn %d/%d", turn + 1, max_turns)

            try:
                # Invoke agent with full message history
                result = assistant_agent.invoke(
                    messages=conversation.messages,
                    user_profile=entry.patient_profile_xml,
                )

                # Append only new messages from agent result
                pre_extend_len = len(conversation)
                conversation.extend_from_agent_result(result["messages"])

            except AssistantAgentError as e:
                # Log error, add error AIMessage, and stop conversation
                logger.error("Conversation stopped due to agent error: %s", e)
                error_message = str(e)
                conversation.add_message(
                    AIMessage(content=f"[Conversation ended due to error: {e}]")
                )
                break

            # Set response_shown_to_user_agent on ALL new AI messages from this turn
            conversation.set_all_new_assistant_responses(
                from_index=pre_extend_len,
                strip_thinking=self.experiment.strip_thinking_content,
            )

            # Get full response for user agent (all AI text from this turn)
            assistant_response = conversation.get_all_new_assistant_text(pre_extend_len)
            if self.experiment.strip_thinking_content:
                assistant_response = self._strip_thinking_content(assistant_response)
                logger.debug("Stripped response for user agent: %s", assistant_response)

            if user_agent.is_conversation_complete():
                logger.info("Conversation completed after %d turns", turn + 1)
                break

            if turn < max_turns - 1:
                user_message = user_agent.respond(assistant_response)

                # Check if user agent signaled end of conversation
                if BaseUserAgent.CONVERSATION_END_SIGNAL in user_message:
                    logger.info(
                        "User agent dropped off conversation after %d turns", turn + 1
                    )
                    conversation.add_message(HumanMessage(content=user_message))
                    break

                conversation.add_message(HumanMessage(content=user_message))

        return ConversationResult(
            case_id=entry.id,
            conversation=conversation,
            user_profile=entry.patient_profile_xml,
            scenario=entry.scenario,
            num_turns=len(conversation) // 2,
            personality=entry.personality or "",
            error=error_message,
        )

    def _strip_thinking_content(self, text: str) -> str:
        """Remove <thinking>...</thinking> and <think>...</think> blocks from text."""
        text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        return text
