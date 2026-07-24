# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Default Assistant Agent for PatientAgentBench.

Provides the built-in ReAct-based assistant agent implementation.
Registered as agent_class="default" in the assistant agent registry.
"""

from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from patient_agent_bench.assistant_agent.base import BaseAssistantAgent
from patient_agent_bench.config import (
    ModelConfig, create_chat_model, create_bedrock_client_with_role,
    format_prompt_safe, load_prompt,
)
from patient_agent_bench.logging_config import get_logger
from patient_agent_bench.tools.registry import create_tool_registry
from patient_agent_bench.assistant_agent.default_prompt import SYSTEM_PROMPT as DEFAULT_PROMPT
from patient_agent_bench.utils.retry import retry_sync_with_backoff, LLM_RETRY_CONFIG

logger = get_logger(__name__)


class AssistantAgentError(Exception):
    """Error during assistant agent invocation."""


class DefaultAssistantAgent(BaseAssistantAgent):
    """
    Default assistant agent using LangGraph ReAct pattern.
    """

    NAME = "default"

    def __init__(
        self,
        model_config: ModelConfig,
        current_datetime: str,
        tools: Optional[List[BaseTool]] = None,
        prompt_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        role_arn: Optional[str] = None,
    ) -> None:
        """
        Initialize the default assistant agent.

        Args:
            model_config: Model configuration (required, must have model_id).
            current_datetime: Current datetime string for temporal context (required).
                            Format: "Tuesday, February 10, 2026 at 2:35 PM"
            tools: List of tools to bind to the agent.
            prompt_name: Name of prompt file to load (without .py extension).
            system_prompt: Custom system prompt. Takes precedence over prompt_name.
            role_arn: Optional AWS IAM role ARN for Bedrock credential management.

        Raises:
            ValueError: If model_config.model_id or current_datetime is not set.
        """
        if not model_config.model_id:
            raise ValueError("model_config.model_id is required")
        if not current_datetime:
            raise ValueError("current_datetime is required")

        self.model_config = model_config
        self._role_arn = role_arn
        self.tools = tools or create_tool_registry().get_tools()
        self.current_datetime = current_datetime

        # Determine which prompt to use (priority: system_prompt > prompt_name > default)
        if system_prompt is not None:
            self.system_prompt_template = system_prompt
        elif prompt_name is not None:
            self.system_prompt_template = load_prompt("assistant_agent", prompt_name)
        else:
            self.system_prompt_template = DEFAULT_PROMPT

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
            self.model_config, self._bedrock_client, role_arn=self._role_arn
        )

    def _refresh_llm(self) -> None:
        """Recreate the bedrock client and LLM after credential refresh."""
        if self.model_config.requires_bedrock:
            self._bedrock_client = self._create_bedrock_client()
            self.llm = self._create_llm()
            logger.debug("Refreshed LLM client for DefaultAssistantAgent")

    def _create_agent(self, user_profile: str) -> Any:
        """
        Create the agent with the given user profile in system prompt.

        Args:
            user_profile: User profile string to inject into system prompt

        Returns:
            Compiled agent graph
        """
        system_prompt = format_prompt_safe(
            self.system_prompt_template,
            user_profile=user_profile,
            current_datetime=self.current_datetime,
        )

        if self.model_config.thinking_prompt_suffix:
            system_prompt = system_prompt + "\n" + self.model_config.thinking_prompt_suffix

        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
        )

    def invoke(
        self,
        messages: List[BaseMessage],
        user_profile: str,
    ) -> Dict[str, Any]:
        """
        Invoke the agent with a list of messages.

        Creates a fresh agent graph for each invocation with the user profile
        injected into the system prompt.

        Args:
            messages: Full conversation history (HumanMessage, AIMessage, ToolMessage)
            user_profile: User profile string for system prompt context

        Returns:
            Agent result dict with 'messages' key containing full message chain

        Raises:
            AssistantAgentError: If agent invocation fails
        """
        try:
            result = retry_sync_with_backoff(
                lambda msg_input: self._create_agent(user_profile).invoke(msg_input),
                {"messages": messages},
                config=LLM_RETRY_CONFIG,
                on_credential_refresh=self._refresh_llm,
            )
            return dict(result)  # type: ignore[arg-type]

        except Exception as e:
            logger.error("Agent invocation failed: %s", e)
            raise AssistantAgentError(str(e)) from e

    def get_tools(self) -> List[BaseTool]:
        """Get the list of tools bound to this agent."""
        return self.tools
