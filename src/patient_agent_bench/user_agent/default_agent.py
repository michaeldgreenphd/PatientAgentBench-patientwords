# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Default User Agent for PatientAgentBench.

Provides the built-in LLM-based patient simulation agent implementation.
Registered as agent_class="default" in the user agent registry.
"""

from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from patient_agent_bench.config import (
    ModelConfig, create_chat_model, create_bedrock_client_with_role,
    format_prompt_safe, load_prompt,
)
from patient_agent_bench.logging_config import get_logger
from patient_agent_bench.user_agent.base import BaseUserAgent
from patient_agent_bench.user_agent.default_prompt import SYSTEM_PROMPT as DEFAULT_PROMPT
from patient_agent_bench.user_agent.personalities import get_personality_prompt
from patient_agent_bench.utils.retry import retry_sync_with_backoff, LLM_RETRY_CONFIG

logger = get_logger(__name__)


class DefaultUserAgent(BaseUserAgent):
    """
    Default user agent using LLM-based patient simulation.
    """

    NAME = "default"

    def __init__(
        self,
        scenario: str,
        user_profile: str,
        model_config: ModelConfig,
        current_datetime: str,
        prompt_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        personality: Optional[str] = None,
        role_arn: Optional[str] = None,
    ) -> None:
        """
        Initialize the default user agent.

        Args:
            scenario: The patient scenario describing the situation
            user_profile: Patient profile information
            model_config: Model configuration (required).
            current_datetime: Current datetime string for temporal context (required).
                            Format: "Tuesday, February 10, 2026 at 2:35 PM"
            prompt_name: Name of prompt file to load (without .py extension).
                        Defaults to "default_prompt" if neither prompt_name nor system_prompt provided.
            system_prompt: Custom system prompt template. Takes precedence over prompt_name.
            role_arn: Optional AWS IAM role ARN for Bedrock credential management.

        Raises:
            ValueError: If current_datetime is not provided.
        """
        if not current_datetime:
            raise ValueError("current_datetime is required")

        self.scenario = scenario
        self.user_profile = user_profile
        self.personality_type = personality
        self.model_config = model_config
        self._role_arn = role_arn
        self.current_datetime = current_datetime

        # Determine which prompt to use (priority: system_prompt > prompt_name > default)
        if system_prompt is not None:
            self.system_prompt_template = system_prompt
        elif prompt_name is not None:
            self.system_prompt_template = load_prompt("user_agent", prompt_name)
        else:
            self.system_prompt_template = DEFAULT_PROMPT

        # Format the system prompt using format_prompt_safe
        self.system_prompt = format_prompt_safe(
            self.system_prompt_template,
            scenario=scenario,
            user_profile=user_profile,
            current_datetime=current_datetime,
            personality_traits=get_personality_prompt(personality),  # type: ignore[arg-type]
        )

        # Initialize the LLM
        self._bedrock_client = self._create_bedrock_client()
        self.llm = self._create_llm()

        # Conversation history
        self.chat_history: List[Dict[str, str]] = []

    def _create_bedrock_client(self):
        """Create a bedrock client if needed for this model config."""
        if not self.model_config.requires_bedrock:
            return None
        return create_bedrock_client_with_role(self._role_arn)

    def _create_llm(self) -> BaseChatModel:
        """Create the LLM instance via the centralized factory."""
        return create_chat_model(self.model_config, self._bedrock_client)

    def _refresh_llm(self) -> None:
        """Recreate the bedrock client and LLM after credential refresh."""
        if self.model_config.requires_bedrock:
            self._bedrock_client = self._create_bedrock_client()
            self.llm = self._create_llm()
            logger.debug("Refreshed LLM client for DefaultUserAgent")

    def invoke(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """
        Invoke the agent with a list of messages.

        Generates a patient response based on the full conversation history.
        This method satisfies the BaseUserAgent contract.

        Args:
            messages: Full conversation history (HumanMessage, AIMessage).

        Returns:
            The patient's response as a string.
        """
        # Build messages with system prompt prepended
        llm_messages: List[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        llm_messages.extend(messages)

        response = retry_sync_with_backoff(
            lambda msgs: self.llm.invoke(msgs),
            llm_messages,
            config=LLM_RETRY_CONFIG,
            on_credential_refresh=self._refresh_llm,
        )
        return (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

    def start_conversation(self) -> str:
        """
        Generate the initial message to start the conversation.

        Following the BedrockLLMMockPatient pattern:
        - role: "user" = health assistant's messages (input to patient LLM)
        - role: "assistant" = patient's messages (output from patient LLM)

        Returns:
            The initial user message based on the scenario

        Raises:
            Exception: If LLM fails to generate initial message
        """
        # Add an opening prompt as "user" (from patient LLM's perspective, this is input)
        opening_prompt = "What can I help you with today?"
        self.chat_history.append({"role": "user", "content": opening_prompt})

        # Build messages for the LLM
        messages = self._build_messages()

        response = retry_sync_with_backoff(
            lambda msgs: self.llm.invoke(msgs),
            messages,
            config=LLM_RETRY_CONFIG,
            on_credential_refresh=self._refresh_llm,
        )
        # Handle response.content which can be str or list
        initial_message = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        # Store patient's response as "assistant" (from patient LLM's perspective)
        self.chat_history.append({"role": "assistant", "content": initial_message})

        return initial_message

    def respond(self, assistant_message: str) -> str:
        """
        Generate a response to the assistant's message.

        Following the BedrockLLMMockPatient pattern:
        - role: "user" = health assistant's messages (input to patient LLM)
        - role: "assistant" = patient's messages (output from patient LLM)

        Args:
            assistant_message: The message from the health AI assistant

        Returns:
            The patient's response
        """
        # Add health assistant's message as "user" (from patient LLM's perspective, this is input)
        self.chat_history.append({"role": "user", "content": assistant_message})

        # Build messages for the LLM
        messages = self._build_messages()

        try:
            response = retry_sync_with_backoff(
                lambda msgs: self.llm.invoke(msgs),
                messages,
                config=LLM_RETRY_CONFIG,
                on_credential_refresh=self._refresh_llm,
            )
            # Handle response.content which can be str or list
            user_response = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )

            # Store patient's response as "assistant" (from patient LLM's perspective)
            self.chat_history.append({"role": "assistant", "content": user_response})

            return user_response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm not sure what to say."

    def _build_messages(self) -> List[BaseMessage]:
        """
        Build messages for the LLM from chat history.

        Returns:
            List of LangChain messages [SystemMessage, HumanMessage, AIMessage, ...]
        """
        messages: List[BaseMessage] = [SystemMessage(content=self.system_prompt)]

        # Convert chat history to LangChain message types
        # role: "user" -> HumanMessage (input to patient LLM)
        # role: "assistant" -> AIMessage (output from patient LLM)
        for msg in self.chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        return messages

    def is_conversation_complete(self) -> bool:
        """
        Check if the conversation appears to be complete.

        Returns:
            True if the patient signaled end of conversation
        """
        if not self.chat_history:
            return False

        # Check if the last patient message is the drop-off signal
        for msg in reversed(self.chat_history):
            if msg["role"] == "assistant":
                return self.CONVERSATION_END_SIGNAL in msg["content"]
            break

        return False

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.chat_history.copy()

    def reset(self) -> None:
        """Reset the conversation history."""
        self.chat_history = []
