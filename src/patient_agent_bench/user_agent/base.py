# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Base class for user agent implementations.

Defines the abstract interface that all user (simulated patient) agents must
implement, enabling multiple patient simulation strategies to coexist behind
a common invoke contract.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.messages import BaseMessage

from patient_agent_bench.config import ModelConfig


class BaseUserAgent(ABC):
    """
    Abstract base for user agent implementations.

    Subclasses implement different patient simulation strategies
    while sharing the same invoke contract.

    Subclasses must define a class-level ``NAME`` string used by the
    registry for auto-discovery (e.g. ``NAME = "default"``).
    """

    NAME: str  # Each concrete subclass must set this

    # Signal the user agent emits when the patient has no further questions.
    # Detected by the conversation runner to end the conversation loop.
    CONVERSATION_END_SIGNAL = "[DROPPED OFF CONVERSATION]"

    @abstractmethod
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
        Initialize the user agent.

        Args:
            scenario: The patient scenario describing the situation (required).
            user_profile: Patient profile information (required).
            model_config: Model configuration (required).
            current_datetime: Current datetime string for temporal context (required).
            prompt_name: Name of prompt file to load (without .py extension).
            system_prompt: Custom system prompt. Takes precedence over prompt_name.
            personality: Personality trait profile (trait -> level mapping).
            role_arn: Optional AWS IAM role ARN for Bedrock credential management.
        """
        ...

    @abstractmethod
    def invoke(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """
        Invoke the agent with a list of messages.

        Args:
            messages: Full conversation history (HumanMessage, AIMessage).

        Returns:
            The patient's response as a string.
        """
        ...

    @abstractmethod
    def start_conversation(self) -> str:
        """
        Generate the patient's opening message.

        Called once by the conversation runner to begin the exchange, before
        the assistant has said anything.

        Returns:
            The patient's first message, based on the scenario.
        """
        ...

    @abstractmethod
    def respond(self, assistant_message: str) -> str:
        """
        Generate the patient's reply to an assistant message.

        Args:
            assistant_message: The message from the health AI assistant.

        Returns:
            The patient's response as a string.
        """
        ...

    @abstractmethod
    def is_conversation_complete(self) -> bool:
        """
        Report whether the patient has signaled the end of the conversation.

        Checked by the conversation runner after each turn to decide whether to
        stop; implementations typically look for ``CONVERSATION_END_SIGNAL`` in
        the patient's most recent message.

        Returns:
            True if the conversation should end.
        """
        ...
