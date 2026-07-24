# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Base class for assistant agent implementations.

Defines the abstract interface that all assistant agents must implement,
enabling multiple agent architectures (ReAct, CoT, etc.) to coexist
behind a common invoke contract.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from patient_agent_bench.config import ModelConfig


class BaseAssistantAgent(ABC):
    """
    Abstract base for assistant agent implementations.

    Subclasses implement different agent architectures (ReAct, CoT, etc.)
    while sharing the same invoke contract.

    Subclasses must define a class-level ``NAME`` string used by the
    registry for auto-discovery (e.g. ``NAME = "default"``).
    """

    NAME: str  # Each concrete subclass must set this

    @abstractmethod
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
        Initialize the assistant agent.

        Args:
            model_config: Model configuration (required, must have model_id).
            current_datetime: Current datetime string for temporal context (required).
            tools: List of tools to bind to the agent.
            prompt_name: Name of prompt file to load (without .py extension).
            system_prompt: Custom system prompt. Takes precedence over prompt_name.
            role_arn: Optional AWS IAM role ARN for Bedrock credential management.
        """
        ...

    @abstractmethod
    def invoke(
        self,
        messages: List[BaseMessage],
        user_profile: str,
    ) -> Dict[str, Any]:
        """
        Invoke the agent with a list of messages.

        Args:
            messages: Full conversation history (HumanMessage, AIMessage, ToolMessage).
            user_profile: User profile string for system prompt context.

        Returns:
            Agent result dict with 'messages' key containing full message chain.
        """
        ...

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Get the list of tools bound to this agent."""
        ...
