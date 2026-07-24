# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Assistant Agent Registry for PatientAgentBench.

Maps agent names to their classes. To add a new agent:
  1. Create a module with a BaseAssistantAgent subclass that has a NAME attribute.
  2. Import it here and add it to _ASSISTANT_AGENT_REGISTRY.
"""

from typing import Dict, List, Optional, Type

from langchain_core.tools import BaseTool

from patient_agent_bench.assistant_agent.base import BaseAssistantAgent
from patient_agent_bench.config import AgentSpec
from patient_agent_bench.tools.registry import ToolRegistry

# Import all agent classes directly.
from patient_agent_bench.assistant_agent.default_agent import DefaultAssistantAgent

_ASSISTANT_AGENT_REGISTRY: Dict[str, Type[BaseAssistantAgent]] = {
    DefaultAssistantAgent.NAME: DefaultAssistantAgent,
}


def register_assistant_agent(
    name: str, cls: Type[BaseAssistantAgent]
) -> None:
    """Register an assistant agent class (for tests or dynamic additions)."""
    _ASSISTANT_AGENT_REGISTRY[name] = cls


def get_assistant_agent_class(name: str) -> Type[BaseAssistantAgent]:
    """Look up assistant agent class by name."""
    if name not in _ASSISTANT_AGENT_REGISTRY:
        available = ", ".join(sorted(_ASSISTANT_AGENT_REGISTRY.keys()))
        raise KeyError(
            f"Unknown assistant agent_class '{name}'. Available: {available}"
        )
    return _ASSISTANT_AGENT_REGISTRY[name]


def create_assistant_agent_from_spec(
    spec: AgentSpec,
    current_datetime: str,
    tools: Optional[List[BaseTool]] = None,
    tool_registry: Optional[ToolRegistry] = None,
    role_arn: Optional[str] = None,
) -> BaseAssistantAgent:
    """Factory: resolve agent_class from registry and instantiate."""
    agent_cls = get_assistant_agent_class(spec.agent_class)
    if tools is None and tool_registry is not None:
        tools = tool_registry.get_tools()
    return agent_cls(
        model_config=spec.model,
        current_datetime=current_datetime,
        tools=tools,
        prompt_name=spec.prompt,
        system_prompt=spec.system_prompt,
        role_arn=role_arn,
    )
