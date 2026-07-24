# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
User Agent Registry for PatientAgentBench.

Maps agent names to their classes. To add a new agent:
  1. Create a module with a BaseUserAgent subclass that has a NAME attribute.
  2. Import it here and add it to _USER_AGENT_REGISTRY.
"""

from typing import Dict, Optional, Type

from patient_agent_bench.config import AgentSpec
from patient_agent_bench.user_agent.base import BaseUserAgent

# Import all agent classes directly.
from patient_agent_bench.user_agent.default_agent import DefaultUserAgent

_USER_AGENT_REGISTRY: Dict[str, Type[BaseUserAgent]] = {
    DefaultUserAgent.NAME: DefaultUserAgent,
}


def register_user_agent(
    name: str, cls: Type[BaseUserAgent]
) -> None:
    """Register a user agent class (for tests or dynamic additions)."""
    _USER_AGENT_REGISTRY[name] = cls


def get_user_agent_class(name: str) -> Type[BaseUserAgent]:
    """Look up user agent class by name."""
    if name not in _USER_AGENT_REGISTRY:
        available = ", ".join(sorted(_USER_AGENT_REGISTRY.keys()))
        raise KeyError(
            f"Unknown user agent_class '{name}'. Available: {available}"
        )
    return _USER_AGENT_REGISTRY[name]


def create_user_agent_from_spec(
    spec: AgentSpec,
    scenario: str,
    user_profile: str,
    current_datetime: str,
    personality: Optional[str] = None,
    role_arn: Optional[str] = None,
) -> BaseUserAgent:
    """Factory: resolve agent_class from registry and instantiate."""
    agent_cls = get_user_agent_class(spec.agent_class)
    return agent_cls(
        scenario=scenario,
        user_profile=user_profile,
        model_config=spec.model,
        current_datetime=current_datetime,
        prompt_name=spec.prompt,
        system_prompt=spec.system_prompt,
        personality=personality,
        role_arn=role_arn,
    )
