# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tool Registry for PatientAgentBench.

Provides centralized tool registration and management for the assistant agent.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

from langchain_core.tools import BaseTool

from patient_agent_bench.logging_config import get_logger
from patient_agent_bench.tools.appointment_tools import get_appointment_tools
from patient_agent_bench.tools.prescription_tools import get_prescription_tools
from patient_agent_bench.tools.profile_tools import get_profile_tools
from patient_agent_bench.tools.telehealth_tools import get_telehealth_tools

if TYPE_CHECKING:
    from patient_agent_bench.sandbox import HealthcareSandbox

logger = get_logger(__name__)


class ToolRegistry:
    """
    Central registry for managing tools available to the assistant agent.

    Tools can be registered individually or in bulk, and retrieved by name
    or as a complete list for agent binding.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, tool: BaseTool, category: Optional[str] = None) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: The tool to register (must have a 'name' attribute)
            category: Optional category for grouping tools
        """
        if not hasattr(tool, "name"):
            raise ValueError("Tool must have a 'name' attribute")

        tool_name = tool.name
        self._tools[tool_name] = tool
        logger.debug(f"Registered tool: {tool_name}")

        if category:
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(tool_name)

    def register_many(self, tools: List[BaseTool], category: Optional[str] = None) -> None:
        """
        Register multiple tools at once.

        Args:
            tools: List of tools to register
            category: Optional category for all tools
        """
        for tool in tools:
            self.register(tool, category)

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            The tool if found, None otherwise
        """
        return self._tools.get(tool_name)

    def get_tools(self, tool_names: Optional[List[str]] = None) -> List[BaseTool]:
        """
        Get tools by name or all tools if no names specified.

        Args:
            tool_names: Optional list of tool names to retrieve

        Returns:
            List of tools
        """
        if tool_names is None:
            return list(self._tools.values())

        tools = []
        for name in tool_names:
            tool = self._tools.get(name)
            if tool:
                tools.append(tool)
            else:
                logger.warning(f"Tool not found: {name}")
        return tools

    def get_by_category(self, category: str) -> List[BaseTool]:
        """
        Get all tools in a category.

        Args:
            category: Category name

        Returns:
            List of tools in the category
        """
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def list_tools(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def list_categories(self) -> List[str]:
        """
        List all categories.

        Returns:
            List of category names
        """
        return list(self._categories.keys())

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools


def create_tool_registry(sandbox: Optional["HealthcareSandbox"] = None) -> ToolRegistry:
    """
    Create a new tool registry pre-populated with all standard tools.

    All tools require a sandbox instance for state management. If no sandbox
    is provided, an empty registry is returned.

    Args:
        sandbox: HealthcareSandbox instance for sandbox-aware tools.
                 Required for tool registration.

    Returns:
        A new ToolRegistry instance with all tools registered
    """
    registry = ToolRegistry()

    # All tools require sandbox for state management
    if sandbox is None:
        logger.warning("No sandbox provided - returning empty registry")
        return registry

    # Register sandbox-aware tools from each category
    registry.register_many(get_appointment_tools(sandbox), category="appointment")
    registry.register_many(get_prescription_tools(sandbox), category="prescription")
    registry.register_many(get_profile_tools(sandbox), category="profile")
    registry.register_many(get_telehealth_tools(sandbox), category="telehealth")

    logger.info(f"Created registry with {len(registry)} tools")

    return registry
