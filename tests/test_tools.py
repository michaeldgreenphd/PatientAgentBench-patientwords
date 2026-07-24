# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for healthcare tools.

Provides template-based testing for tool categories - when adding a new tool category,
add it to TOOL_CATEGORIES and it will automatically be tested.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.tools import BaseTool

from patient_agent_bench.tools.registry import ToolRegistry, create_tool_registry
from patient_agent_bench.tools.appointment_tools import get_appointment_tools
from patient_agent_bench.tools.prescription_tools import get_prescription_tools
from patient_agent_bench.tools.profile_tools import get_profile_tools
from patient_agent_bench.tools.telehealth_tools import get_telehealth_tools
from patient_agent_bench.sandbox import (
    HealthcareSandbox,
    Medication,
    MedicationStatus,
)
from tests.conftest import create_test_patient_profile


def _get_test_sandbox() -> HealthcareSandbox:
    """Create a sandbox with test data for tool testing."""
    sandbox = HealthcareSandbox()
    # Manually populate with test data (bypassing LLM generation)
    from patient_agent_bench.sandbox import (
        OfficeLocation, Doctor, AppointmentSlot
    )
    
    # Add test office
    sandbox.offices["office_001"] = OfficeLocation(
        id="office_001", name="Test Clinic", address="123 Main St",
        city="Boston", state="MA", zip_code="02101", phone="555-1234", hours="Mon-Fri 9-5"
    )
    
    # Add test doctor
    sandbox.doctors["doc_001"] = Doctor(
        id="doc_001", name="Dr. Smith", specialty="Primary Care",
        credentials="MD", office_id="office_001"
    )
    sandbox.pcp_id = "doc_001"
    
    # Add test slots
    sandbox.slots["slot_001"] = AppointmentSlot(
        id="slot_001", doctor_id="doc_001", office_id="office_001",
        date="2026-01-20", time="09:00", duration_minutes=30,
        available=True, appointment_type="in_person"
    )
    sandbox.slots["slot_002"] = AppointmentSlot(
        id="slot_002", doctor_id="doc_001", office_id="office_001",
        date="2026-01-20", time="14:00", duration_minutes=30,
        available=True, appointment_type="telehealth"
    )

    # Add test patient profile with medications
    sandbox.patient_profile = create_test_patient_profile(
        first_name="Test",
        last_name="Patient",
        pharmacy_name="CVS",
        pharmacy_address="456 Oak St",
        pcp_id="doc_001",
        pcp_name="Dr. Smith",
    )

    # Add test medications to patient_profile
    sandbox.patient_profile.add_medication(Medication(
        id="med_001", name="Metformin", dosage="500mg", frequency="twice daily",
        status=MedicationStatus.ACTIVE, prescribed_date="2025-01-01",
        pharmacy="CVS", refills_remaining=3,
        last_filled="2025-12-01", reason="Diabetes"
    ))
    sandbox.patient_profile.add_medication(Medication(
        id="med_002", name="Lisinopril", dosage="10mg", frequency="once daily",
        status=MedicationStatus.ACTIVE, prescribed_date="2025-06-01",
        pharmacy="CVS", refills_remaining=5,
        last_filled="2025-11-01", reason="Blood pressure"
    ))
    sandbox._initialized = True
    return sandbox


# =============================================================================
# TOOL CATEGORY REGISTRY - Add new tool categories here for automatic testing
# =============================================================================

# Create a shared test sandbox for appointment tools
_TEST_SANDBOX = None

def _get_shared_sandbox():
    global _TEST_SANDBOX
    if _TEST_SANDBOX is None:
        _TEST_SANDBOX = _get_test_sandbox()
    return _TEST_SANDBOX


TOOL_CATEGORIES = {
    "appointment": {
        "getter": lambda: get_appointment_tools(_get_shared_sandbox()),
        "expected_tools": [
            "list_doctors",
            "get_available_appointments",
            "schedule_appointment",
            "cancel_appointment",
            "list_appointments",
        ],
        "sample_invocation": {
            "tool_name": "schedule_appointment",
            "args": {
                "appointment_type": "in_person",
                "preferred_date": "2026-01-20",
                "preferred_time": "morning",
                "reason": "Check-up",
            },
            "expected_keywords": ["scheduled", "appointment"],
        },
    },
    "prescription": {
        "getter": lambda: get_prescription_tools(_get_shared_sandbox()),
        "expected_tools": ["request_refill", "list_medications", "request_new_prescription"],
        "sample_invocation": {
            "tool_name": "request_refill",
            "args": {"medication_name": "Metformin"},
            "expected_keywords": ["metformin"],
        },
    },
    "profile": {
        "getter": lambda: get_profile_tools(_get_shared_sandbox()),
        "expected_tools": ["update_pcp", "update_pharmacy", "get_profile"],
        "sample_invocation": {
            "tool_name": "update_pharmacy",
            "args": {"pharmacy_name": "Walgreens", "pharmacy_address": "123 Main St"},
            "expected_keywords": ["walgreens"],
        },
    },
    "telehealth": {
        "getter": lambda: get_telehealth_tools(_get_shared_sandbox()),
        "expected_tools": ["message_pcp", "join_virtual_call_queue"],
        "sample_invocation": {
            "tool_name": "message_pcp",
            "args": {
                "reason_for_consultation": "Question about medication",
                "message_body": "Test message",
            },
            "expected_keywords": ["message", "sent"],
        },
    },
}


class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_create_empty_registry(self):
        """Test creating an empty registry."""
        registry = ToolRegistry()
        assert len(registry.get_tools()) == 0

    def test_register_tool(self):
        """Test registering a single tool."""
        registry = ToolRegistry()
        sandbox = _get_test_sandbox()
        tools = get_appointment_tools(sandbox)
        registry.register(tools[0])
        assert len(registry.get_tools()) == 1

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        registry = ToolRegistry()
        sandbox = _get_test_sandbox()
        tools = get_appointment_tools(sandbox)
        registry.register_many(tools)
        assert len(registry.get_tools()) == len(tools)

    def test_get_tool_by_name(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        sandbox = _get_test_sandbox()
        tools = get_appointment_tools(sandbox)
        registry.register_many(tools)

        tool = registry.get("schedule_appointment")
        assert tool is not None
        assert tool.name == "schedule_appointment"

    def test_get_nonexistent_tool(self):
        """Test getting a tool that doesn't exist."""
        registry = ToolRegistry()
        tool = registry.get("nonexistent_tool")
        assert tool is None

    def test_create_tool_registry(self):
        """Test creating a registry without sandbox returns empty registry."""
        registry = create_tool_registry()
        tools = registry.get_tools()
        
        # Without sandbox, registry should be empty (all tools require sandbox)
        assert len(tools) == 0

    def test_create_tool_registry_with_sandbox(self):
        """Test creating a registry with sandbox includes all sandbox-aware tools."""
        sandbox = _get_test_sandbox()
        registry = create_tool_registry(sandbox)
        tools = registry.get_tools()
        
        tool_names = [t.name for t in tools]
        
        # Sandbox-aware tools should be present
        assert "schedule_appointment" in tool_names
        assert "list_doctors" in tool_names
        assert "get_available_appointments" in tool_names
        assert "request_refill" in tool_names
        assert "list_medications" in tool_names
        assert "get_profile" in tool_names
        assert "message_pcp" in tool_names
        assert "join_virtual_call_queue" in tool_names
        
        # Deprecated tools should NOT be present
        assert "escalate_to_clinician" not in tool_names
        assert "request_care_delivery" not in tool_names
        assert "message_doctor" not in tool_names  # Replaced by message_pcp


class TestAppointmentTools:
    """Tests for appointment tools - basic validation only.
    
    Comprehensive tests are in test_appointment_tools.py.
    """

    def test_get_appointment_tools(self):
        """Test getting appointment tools."""
        sandbox = _get_test_sandbox()
        tools = get_appointment_tools(sandbox)
        assert len(tools) == 5
        
        tool_names = [t.name for t in tools]
        assert "list_doctors" in tool_names
        assert "get_available_appointments" in tool_names
        assert "schedule_appointment" in tool_names
        assert "cancel_appointment" in tool_names
        assert "list_appointments" in tool_names


class TestPrescriptionTools:
    """Tests for prescription tools - basic validation only.
    
    Comprehensive tests are in test_prescription_tools.py.
    """

    def test_get_prescription_tools(self):
        """Test getting prescription tools."""
        sandbox = _get_test_sandbox()
        tools = get_prescription_tools(sandbox)
        assert len(tools) == 3  # list_medications, request_refill, request_new_prescription

        tool_names = [t.name for t in tools]
        assert "request_refill" in tool_names
        assert "list_medications" in tool_names
        assert "request_new_prescription" in tool_names


class TestProfileTools:
    """Tests for profile tools - basic validation only.

    Comprehensive tests are in test_profile_tools.py.
    """

    def test_get_profile_tools(self):
        """Test getting profile tools."""
        sandbox = _get_test_sandbox()
        tools = get_profile_tools(sandbox)
        assert len(tools) == 5

        tool_names = [t.name for t in tools]
        assert "update_pcp" in tool_names
        assert "update_pharmacy" in tool_names
        assert "get_profile" in tool_names



# =============================================================================
# Template Tests - Applied to ALL tool categories
# =============================================================================

class TestToolCategoryRegistry:
    """Tests to ensure all tool categories are properly configured."""

    def test_all_categories_have_unique_tool_names(self):
        """Verify all tools across categories have unique names."""
        all_tool_names = []
        for category_name, category_config in TOOL_CATEGORIES.items():
            tools = category_config["getter"]()
            for tool in tools:
                all_tool_names.append(tool.name)
        
        assert len(all_tool_names) == len(set(all_tool_names)), \
            f"Duplicate tool names found: {[n for n in all_tool_names if all_tool_names.count(n) > 1]}"

    def test_all_tools_are_base_tool_instances(self):
        """Verify all tools are BaseTool instances."""
        for category_name, category_config in TOOL_CATEGORIES.items():
            tools = category_config["getter"]()
            for tool in tools:
                assert isinstance(tool, BaseTool), \
                    f"Tool {tool.name} in {category_name} is not a BaseTool instance"


@pytest.mark.parametrize("category_name,category_config", TOOL_CATEGORIES.items())
class TestToolCategoryTemplate:
    """Template tests applied to each tool category."""

    def test_category_returns_tools(self, category_name, category_config):
        """Test category getter returns non-empty list of tools."""
        tools = category_config["getter"]()
        
        assert isinstance(tools, list)
        assert len(tools) > 0, f"{category_name} returned no tools"

    def test_category_has_expected_tools(self, category_name, category_config):
        """Test category contains expected tools."""
        tools = category_config["getter"]()
        tool_names = [t.name for t in tools]
        
        for expected_tool in category_config["expected_tools"]:
            assert expected_tool in tool_names, \
                f"Expected tool '{expected_tool}' not found in {category_name}"

    def test_all_tools_have_descriptions(self, category_name, category_config):
        """Test all tools have non-empty descriptions."""
        tools = category_config["getter"]()
        
        for tool in tools:
            assert tool.description, f"Tool {tool.name} in {category_name} has no description"
            assert len(tool.description) > 10, \
                f"Tool {tool.name} in {category_name} has too short description"

    def test_all_tools_have_valid_names(self, category_name, category_config):
        """Test all tools have valid snake_case names."""
        tools = category_config["getter"]()
        
        for tool in tools:
            assert tool.name, f"Tool in {category_name} has no name"
            assert "_" in tool.name or tool.name.islower(), \
                f"Tool {tool.name} should use snake_case naming"

    def test_sample_invocation(self, category_name, category_config):
        """Test sample tool invocation if provided."""
        sample = category_config.get("sample_invocation")
        if sample is None:
            pytest.skip(f"No sample invocation for {category_name}")
        
        tools = category_config["getter"]()
        tool = next((t for t in tools if t.name == sample["tool_name"]), None)
        
        assert tool is not None, f"Sample tool {sample['tool_name']} not found"
        
        result = tool.invoke(sample["args"])
        result_lower = result.lower()
        
        for keyword in sample["expected_keywords"]:
            assert keyword.lower() in result_lower, \
                f"Expected '{keyword}' in result from {sample['tool_name']}"



class TestToolRegistryAdvanced:
    """Advanced tests for ToolRegistry functionality."""

    def test_get_by_category(self):
        """Test getting tools by category."""
        registry = ToolRegistry()
        sandbox = _get_test_sandbox()
        tools = get_appointment_tools(sandbox)
        registry.register_many(tools, category="appointments")

        category_tools = registry.get_by_category("appointments")

        assert len(category_tools) == len(tools)
        assert all(t.name in [tool.name for tool in tools] for t in category_tools)

    def test_get_by_nonexistent_category(self):
        """Test getting tools from nonexistent category."""
        registry = ToolRegistry()

        category_tools = registry.get_by_category("nonexistent")

        assert category_tools == []

    def test_list_tools(self):
        """Test listing all tool names."""
        registry = ToolRegistry()
        sandbox = _get_test_sandbox()
        tools = get_appointment_tools(sandbox)
        registry.register_many(tools)

        tool_names = registry.list_tools()

        assert len(tool_names) == len(tools)
        assert "schedule_appointment" in tool_names

    def test_list_categories(self):
        """Test listing all categories."""
        registry = ToolRegistry()
        sandbox = _get_test_sandbox()
        registry.register_many(get_appointment_tools(sandbox), category="appointments")
        registry.register_many(get_prescription_tools(sandbox), category="prescriptions")

        categories = registry.list_categories()

        assert "appointments" in categories
        assert "prescriptions" in categories
        assert len(categories) == 2

    def test_len_operator(self):
        """Test __len__ operator."""
        registry = ToolRegistry()
        sandbox = _get_test_sandbox()
        tools = get_appointment_tools(sandbox)
        registry.register_many(tools)

        assert len(registry) == len(tools)

    def test_contains_operator(self):
        """Test __contains__ operator."""
        registry = ToolRegistry()
        sandbox = _get_test_sandbox()
        tools = get_appointment_tools(sandbox)
        registry.register_many(tools)

        assert "schedule_appointment" in registry
        assert "nonexistent_tool" not in registry

    def test_get_tools_with_specific_names(self):
        """Test getting specific tools by name list."""
        registry = ToolRegistry()
        sandbox = _get_test_sandbox()
        registry.register_many(get_appointment_tools(sandbox))

        specific_tools = registry.get_tools(["schedule_appointment", "cancel_appointment"])

        assert len(specific_tools) == 2
        tool_names = [t.name for t in specific_tools]
        assert "schedule_appointment" in tool_names
        assert "cancel_appointment" in tool_names

    def test_get_tools_with_nonexistent_names(self):
        """Test getting tools with some nonexistent names."""
        registry = ToolRegistry()
        sandbox = _get_test_sandbox()
        registry.register_many(get_appointment_tools(sandbox))

        specific_tools = registry.get_tools(["schedule_appointment", "nonexistent"])

        # Should only return existing tools
        assert len(specific_tools) == 1
        assert specific_tools[0].name == "schedule_appointment"

    def test_register_tool_without_name_attribute(self):
        """Test registering tool without name attribute raises error."""
        registry = ToolRegistry()
        invalid_tool = MagicMock(spec=[])  # No 'name' attribute

        with pytest.raises(ValueError, match="Tool must have a 'name' attribute"):
            registry.register(invalid_tool)

    def test_register_multiple_categories(self):
        """Test registering tools in multiple categories."""
        registry = ToolRegistry()
        sandbox = _get_test_sandbox()
        tool1 = get_appointment_tools(sandbox)[0]
        tool2 = get_prescription_tools(sandbox)[0]

        registry.register(tool1, category="cat1")
        registry.register(tool2, category="cat2")

        assert len(registry.list_categories()) == 2
        assert len(registry.get_by_category("cat1")) == 1
        assert len(registry.get_by_category("cat2")) == 1
