# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for Telehealth Tools with Healthcare Sandbox.

Tests the sandbox-aware telehealth tools for messaging doctors and joining virtual call queues.
"""

import pytest

from tests.conftest import create_test_patient_profile
from patient_agent_bench.sandbox import (
    HealthcareSandbox,
    Doctor,
    OfficeLocation,
)
from patient_agent_bench.tools.telehealth_tools import (
    get_telehealth_tools,
    _message_pcp,
    _join_virtual_call_queue,
)


@pytest.fixture
def sandbox_with_doctors():
    """Create a sandbox with test doctor data."""
    sandbox = HealthcareSandbox()

    # Add test offices
    sandbox.offices["office_001"] = OfficeLocation(
        id="office_001",
        name="Downtown Primary Care",
        address="123 Main St",
        city="Boston",
        state="MA",
        zip_code="02101",
        phone="555-123-4567",
        hours="Mon-Fri 8AM-6PM",
    )
    sandbox.offices["office_002"] = OfficeLocation(
        id="office_002",
        name="Suburban Medical Center",
        address="456 Oak Ave",
        city="Cambridge",
        state="MA",
        zip_code="02139",
        phone="555-987-6543",
        hours="Mon-Sat 9AM-5PM",
    )

    # Add test doctors
    sandbox.doctors["doc_001"] = Doctor(
        id="doc_001",
        name="Dr. Sarah Johnson",
        specialty="Primary Care",
        credentials="MD",
        office_id="office_001",
    )
    sandbox.doctors["doc_002"] = Doctor(
        id="doc_002",
        name="Dr. Michael Chen",
        specialty="Cardiology",
        credentials="MD",
        office_id="office_001",
    )
    sandbox.doctors["doc_003"] = Doctor(
        id="doc_003",
        name="Dr. Emily Brown",
        specialty="Dermatology",
        credentials="DO",
        office_id="office_002",
    )

    # Set PCP
    sandbox.pcp_id = "doc_001"

    # Add patient profile
    sandbox.patient_profile = create_test_patient_profile(
        first_name="John",
        last_name="Doe",
        pcp_name="Dr. Sarah Johnson",
        pcp_id="doc_001",
    )

    sandbox._initialized = True
    return sandbox


@pytest.fixture
def sandbox_without_pcp():
    """Create a sandbox without a PCP assigned."""
    sandbox = HealthcareSandbox()

    sandbox.offices["office_001"] = OfficeLocation(
        id="office_001",
        name="Downtown Primary Care",
        address="123 Main St",
        city="Boston",
        state="MA",
        zip_code="02101",
        phone="555-123-4567",
        hours="Mon-Fri 8AM-6PM",
    )

    sandbox.doctors["doc_001"] = Doctor(
        id="doc_001",
        name="Dr. Sarah Johnson",
        specialty="Primary Care",
        credentials="MD",
        office_id="office_001",
    )

    # No PCP assigned
    sandbox.pcp_id = None

    sandbox.patient_profile = create_test_patient_profile(
        first_name="Jane",
        last_name="Smith",
    )

    sandbox._initialized = True
    return sandbox


class TestGetTelehealthTools:
    """Tests for get_telehealth_tools function."""

    def test_returns_two_tools(self, sandbox_with_doctors):
        """Test that get_telehealth_tools returns exactly 2 tools."""
        tools = get_telehealth_tools(sandbox_with_doctors)
        assert len(tools) == 2

    def test_tool_names(self, sandbox_with_doctors):
        """Test that tools have correct names."""
        tools = get_telehealth_tools(sandbox_with_doctors)
        tool_names = [t.name for t in tools]

        assert "message_pcp" in tool_names
        assert "join_virtual_call_queue" in tool_names

    def test_tools_have_descriptions(self, sandbox_with_doctors):
        """Test that all tools have descriptions."""
        tools = get_telehealth_tools(sandbox_with_doctors)

        for tool in tools:
            assert tool.description
            assert len(tool.description) > 20


class TestMessagePcp:
    """Tests for message_pcp functionality."""

    def test_message_pcp_success(self, sandbox_with_doctors):
        """Test sending a message to PCP successfully."""
        result = _message_pcp(
            sandbox_with_doctors,
            reason_for_consultation="Question about medication",
            message_body="I have a question about my prescription.",
        )

        assert "✓" in result
        assert "Message sent successfully to your PCP" in result
        assert "Dr. Sarah Johnson" in result
        assert "Question about medication" in result

    def test_message_pcp_error_when_no_pcp(self, sandbox_without_pcp):
        """Test error when no PCP is assigned."""
        result = _message_pcp(
            sandbox_without_pcp,
            reason_for_consultation="Test",
            message_body="Test message.",
        )

        assert "Error" in result
        assert "do not have a Primary Care Provider" in result

    def test_message_pcp_required_fields(self, sandbox_with_doctors):
        """Test that required fields are included in output."""
        result = _message_pcp(
            sandbox_with_doctors,
            reason_for_consultation="Follow-up question",
            message_body="I wanted to follow up on our last visit.",
        )

        assert "Reason:" in result
        assert "Message:" in result
        assert "Follow-up question" in result
        assert "I wanted to follow up" in result

    def test_message_pcp_optional_fields(self, sandbox_with_doctors):
        """Test that optional fields are accepted and displayed."""
        result = _message_pcp(
            sandbox_with_doctors,
            reason_for_consultation="Symptom concern",
            message_body="I have been experiencing headaches.",
            symptom_onset="3 days ago",
            symptom_severity="moderate",
            current_medications_relevant="Ibuprofen 400mg",
            additional_context="Headaches are worse in the morning",
        )

        assert "Symptom onset: 3 days ago" in result
        assert "Severity: moderate" in result
        assert "Relevant medications: Ibuprofen 400mg" in result
        assert "Additional context: Headaches are worse" in result

    def test_message_pcp_routine_urgency(self, sandbox_with_doctors):
        """Test routine urgency response time."""
        result = _message_pcp(
            sandbox_with_doctors,
            reason_for_consultation="General question",
            message_body="Non-urgent question.",
            urgency="routine",
        )

        assert "24-48 hours" in result

    def test_message_pcp_urgent_urgency(self, sandbox_with_doctors):
        """Test urgent urgency response time."""
        result = _message_pcp(
            sandbox_with_doctors,
            reason_for_consultation="Urgent concern",
            message_body="I need a quick response.",
            urgency="urgent",
        )

        assert "4 hours" in result

    def test_message_pcp_default_urgency_is_routine(self, sandbox_with_doctors):
        """Test that default urgency is routine."""
        result = _message_pcp(
            sandbox_with_doctors,
            reason_for_consultation="Question",
            message_body="Test message.",
        )

        assert "24-48 hours" in result
        assert "Routine" in result

    def test_message_pcp_output_includes_response_timeline(self, sandbox_with_doctors):
        """Test that output includes response timeline."""
        result = _message_pcp(
            sandbox_with_doctors,
            reason_for_consultation="Test",
            message_body="Test message.",
        )

        assert "Expected response time:" in result
        assert "notification when your PCP responds" in result

    def test_message_pcp_tool_invocation(self, sandbox_with_doctors):
        """Test message_pcp tool can be invoked."""
        tools = get_telehealth_tools(sandbox_with_doctors)
        message_tool = next(t for t in tools if t.name == "message_pcp")

        result = message_tool.invoke({
            "reason_for_consultation": "Test reason",
            "message_body": "Test message body",
            "urgency": "routine",
        })

        assert "✓" in result
        assert "Message sent" in result

    def test_message_pcp_tool_description_mentions_pcp_only(self, sandbox_with_doctors):
        """Test that tool description mentions PCP-only restriction."""
        tools = get_telehealth_tools(sandbox_with_doctors)
        message_tool = next(t for t in tools if t.name == "message_pcp")

        assert "PCP" in message_tool.description
        assert "restricted" in message_tool.description.lower()


class TestJoinVirtualCallQueue:
    """Tests for join_virtual_call_queue functionality."""

    def test_join_queue_by_doctor_name(self, sandbox_with_doctors):
        """Test joining queue for a specific doctor."""
        result = _join_virtual_call_queue(
            sandbox_with_doctors,
            doctor_name="Dr. Sarah Johnson",
            reason="Discuss test results",
        )

        assert "✓" in result
        assert "added to the virtual call queue" in result
        assert "Dr. Sarah Johnson" in result

    def test_join_queue_for_pcp(self, sandbox_with_doctors):
        """Test joining queue for PCP."""
        result = _join_virtual_call_queue(
            sandbox_with_doctors,
            doctor_name="PCP",
            reason="Follow-up consultation",
        )

        assert "✓" in result
        assert "Dr. Sarah Johnson" in result  # PCP name

    def test_join_queue_shows_wait_time(self, sandbox_with_doctors):
        """Test that queue confirmation shows estimated wait time."""
        result = _join_virtual_call_queue(
            sandbox_with_doctors,
            doctor_name="Dr. Sarah Johnson",
            reason="Quick question",
        )

        assert "wait time" in result.lower()
        assert "minutes" in result.lower()

    def test_join_queue_shows_reason(self, sandbox_with_doctors):
        """Test that queue confirmation shows the reason."""
        reason = "Discuss medication side effects"
        result = _join_virtual_call_queue(
            sandbox_with_doctors,
            doctor_name="Dr. Sarah Johnson",
            reason=reason,
        )

        assert reason in result

    def test_join_queue_shows_practice_name(self, sandbox_with_doctors):
        """Test that queue confirmation shows practice name."""
        result = _join_virtual_call_queue(
            sandbox_with_doctors,
            doctor_name="Dr. Sarah Johnson",
            reason="Consultation",
        )

        assert "Downtown Primary Care" in result

    def test_join_queue_nonexistent_doctor(self, sandbox_with_doctors):
        """Test joining queue for nonexistent doctor."""
        result = _join_virtual_call_queue(
            sandbox_with_doctors,
            doctor_name="Dr. Nonexistent",
            reason="Test",
        )

        assert "Error" in result
        assert "Could not find doctor" in result

    def test_join_queue_pcp_when_not_assigned(self, sandbox_without_pcp):
        """Test joining queue for PCP when no PCP is assigned."""
        result = _join_virtual_call_queue(
            sandbox_without_pcp,
            doctor_name="PCP",
            reason="Test",
        )

        assert "Error" in result
        assert "do not have a Primary Care Provider" in result

    def test_join_queue_shows_instructions(self, sandbox_with_doctors):
        """Test that queue confirmation shows instructions."""
        result = _join_virtual_call_queue(
            sandbox_with_doctors,
            doctor_name="Dr. Sarah Johnson",
            reason="Consultation",
        )

        assert "camera" in result.lower() or "microphone" in result.lower()

    def test_join_queue_shows_doctor_specialty(self, sandbox_with_doctors):
        """Test that queue confirmation shows doctor specialty."""
        result = _join_virtual_call_queue(
            sandbox_with_doctors,
            doctor_name="Dr. Michael Chen",
            reason="Heart consultation",
        )

        assert "Cardiology" in result

    def test_join_queue_tool_invocation(self, sandbox_with_doctors):
        """Test join_virtual_call_queue tool can be invoked."""
        tools = get_telehealth_tools(sandbox_with_doctors)
        queue_tool = next(t for t in tools if t.name == "join_virtual_call_queue")

        result = queue_tool.invoke({
            "doctor_name": "Dr. Sarah Johnson",
            "reason": "Test consultation",
        })

        assert "✓" in result
        assert "queue" in result.lower()


class TestTelehealthToolsIntegration:
    """Integration tests for telehealth tools."""

    def test_message_pcp_with_different_urgencies(self, sandbox_with_doctors):
        """Test messaging PCP with different urgency levels."""
        tools = get_telehealth_tools(sandbox_with_doctors)
        message_tool = next(t for t in tools if t.name == "message_pcp")

        # Routine message
        result1 = message_tool.invoke({
            "reason_for_consultation": "General question",
            "message_body": "Question for my PCP.",
        })
        assert "Dr. Sarah Johnson" in result1
        assert "24-48 hours" in result1

        # Urgent message
        result2 = message_tool.invoke({
            "reason_for_consultation": "Urgent concern",
            "message_body": "Urgent question for my PCP.",
            "urgency": "urgent",
        })
        assert "Dr. Sarah Johnson" in result2
        assert "4 hours" in result2

    def test_queue_different_doctors(self, sandbox_with_doctors):
        """Test joining queue for different doctors."""
        tools = get_telehealth_tools(sandbox_with_doctors)
        queue_tool = next(t for t in tools if t.name == "join_virtual_call_queue")

        # Queue for PCP
        result1 = queue_tool.invoke({
            "doctor_name": "PCP",
            "reason": "General consultation",
        })
        assert "Dr. Sarah Johnson" in result1

        # Queue for specialist
        result2 = queue_tool.invoke({
            "doctor_name": "Dr. Emily Brown",
            "reason": "Skin concern",
        })
        assert "Dr. Emily Brown" in result2
        assert "Dermatology" in result2


# =============================================================================
# Property-Based Tests for Message PCP
# =============================================================================

from hypothesis import given, settings, strategies as st


@st.composite
def sandbox_with_pcp_strategy(draw):
    """Generate a sandbox with a PCP assigned."""
    sandbox = HealthcareSandbox()

    sandbox.offices["office_001"] = OfficeLocation(
        id="office_001",
        name="Test Medical Center",
        address="123 Test St",
        city="Boston",
        state="MA",
        zip_code="02101",
        phone="555-1234",
        hours="Mon-Fri 9-5",
    )

    # Generate a PCP
    first_letter = draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=1, max_size=1))
    rest_name = draw(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=2, max_size=10))
    pcp_name = f"Dr. {first_letter}{rest_name}"
    sandbox.doctors["doc_pcp"] = Doctor(
        id="doc_pcp",
        name=pcp_name,
        specialty="Primary Care",
        credentials="MD",
        office_id="office_001",
    )
    sandbox.pcp_id = "doc_pcp"

    sandbox.patient_profile = create_test_patient_profile(
        first_name="Test",
        last_name="Patient",
        pcp_name=pcp_name,
        pcp_id="doc_pcp",
    )

    sandbox._initialized = True
    return sandbox


@st.composite
def sandbox_without_pcp_strategy(draw):
    """Generate a sandbox without a PCP assigned."""
    sandbox = HealthcareSandbox()

    sandbox.offices["office_001"] = OfficeLocation(
        id="office_001",
        name="Test Medical Center",
        address="123 Test St",
        city="Boston",
        state="MA",
        zip_code="02101",
        phone="555-1234",
        hours="Mon-Fri 9-5",
    )

    # Add a doctor but don't assign as PCP
    sandbox.doctors["doc_001"] = Doctor(
        id="doc_001",
        name="Dr. NotPCP",
        specialty="Cardiology",
        credentials="MD",
        office_id="office_001",
    )
    sandbox.pcp_id = None

    sandbox.patient_profile = create_test_patient_profile(
        first_name="Test",
        last_name="Patient",
    )

    sandbox._initialized = True
    return sandbox


class TestMessagePcpProperties:
    """Property-based tests for message_pcp functionality."""

    @given(
        sandbox=sandbox_with_pcp_strategy(),
        reason=st.text(min_size=1, max_size=100),
        message=st.text(min_size=1, max_size=500),
    )
    @settings(max_examples=100)
    def test_property_10_message_pcp_only_messages_assigned_pcp(self, sandbox, reason, message):
        """
        Feature: sandbox-tool-improvements
        Property 10: Message PCP only messages assigned PCP

        For any sandbox with a PCP assigned, calling message_pcp SHALL send
        the message to the assigned PCP only.

        **Validates: Requirements 6.2**
        """
        result = _message_pcp(sandbox, reason, message)

        # Should succeed
        assert "✓" in result
        assert "Message sent successfully to your PCP" in result

        # Should be sent to the assigned PCP
        pcp = sandbox.get_pcp()
        assert pcp.name in result

    @given(
        sandbox=sandbox_without_pcp_strategy(),
        reason=st.text(min_size=1, max_size=100),
        message=st.text(min_size=1, max_size=500),
    )
    @settings(max_examples=100)
    def test_property_11_message_pcp_error_when_no_pcp_assigned(self, sandbox, reason, message):
        """
        Feature: sandbox-tool-improvements
        Property 11: Message PCP error when no PCP assigned

        For any sandbox without a PCP assigned, calling message_pcp SHALL
        return an error message explaining the patient needs to assign a PCP first.

        **Validates: Requirements 6.3**
        """
        result = _message_pcp(sandbox, reason, message)

        # Should return error
        assert "Error" in result
        assert "do not have a Primary Care Provider" in result


# =============================================================================
# Tests for Join Virtual Call Queue Output (Requirement 9)
# =============================================================================


class TestJoinVirtualCallQueueOutput:
    """Tests for join_virtual_call_queue output completeness per Requirement 9."""

    def test_output_contains_confirmation(self, sandbox_with_doctors):
        """
        Test that output contains confirmation stating what was achieved.
        Validates: Requirements 9.1
        """
        result = _join_virtual_call_queue(
            sandbox_with_doctors,
            doctor_name="Dr. Sarah Johnson",
            reason="Test consultation",
        )

        assert "✓" in result
        assert "added to the virtual call queue" in result

    def test_output_contains_wait_time(self, sandbox_with_doctors):
        """
        Test that output contains estimated wait time.
        Validates: Requirements 9.2
        """
        result = _join_virtual_call_queue(
            sandbox_with_doctors,
            doctor_name="Dr. Sarah Johnson",
            reason="Test consultation",
        )

        assert "Estimated wait time:" in result
        assert "minutes" in result.lower()

    def test_output_contains_video_call_instructions(self, sandbox_with_doctors):
        """
        Test that output contains video call instructions.
        Validates: Requirements 9.3
        """
        result = _join_virtual_call_queue(
            sandbox_with_doctors,
            doctor_name="Dr. Sarah Johnson",
            reason="Test consultation",
        )

        # Check for camera and microphone instructions
        assert "camera" in result.lower()
        assert "microphone" in result.lower()
        # Check for stay on page instruction
        assert "stay on this page" in result.lower()

    def test_output_contains_all_required_elements(self, sandbox_with_doctors):
        """
        Test that output contains all required elements per Requirement 9.
        Validates: Requirements 9.1, 9.2, 9.3
        """
        result = _join_virtual_call_queue(
            sandbox_with_doctors,
            doctor_name="Dr. Sarah Johnson",
            reason="Test consultation",
        )

        # 9.1: Confirmation
        assert "✓" in result

        # 9.2: Wait time
        assert "wait time" in result.lower()

        # 9.3: Video call instructions
        assert "camera" in result.lower()
        assert "microphone" in result.lower()
