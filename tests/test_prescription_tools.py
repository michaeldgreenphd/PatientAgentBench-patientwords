# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for Prescription Tools with Healthcare Sandbox.

Tests the sandbox-aware prescription tools for medication management.
"""

from datetime import date

import pytest

from tests.conftest import create_test_patient_profile
from patient_agent_bench.sandbox import (
    HealthcareSandbox,
    Doctor,
    Medication,
    MedicationStatus,
)
from patient_agent_bench.tools.prescription_tools import (
    get_prescription_tools,
    _list_medications,
    _request_refill,
    _request_new_prescription,
)


@pytest.fixture
def sandbox_with_medications():
    """Create a sandbox with test medications."""
    sandbox = HealthcareSandbox()

    # Set current_date for refill operations
    sandbox.current_date = "2026-02-11"

    # Add test doctor (PCP)
    sandbox.doctors["doc_001"] = Doctor(
        id="doc_001",
        name="Dr. Sarah Johnson",
        specialty="Primary Care",
        credentials="MD",
        office_id="office_001",
    )
    sandbox.pcp_id = "doc_001"

    # Add patient profile
    sandbox.patient_profile = create_test_patient_profile(
        first_name="John",
        last_name="Doe",
        pharmacy_name="CVS Pharmacy",
        pharmacy_address="123 Main St",
        pcp_name="Dr. Sarah Johnson",
        pcp_id="doc_001",
    )

    # Add test medications to patient_profile
    sandbox.patient_profile.add_medication(Medication(
        id="med_001",
        name="Metformin",
        dosage="500mg",
        frequency="twice daily",
        status=MedicationStatus.ACTIVE,
        prescribed_date="2025-01-01",
        pharmacy="CVS Pharmacy",
        refills_remaining=3,
        last_filled="2025-12-01",
        reason="Type 2 Diabetes",
    ))
    sandbox.patient_profile.add_medication(Medication(
        id="med_002",
        name="Lisinopril",
        dosage="10mg",
        frequency="once daily",
        status=MedicationStatus.ACTIVE,
        prescribed_date="2025-06-01",
        pharmacy="CVS Pharmacy",
        refills_remaining=5,
        last_filled="2025-11-01",
        reason="Blood pressure",
    ))
    sandbox.patient_profile.add_medication(Medication(
        id="med_003",
        name="Amoxicillin",
        dosage="500mg",
        frequency="three times daily",
        status=MedicationStatus.PAST,
        prescribed_date="2024-01-01",
        pharmacy="Walgreens",
        refills_remaining=0,
        last_filled="2024-01-01",
        reason="Infection",
    ))
    sandbox.patient_profile.add_medication(Medication(
        id="med_004",
        name="Atorvastatin",
        dosage="20mg",
        frequency="once daily at bedtime",
        status=MedicationStatus.UNDER_RENEWAL,
        prescribed_date="2025-01-01",
        pharmacy="CVS Pharmacy",
        refills_remaining=0,
        last_filled="2025-12-15",
        reason="Cholesterol",
    ))

    sandbox._initialized = True
    return sandbox


@pytest.fixture
def sandbox_without_pharmacy():
    """Create a sandbox without pharmacy configured."""
    sandbox = HealthcareSandbox()

    # Add test doctor (PCP)
    sandbox.doctors["doc_001"] = Doctor(
        id="doc_001",
        name="Dr. Sarah Johnson",
        specialty="Primary Care",
        credentials="MD",
        office_id="office_001",
    )
    sandbox.pcp_id = "doc_001"

    # Add patient profile without pharmacy
    sandbox.patient_profile = create_test_patient_profile(
        first_name="John",
        last_name="Doe",
        pcp_name="Dr. Sarah Johnson",
        pcp_id="doc_001",
    )

    # Add medication without pharmacy to patient_profile
    sandbox.patient_profile.add_medication(Medication(
        id="med_001",
        name="Metformin",
        dosage="500mg",
        frequency="twice daily",
        status=MedicationStatus.ACTIVE,
        prescribed_date="2025-01-01",
        pharmacy="",  # No pharmacy on medication
        refills_remaining=3,
        last_filled="2025-12-01",
        reason="Type 2 Diabetes",
    ))

    sandbox._initialized = True
    return sandbox


class TestGetPrescriptionTools:
    """Tests for get_prescription_tools function."""

    def test_returns_three_tools(self, sandbox_with_medications):
        """Test that get_prescription_tools returns exactly 3 tools."""
        tools = get_prescription_tools(sandbox_with_medications)
        assert len(tools) == 3

    def test_tool_names(self, sandbox_with_medications):
        """Test that tools have correct names."""
        tools = get_prescription_tools(sandbox_with_medications)
        tool_names = [t.name for t in tools]

        assert "list_medications" in tool_names
        assert "request_refill" in tool_names
        assert "request_new_prescription" in tool_names

    def test_tools_have_descriptions(self, sandbox_with_medications):
        """Test that all tools have descriptions."""
        tools = get_prescription_tools(sandbox_with_medications)

        for tool in tools:
            assert tool.description
            assert len(tool.description) > 20


class TestListMedications:
    """Tests for list_medications functionality."""

    def test_list_all_medications(self, sandbox_with_medications):
        """Test listing all medications."""
        result = _list_medications(sandbox_with_medications)

        assert "Metformin" in result
        assert "Lisinopril" in result
        assert "Amoxicillin" in result
        assert "Atorvastatin" in result

    def test_medications_grouped_by_status(self, sandbox_with_medications):
        """Test medications are grouped by status."""
        result = _list_medications(sandbox_with_medications)

        assert "ACTIVE MEDICATIONS:" in result
        assert "PAST MEDICATIONS:" in result
        assert "PENDING RENEWAL:" in result

    def test_medication_details_displayed(self, sandbox_with_medications):
        """Test medication details are shown."""
        result = _list_medications(sandbox_with_medications)

        # Check for dosage and frequency
        assert "500mg" in result
        assert "twice daily" in result

        # Check for refills
        assert "Refills remaining:" in result

        # Check for pharmacy
        assert "CVS Pharmacy" in result

    def test_empty_medications(self):
        """Test listing when no medications exist."""
        sandbox = HealthcareSandbox()
        sandbox._initialized = True

        result = _list_medications(sandbox)

        assert "No medications on file" in result

    def test_list_medications_tool_invocation(self, sandbox_with_medications):
        """Test list_medications tool can be invoked."""
        tools = get_prescription_tools(sandbox_with_medications)
        list_tool = next(t for t in tools if t.name == "list_medications")

        result = list_tool.invoke({})

        assert "Metformin" in result
        assert "ACTIVE MEDICATIONS:" in result


class TestRequestRefill:
    """Tests for request_refill functionality."""

    def test_refill_active_medication(self, sandbox_with_medications):
        """Test refilling an active medication."""
        original_refills = sandbox_with_medications.patient_profile.get_medication("med_001").refills_remaining

        result = _request_refill(sandbox_with_medications, "Metformin")

        assert "✓" in result
        assert "Metformin" in result
        assert "500mg" in result
        assert sandbox_with_medications.patient_profile.get_medication("med_001").refills_remaining == original_refills - 1

    def test_refill_case_insensitive(self, sandbox_with_medications):
        """Test medication name matching is case-insensitive."""
        result = _request_refill(sandbox_with_medications, "METFORMIN")

        assert "✓" in result
        assert "Metformin" in result

    def test_refill_partial_name_match(self, sandbox_with_medications):
        """Test medication name partial matching."""
        result = _request_refill(sandbox_with_medications, "metfor")

        assert "✓" in result
        assert "Metformin" in result

    def test_refill_nonexistent_medication(self, sandbox_with_medications):
        """Test refilling a medication that doesn't exist."""
        result = _request_refill(sandbox_with_medications, "NonexistentMed")

        assert "Error" in result
        assert "not found" in result

    def test_refill_past_medication(self, sandbox_with_medications):
        """Test refilling a past medication fails."""
        result = _request_refill(sandbox_with_medications, "Amoxicillin")

        assert "Error" in result
        assert "past medication" in result

    def test_refill_no_refills_remaining(self, sandbox_with_medications):
        """Test refilling when no refills remaining."""
        # Set refills to 0
        sandbox_with_medications.patient_profile.get_medication("med_001").refills_remaining = 0

        result = _request_refill(sandbox_with_medications, "Metformin")

        assert "Error" in result
        assert "no refills remaining" in result

    def test_refill_urgent(self, sandbox_with_medications):
        """Test urgent refill request."""
        result = _request_refill(sandbox_with_medications, "Metformin", urgent=True)

        assert "URGENT" in result
        assert "4-8 hours" in result

    def test_refill_non_urgent(self, sandbox_with_medications):
        """Test non-urgent refill request."""
        result = _request_refill(sandbox_with_medications, "Metformin", urgent=False)

        assert "URGENT" not in result
        assert "24-48 hours" in result

    def test_refill_updates_last_filled(self, sandbox_with_medications):
        """Test that refill updates last_filled date."""
        _request_refill(sandbox_with_medications, "Metformin")

        med = sandbox_with_medications.patient_profile.get_medication("med_001")
        # Uses sandbox.current_date which is set to "2026-02-11" in fixture
        assert med.last_filled == "2026-02-11"

    def test_refill_tool_invocation(self, sandbox_with_medications):
        """Test request_refill tool can be invoked."""
        tools = get_prescription_tools(sandbox_with_medications)
        refill_tool = next(t for t in tools if t.name == "request_refill")

        result = refill_tool.invoke({"medication_name": "Lisinopril"})

        assert "✓" in result
        assert "Lisinopril" in result

    def test_refill_error_when_no_pharmacy_set(self, sandbox_without_pharmacy):
        """Test error when no pharmacy is configured."""
        result = _request_refill(sandbox_without_pharmacy, "Metformin")

        assert "Error" in result
        assert "No pharmacy is configured" in result
        assert "update_pharmacy" in result

    def test_refill_success_with_pharmacy_in_profile(self, sandbox_with_medications):
        """Test successful refill when pharmacy is in profile."""
        result = _request_refill(sandbox_with_medications, "Metformin")

        assert "✓" in result
        assert "CVS Pharmacy" in result

    def test_refill_output_contains_all_required_info(self, sandbox_with_medications):
        """Test that refill output contains confirmation, pickup timeline, and pharmacy."""
        result = _request_refill(sandbox_with_medications, "Metformin")

        assert "✓" in result  # Confirmation
        assert "pickup time" in result.lower()  # Pickup timeline
        assert "CVS Pharmacy" in result  # Pharmacy info
        assert "notification" in result.lower()  # Next steps

    def test_refill_description_mentions_pharmacy_requirement(self, sandbox_with_medications):
        """Test that tool description mentions pharmacy requirement."""
        tools = get_prescription_tools(sandbox_with_medications)
        refill_tool = next(t for t in tools if t.name == "request_refill")

        assert "pharmacy" in refill_tool.description.lower()
        assert "update_pharmacy" in refill_tool.description


class TestRequestNewPrescription:
    """Tests for request_new_prescription functionality."""

    def test_request_new_prescription(self, sandbox_with_medications):
        """Test requesting a new prescription."""
        result = _request_new_prescription(
            sandbox_with_medications,
            medication_name="Gabapentin",
            reason="Nerve pain",
        )

        assert "✓" in result
        assert "Gabapentin" in result
        assert "Nerve pain" in result

    def test_new_prescription_added_to_sandbox(self, sandbox_with_medications):
        """Test new prescription is added to sandbox."""
        initial_count = len(sandbox_with_medications.patient_profile.medications)

        _request_new_prescription(
            sandbox_with_medications,
            medication_name="Gabapentin",
            reason="Nerve pain",
        )

        assert len(sandbox_with_medications.patient_profile.medications) == initial_count + 1

    def test_new_prescription_status_under_renewal(self, sandbox_with_medications):
        """Test new prescription has under_renewal status."""
        _request_new_prescription(
            sandbox_with_medications,
            medication_name="Gabapentin",
            reason="Nerve pain",
        )

        # Find the new medication
        new_med = None
        for med in sandbox_with_medications.patient_profile.medications:
            if med.name == "Gabapentin":
                new_med = med
                break

        assert new_med is not None
        assert new_med.status == MedicationStatus.UNDER_RENEWAL

    def test_new_prescription_uses_profile_pharmacy(self, sandbox_with_medications):
        """Test new prescription uses profile pharmacy when not specified."""
        result = _request_new_prescription(
            sandbox_with_medications,
            medication_name="Gabapentin",
            reason="Nerve pain",
        )

        assert "CVS Pharmacy" in result

    def test_new_prescription_tool_invocation(self, sandbox_with_medications):
        """Test request_new_prescription tool can be invoked."""
        tools = get_prescription_tools(sandbox_with_medications)
        new_rx_tool = next(t for t in tools if t.name == "request_new_prescription")

        result = new_rx_tool.invoke({
            "medication_name": "Gabapentin",
            "reason": "Nerve pain",
        })

        assert "✓" in result
        assert "Gabapentin" in result

    def test_new_prescription_error_when_no_pharmacy_set(self, sandbox_without_pharmacy):
        """Test error when no pharmacy is configured."""
        result = _request_new_prescription(
            sandbox_without_pharmacy,
            medication_name="Gabapentin",
            reason="Nerve pain",
        )

        assert "Error" in result
        assert "No pharmacy is configured" in result
        assert "update_pharmacy" in result

    def test_new_prescription_success_with_pharmacy_in_profile(self, sandbox_with_medications):
        """Test successful request when pharmacy is in profile."""
        result = _request_new_prescription(
            sandbox_with_medications,
            medication_name="Gabapentin",
            reason="Nerve pain",
        )

        assert "✓" in result
        assert "CVS Pharmacy" in result

    def test_new_prescription_output_contains_all_required_info(self, sandbox_with_medications):
        """Test that output contains confirmation, review timeline, and next steps."""
        result = _request_new_prescription(
            sandbox_with_medications,
            medication_name="Gabapentin",
            reason="Nerve pain",
        )

        assert "✓" in result  # Confirmation
        assert "1-2 business days" in result  # Review timeline
        assert "What happens next:" in result  # Next steps
        assert "approved" in result.lower()

    def test_new_prescription_description_mentions_pharmacy_requirement(
        self, sandbox_with_medications
    ):
        """Test that tool description mentions pharmacy requirement."""
        tools = get_prescription_tools(sandbox_with_medications)
        new_rx_tool = next(t for t in tools if t.name == "request_new_prescription")

        assert "pharmacy" in new_rx_tool.description.lower()
        assert "update_pharmacy" in new_rx_tool.description


class TestPrescriptionToolsIntegration:
    """Integration tests for prescription tools workflow."""

    def test_list_then_refill_workflow(self, sandbox_with_medications):
        """Test listing medications then refilling one."""
        tools = get_prescription_tools(sandbox_with_medications)
        list_tool = next(t for t in tools if t.name == "list_medications")
        refill_tool = next(t for t in tools if t.name == "request_refill")

        # List medications
        list_result = list_tool.invoke({})
        assert "Metformin" in list_result

        # Refill one
        refill_result = refill_tool.invoke({"medication_name": "Metformin"})
        assert "✓" in refill_result

        # List again to see updated refills
        list_result_after = list_tool.invoke({})
        assert "Refills remaining: 2" in list_result_after

    def test_request_new_then_list_workflow(self, sandbox_with_medications):
        """Test requesting new prescription then listing."""
        tools = get_prescription_tools(sandbox_with_medications)
        list_tool = next(t for t in tools if t.name == "list_medications")
        new_rx_tool = next(t for t in tools if t.name == "request_new_prescription")

        # Request new prescription
        new_rx_result = new_rx_tool.invoke({
            "medication_name": "Gabapentin",
            "reason": "Nerve pain",
        })
        assert "✓" in new_rx_result

        # List medications - should include new one
        list_result = list_tool.invoke({})
        assert "Gabapentin" in list_result
        assert "PENDING RENEWAL:" in list_result

    def test_multiple_refills_decrement_correctly(self, sandbox_with_medications):
        """Test multiple refills decrement refills_remaining correctly."""
        tools = get_prescription_tools(sandbox_with_medications)
        refill_tool = next(t for t in tools if t.name == "request_refill")

        # Lisinopril has 5 refills
        for i in range(3):
            result = refill_tool.invoke({"medication_name": "Lisinopril"})
            assert "✓" in result

        # Should have 2 refills remaining
        med = sandbox_with_medications.patient_profile.get_medication("med_002")
        assert med.refills_remaining == 2


# =============================================================================
# Property-Based Tests for Request Refill
# =============================================================================

from hypothesis import given, settings, strategies as st


@st.composite
def sandbox_no_pharmacy_strategy(draw):
    """Generate a sandbox without any pharmacy configured."""
    sandbox = HealthcareSandbox()

    sandbox.doctors["doc_001"] = Doctor(
        id="doc_001",
        name="Dr. Test",
        specialty="Primary Care",
        credentials="MD",
        office_id="office_001",
    )
    sandbox.pcp_id = "doc_001"

    # Profile without pharmacy
    sandbox.patient_profile = create_test_patient_profile(
        first_name="Test",
        last_name="Patient",
        pcp_name="Dr. Test",
        pcp_id="doc_001",
    )

    # Medication without pharmacy added to patient_profile
    sandbox.patient_profile.add_medication(Medication(
        id="med_001",
        name="TestMed",
        dosage="100mg",
        frequency="daily",
        status=MedicationStatus.ACTIVE,
        prescribed_date="2025-01-01",
        pharmacy="",
        refills_remaining=3,
        last_filled="2025-12-01",
        reason="Test condition",
    ))

    sandbox._initialized = True
    return sandbox


class TestRequestRefillProperties:
    """Property-based tests for request_refill functionality."""

    @given(sandbox=sandbox_no_pharmacy_strategy())
    @settings(max_examples=100)
    def test_property_14_request_refill_error_when_no_pharmacy_set(self, sandbox):
        """
        Feature: sandbox-tool-improvements
        Property 14: Request refill error when no pharmacy set

        For any sandbox where the patient has no pharmacy set (neither in profile
        nor provided as parameter), calling request_refill SHALL return an error
        message explaining the patient needs to set a pharmacy first.

        **Validates: Requirements 7.2**
        """
        result = _request_refill(sandbox, "TestMed")

        assert "Error" in result
        assert "No pharmacy is configured" in result


# =============================================================================
# Property-Based Tests for Request New Prescription
# =============================================================================


class TestRequestNewPrescriptionProperties:
    """Property-based tests for request_new_prescription functionality."""

    @given(sandbox=sandbox_no_pharmacy_strategy())
    @settings(max_examples=100)
    def test_property_16_request_new_prescription_error_when_no_pharmacy_set(self, sandbox):
        """
        Feature: sandbox-tool-improvements
        Property 16: Request new prescription error when no pharmacy set

        For any sandbox where the patient has no pharmacy set (neither in profile
        nor provided as parameter), calling request_new_prescription SHALL return
        an error message explaining the patient needs to set a pharmacy first.

        **Validates: Requirements 8.2**
        """
        result = _request_new_prescription(
            sandbox,
            medication_name="TestMedication",
            reason="Test reason",
        )

        assert "Error" in result
        assert "No pharmacy is configured" in result
