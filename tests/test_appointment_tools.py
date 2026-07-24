# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for Appointment Tools with Healthcare Sandbox.

Tests the sandbox-aware appointment tools for scheduling and managing appointments.
"""

import pytest

from patient_agent_bench.sandbox import (
    HealthcareSandbox,
    Doctor,
    OfficeLocation,
    AppointmentSlot,
)
from patient_agent_bench.patient import PatientProfile
from tests.conftest import create_test_patient_profile
from patient_agent_bench.tools.appointment_tools import (
    get_appointment_tools,
    _schedule_appointment,
    _cancel_appointment,
    _list_appointments,
    _list_doctors,
    _get_available_appointments,
)


@pytest.fixture
def sandbox_with_appointments():
    """Create a sandbox with test appointment data."""
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
    sandbox.pcp_id = "doc_001"

    # Add test appointment slots
    # Morning slots (in_person)
    sandbox.slots["slot_001"] = AppointmentSlot(
        id="slot_001",
        doctor_id="doc_001",
        office_id="office_001",
        date="2026-01-25",
        time="09:00",
        duration_minutes=30,
        available=True,
        appointment_type="in_person",
    )
    sandbox.slots["slot_002"] = AppointmentSlot(
        id="slot_002",
        doctor_id="doc_001",
        office_id="office_001",
        date="2026-01-25",
        time="10:00",
        duration_minutes=30,
        available=True,
        appointment_type="in_person",
    )
    # Afternoon slot (telehealth)
    sandbox.slots["slot_003"] = AppointmentSlot(
        id="slot_003",
        doctor_id="doc_001",
        office_id="office_001",
        date="2026-01-25",
        time="14:00",
        duration_minutes=30,
        available=True,
        appointment_type="telehealth",
    )
    # Unavailable slot
    sandbox.slots["slot_004"] = AppointmentSlot(
        id="slot_004",
        doctor_id="doc_001",
        office_id="office_001",
        date="2026-01-25",
        time="15:00",
        duration_minutes=30,
        available=False,
        appointment_type="in_person",
    )
    # Different doctor slot
    sandbox.slots["slot_005"] = AppointmentSlot(
        id="slot_005",
        doctor_id="doc_002",
        office_id="office_001",
        date="2026-01-26",
        time="11:00",
        duration_minutes=30,
        available=True,
        appointment_type="in_person",
    )
    # Evening slot
    sandbox.slots["slot_006"] = AppointmentSlot(
        id="slot_006",
        doctor_id="doc_001",
        office_id="office_001",
        date="2026-01-27",
        time="18:00",
        duration_minutes=30,
        available=True,
        appointment_type="in_person",
    )

    # Add patient profile
    sandbox.patient_profile = create_test_patient_profile(
        first_name="John",
        last_name="Doe",
        pcp_name="Dr. Sarah Johnson",
        pcp_id="doc_001",
    )

    sandbox._initialized = True
    return sandbox


class TestGetAppointmentTools:
    """Tests for get_appointment_tools function."""

    def test_returns_five_tools(self, sandbox_with_appointments):
        """Test that get_appointment_tools returns exactly 5 tools."""
        tools = get_appointment_tools(sandbox_with_appointments)
        assert len(tools) == 5

    def test_tool_names(self, sandbox_with_appointments):
        """Test that tools have correct names."""
        tools = get_appointment_tools(sandbox_with_appointments)
        tool_names = [t.name for t in tools]

        assert "list_doctors" in tool_names
        assert "get_available_appointments" in tool_names
        assert "schedule_appointment" in tool_names
        assert "cancel_appointment" in tool_names
        assert "list_appointments" in tool_names

    def test_tools_have_descriptions(self, sandbox_with_appointments):
        """Test that all tools have descriptions."""
        tools = get_appointment_tools(sandbox_with_appointments)

        for tool in tools:
            assert tool.description
            assert len(tool.description) > 20


class TestListDoctors:
    """Tests for list_doctors functionality."""

    def test_list_doctors_empty_sandbox(self):
        """Test listing doctors when sandbox has no doctors."""
        sandbox = HealthcareSandbox()
        sandbox._initialized = True

        result = _list_doctors(sandbox)

        assert "No doctors are currently available" in result

    def test_list_doctors_single_doctor(self, sandbox_with_appointments):
        """Test listing a single doctor."""
        # Create sandbox with just one doctor
        sandbox = HealthcareSandbox()
        sandbox.offices["office_001"] = OfficeLocation(
            id="office_001",
            name="Downtown Clinic",
            address="123 Main St",
            city="Boston",
            state="MA",
            zip_code="02101",
            phone="555-1234",
            hours="Mon-Fri 9-5",
        )
        sandbox.doctors["doc_001"] = Doctor(
            id="doc_001",
            name="Dr. Jane Smith",
            specialty="Primary Care",
            credentials="MD",
            office_id="office_001",
        )
        sandbox._initialized = True

        result = _list_doctors(sandbox)

        assert "Dr. Jane Smith" in result
        assert "Primary Care" in result
        assert "MD" in result
        assert "Downtown Clinic" in result

    def test_list_doctors_multiple_doctors(self, sandbox_with_appointments):
        """Test listing multiple doctors."""
        result = _list_doctors(sandbox_with_appointments)

        assert "Dr. Sarah Johnson" in result
        assert "Dr. Michael Chen" in result
        assert "Dr. Emily Brown" in result
        assert "Primary Care" in result
        assert "Cardiology" in result
        assert "Dermatology" in result

    def test_list_doctors_specialty_filter_matching(self, sandbox_with_appointments):
        """Test filtering doctors by specialty with matching doctors."""
        result = _list_doctors(sandbox_with_appointments, specialty="Cardiology")

        assert "Dr. Michael Chen" in result
        assert "Cardiology" in result
        # Should not include other specialties
        assert "Dr. Sarah Johnson" not in result
        assert "Dr. Emily Brown" not in result

    def test_list_doctors_specialty_filter_no_match(self, sandbox_with_appointments):
        """Test filtering doctors by specialty with no matching doctors."""
        result = _list_doctors(sandbox_with_appointments, specialty="Psychiatry")

        assert "No doctors found with specialty" in result

    def test_list_doctors_specialty_filter_case_insensitive(self, sandbox_with_appointments):
        """Test that specialty filter is case-insensitive."""
        result_lower = _list_doctors(sandbox_with_appointments, specialty="cardiology")
        result_upper = _list_doctors(sandbox_with_appointments, specialty="CARDIOLOGY")
        result_mixed = _list_doctors(sandbox_with_appointments, specialty="CarDioLoGy")

        assert "Dr. Michael Chen" in result_lower
        assert "Dr. Michael Chen" in result_upper
        assert "Dr. Michael Chen" in result_mixed

    def test_list_doctors_invalid_specialty(self, sandbox_with_appointments):
        """Test filtering with invalid specialty value."""
        result = _list_doctors(sandbox_with_appointments, specialty="InvalidSpecialty")

        assert "Invalid specialty" in result
        assert "Valid values:" in result

    def test_list_doctors_output_contains_required_fields(self, sandbox_with_appointments):
        """Test that output contains all required fields."""
        result = _list_doctors(sandbox_with_appointments)

        # Check for required fields for each doctor
        assert "Specialty:" in result
        assert "Location:" in result
        assert "Address:" in result

    def test_list_doctors_tool_invocation(self, sandbox_with_appointments):
        """Test list_doctors tool can be invoked."""
        tools = get_appointment_tools(sandbox_with_appointments)
        list_doctors_tool = next(t for t in tools if t.name == "list_doctors")

        result = list_doctors_tool.invoke({})

        assert "Available Doctors" in result

    def test_list_doctors_tool_with_specialty_filter(self, sandbox_with_appointments):
        """Test list_doctors tool with specialty filter."""
        tools = get_appointment_tools(sandbox_with_appointments)
        list_doctors_tool = next(t for t in tools if t.name == "list_doctors")

        result = list_doctors_tool.invoke({"specialty": "Primary Care"})

        assert "Dr. Sarah Johnson" in result
        assert "Dr. Michael Chen" not in result


class TestGetAvailableAppointments:
    """Tests for get_available_appointments functionality."""

    def test_get_available_appointments_date_range(self, sandbox_with_appointments):
        """Test filtering appointments by date range."""
        result = _get_available_appointments(
            sandbox_with_appointments,
            start_date="2026-01-25",
            end_date="2026-01-25",
        )

        assert "Available Appointments" in result
        assert "2026-01-25" in result
        # Should not include slots from other dates
        assert "2026-01-26" not in result
        assert "2026-01-27" not in result

    def test_get_available_appointments_wider_date_range(self, sandbox_with_appointments):
        """Test filtering with wider date range."""
        result = _get_available_appointments(
            sandbox_with_appointments,
            start_date="2026-01-25",
            end_date="2026-01-27",
        )

        assert "2026-01-25" in result
        assert "2026-01-26" in result
        assert "2026-01-27" in result

    def test_get_available_appointments_specialty_filter(self, sandbox_with_appointments):
        """Test filtering by specialty."""
        result = _get_available_appointments(
            sandbox_with_appointments,
            start_date="2026-01-25",
            end_date="2026-01-27",
            specialty="Cardiology",
        )

        assert "Dr. Michael Chen" in result
        assert "Cardiology" in result
        # Should not include other doctors
        assert "Dr. Sarah Johnson" not in result

    def test_get_available_appointments_combined_filters(self, sandbox_with_appointments):
        """Test combined date range and specialty filters."""
        result = _get_available_appointments(
            sandbox_with_appointments,
            start_date="2026-01-26",
            end_date="2026-01-26",
            specialty="Cardiology",
        )

        assert "Dr. Michael Chen" in result
        assert "2026-01-26" in result

    def test_get_available_appointments_output_contains_required_fields(
        self, sandbox_with_appointments
    ):
        """Test that output contains all required fields."""
        result = _get_available_appointments(
            sandbox_with_appointments,
            start_date="2026-01-25",
            end_date="2026-01-25",
        )

        assert "Slot ID:" in result
        assert "Date:" in result
        assert "Time:" in result
        assert "Type:" in result
        assert "Provider:" in result
        assert "Specialty:" in result
        assert "Location:" in result

    def test_get_available_appointments_empty_results(self, sandbox_with_appointments):
        """Test message when no slots available."""
        result = _get_available_appointments(
            sandbox_with_appointments,
            start_date="2030-01-01",
            end_date="2030-01-31",
        )

        assert "No available appointments found" in result

    def test_get_available_appointments_invalid_specialty(self, sandbox_with_appointments):
        """Test filtering with invalid specialty value."""
        result = _get_available_appointments(
            sandbox_with_appointments,
            start_date="2026-01-25",
            end_date="2026-01-25",
            specialty="InvalidSpecialty",
        )

        assert "Invalid specialty" in result
        assert "Valid values:" in result

    def test_get_available_appointments_tool_invocation(self, sandbox_with_appointments):
        """Test get_available_appointments tool can be invoked."""
        tools = get_appointment_tools(sandbox_with_appointments)
        tool = next(t for t in tools if t.name == "get_available_appointments")

        result = tool.invoke({
            "start_date": "2026-01-25",
            "end_date": "2026-01-27",
        })

        assert "Available Appointments" in result

    def test_get_available_appointments_tool_with_specialty(self, sandbox_with_appointments):
        """Test get_available_appointments tool with specialty filter."""
        tools = get_appointment_tools(sandbox_with_appointments)
        tool = next(t for t in tools if t.name == "get_available_appointments")

        result = tool.invoke({
            "start_date": "2026-01-25",
            "end_date": "2026-01-27",
            "specialty": "Primary Care",
        })

        assert "Dr. Sarah Johnson" in result


class TestScheduleAppointment:
    """Tests for schedule_appointment functionality."""

    def test_schedule_in_person_appointment(self, sandbox_with_appointments):
        """Test scheduling an in-person appointment."""
        result = _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
        )

        assert "✓" in result
        assert "scheduled" in result.lower()
        assert "Appointment ID:" in result

    def test_schedule_telehealth_appointment(self, sandbox_with_appointments):
        """Test scheduling a telehealth appointment."""
        result = _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="telehealth",
        )

        assert "✓" in result
        assert "Telehealth" in result

    def test_schedule_with_preferred_date(self, sandbox_with_appointments):
        """Test scheduling with a preferred date."""
        result = _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
            preferred_date="2026-01-25",
        )

        assert "✓" in result
        assert "2026-01-25" in result

    def test_schedule_with_morning_preference(self, sandbox_with_appointments):
        """Test scheduling with morning time preference."""
        result = _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
            preferred_time="morning",
        )

        assert "✓" in result
        # Should get a morning slot (09:00 or 10:00)
        assert "09:00" in result or "10:00" in result

    def test_schedule_with_afternoon_preference(self, sandbox_with_appointments):
        """Test scheduling with afternoon time preference."""
        result = _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="telehealth",
            preferred_time="afternoon",
        )

        assert "✓" in result
        assert "14:00" in result

    def test_schedule_with_evening_preference(self, sandbox_with_appointments):
        """Test scheduling with evening time preference."""
        result = _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
            preferred_time="evening",
        )

        assert "✓" in result
        assert "18:00" in result

    def test_schedule_with_provider_name(self, sandbox_with_appointments):
        """Test scheduling with specific provider."""
        result = _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
            provider_name="Dr. Michael Chen",
        )

        assert "✓" in result
        assert "Dr. Michael Chen" in result

    def test_schedule_with_pcp(self, sandbox_with_appointments):
        """Test scheduling with PCP."""
        result = _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
            provider_name="PCP",
        )

        assert "✓" in result
        assert "Dr. Sarah Johnson" in result

    def test_schedule_with_reason(self, sandbox_with_appointments):
        """Test scheduling with a reason."""
        result = _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
            reason="Annual checkup",
        )

        assert "✓" in result
        assert "Annual checkup" in result

    def test_schedule_nonexistent_provider(self, sandbox_with_appointments):
        """Test scheduling with nonexistent provider."""
        result = _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
            provider_name="Dr. Nonexistent",
        )

        assert "Error" in result
        assert "Could not find provider" in result

    def test_schedule_marks_slot_unavailable(self, sandbox_with_appointments):
        """Test that scheduling marks the slot as unavailable."""
        # Get initial available count
        initial_available = len(sandbox_with_appointments.get_available_slots())

        _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
        )

        # Should have one less available slot
        assert len(sandbox_with_appointments.get_available_slots()) == initial_available - 1

    def test_schedule_creates_booked_appointment(self, sandbox_with_appointments):
        """Test that scheduling creates a booked appointment."""
        initial_booked = len(sandbox_with_appointments.get_booked_appointments())

        _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
        )

        assert len(sandbox_with_appointments.get_booked_appointments()) == initial_booked + 1

    def test_schedule_no_available_slots(self):
        """Test scheduling when no slots are available."""
        sandbox = HealthcareSandbox()
        sandbox.doctors["doc_001"] = Doctor(
            "doc_001", "Dr. Smith", "Primary Care", "MD", "office_001"
        )
        sandbox._initialized = True

        result = _schedule_appointment(sandbox, appointment_type="in_person")

        assert "No available appointment slots" in result

    def test_schedule_appointment_type_normalization(self, sandbox_with_appointments):
        """Test that various appointment type inputs are normalized."""
        # Test "virtual" maps to telehealth
        result = _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="virtual",
        )
        assert "Telehealth" in result

    def test_schedule_tool_invocation(self, sandbox_with_appointments):
        """Test schedule_appointment tool can be invoked."""
        tools = get_appointment_tools(sandbox_with_appointments)
        schedule_tool = next(t for t in tools if t.name == "schedule_appointment")

        result = schedule_tool.invoke({
            "appointment_type": "in_person",
            "preferred_date": "2026-01-25",
            "reason": "Checkup",
        })

        assert "✓" in result
        assert "scheduled" in result.lower()

    def test_schedule_description_contains_workflow_guidance(self, sandbox_with_appointments):
        """Test that schedule_appointment description contains workflow guidance."""
        tools = get_appointment_tools(sandbox_with_appointments)
        schedule_tool = next(t for t in tools if t.name == "schedule_appointment")

        assert "get_available_appointments" in schedule_tool.description
        assert "IMPORTANT" in schedule_tool.description

    def test_schedule_output_contains_patient_guidance_in_person(self, sandbox_with_appointments):
        """Test that in-person appointment output contains patient guidance."""
        result = _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
        )

        assert "What to expect:" in result
        assert "arrive 15 minutes early" in result
        assert "insurance card" in result
        assert "medications" in result

    def test_schedule_output_contains_patient_guidance_telehealth(self, sandbox_with_appointments):
        """Test that telehealth appointment output contains patient guidance."""
        result = _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="telehealth",
        )

        assert "What to expect:" in result
        assert "video call link" in result
        assert "internet connection" in result


class TestCancelAppointment:
    """Tests for cancel_appointment functionality."""

    def test_cancel_existing_appointment(self, sandbox_with_appointments):
        """Test canceling an existing appointment."""
        # First schedule an appointment
        _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
        )
        appointments = sandbox_with_appointments.get_booked_appointments()
        appt_id = appointments[0].id

        # Then cancel it
        result = _cancel_appointment(sandbox_with_appointments, appt_id)

        assert "✓" in result
        assert "cancelled" in result.lower()
        assert appt_id in result

    def test_cancel_with_reason(self, sandbox_with_appointments):
        """Test canceling with a reason."""
        # First schedule
        _schedule_appointment(sandbox_with_appointments, appointment_type="in_person")
        appointments = sandbox_with_appointments.get_booked_appointments()
        appt_id = appointments[0].id

        # Cancel with reason
        result = _cancel_appointment(
            sandbox_with_appointments,
            appt_id,
            reason_for_cancellation="Schedule conflict",
        )

        assert "Schedule conflict" in result

    def test_cancel_nonexistent_appointment(self, sandbox_with_appointments):
        """Test canceling a nonexistent appointment."""
        result = _cancel_appointment(sandbox_with_appointments, "nonexistent_id")

        assert "Error" in result
        assert "not found" in result

    def test_cancel_releases_slot(self, sandbox_with_appointments):
        """Test that canceling releases the slot back to available."""
        # Schedule an appointment
        _schedule_appointment(sandbox_with_appointments, appointment_type="in_person")
        available_after_booking = len(sandbox_with_appointments.get_available_slots())

        # Cancel it
        appointments = sandbox_with_appointments.get_booked_appointments()
        appt_id = appointments[0].id
        _cancel_appointment(sandbox_with_appointments, appt_id)

        # Should have one more available slot
        assert len(sandbox_with_appointments.get_available_slots()) == available_after_booking + 1

    def test_cancel_removes_from_booked(self, sandbox_with_appointments):
        """Test that canceling removes from booked appointments."""
        # Schedule
        _schedule_appointment(sandbox_with_appointments, appointment_type="in_person")
        assert len(sandbox_with_appointments.get_booked_appointments()) == 1

        # Cancel
        appointments = sandbox_with_appointments.get_booked_appointments()
        appt_id = appointments[0].id
        _cancel_appointment(sandbox_with_appointments, appt_id)

        assert len(sandbox_with_appointments.get_booked_appointments()) == 0

    def test_cancel_tool_invocation(self, sandbox_with_appointments):
        """Test cancel_appointment tool can be invoked."""
        tools = get_appointment_tools(sandbox_with_appointments)
        cancel_tool = next(t for t in tools if t.name == "cancel_appointment")

        # Try to cancel nonexistent - should return error message
        result = cancel_tool.invoke({"appointment_id": "nonexistent"})

        assert "Error" in result or "not found" in result

    def test_cancel_description_contains_workflow_guidance(self, sandbox_with_appointments):
        """Test that cancel_appointment description contains workflow guidance."""
        tools = get_appointment_tools(sandbox_with_appointments)
        cancel_tool = next(t for t in tools if t.name == "cancel_appointment")

        assert "list_appointments" in cancel_tool.description
        assert "IMPORTANT" in cancel_tool.description

    def test_cancel_output_contains_rebooking_guidance(self, sandbox_with_appointments):
        """Test that cancel output contains rebooking guidance."""
        # First schedule
        _schedule_appointment(sandbox_with_appointments, appointment_type="in_person")
        appointments = sandbox_with_appointments.get_booked_appointments()
        appt_id = appointments[0].id

        # Cancel
        result = _cancel_appointment(sandbox_with_appointments, appt_id)

        assert "Need to rebook?" in result
        assert "get_available_appointments" in result
        assert "schedule_appointment" in result


class TestListAppointments:
    """Tests for list_appointments functionality."""

    def test_list_no_appointments(self, sandbox_with_appointments):
        """Test listing when no appointments exist."""
        result = _list_appointments(sandbox_with_appointments)

        assert "no scheduled appointments" in result.lower()

    def test_list_with_appointments(self, sandbox_with_appointments):
        """Test listing existing appointments."""
        # Schedule some appointments
        _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
            reason="Checkup",
        )
        _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="telehealth",
            reason="Follow-up",
        )

        result = _list_appointments(sandbox_with_appointments)

        assert "Scheduled Appointments" in result
        assert "Checkup" in result
        assert "Follow-up" in result

    def test_list_shows_appointment_details(self, sandbox_with_appointments):
        """Test that listing shows appointment details."""
        _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
            provider_name="PCP",
        )

        result = _list_appointments(sandbox_with_appointments)

        assert "Dr. Sarah Johnson" in result
        assert "Date:" in result
        assert "Time:" in result
        assert "Type:" in result

    def test_list_shows_location(self, sandbox_with_appointments):
        """Test that listing shows location for in-person appointments."""
        _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
        )

        result = _list_appointments(sandbox_with_appointments)

        assert "Location:" in result or "Downtown Primary Care" in result

    def test_list_appointments_sorted(self, sandbox_with_appointments):
        """Test that appointments are sorted by date and time."""
        # Schedule in reverse order
        _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
            preferred_date="2026-01-27",
        )
        _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
            preferred_date="2026-01-25",
        )

        result = _list_appointments(sandbox_with_appointments)

        # Earlier date should appear first
        pos_25 = result.find("2026-01-25")
        pos_27 = result.find("2026-01-27")
        assert pos_25 < pos_27

    def test_list_appointments_output_contains_appointment_id(self, sandbox_with_appointments):
        """Test that listing includes appointment_id for each appointment."""
        _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
        )

        result = _list_appointments(sandbox_with_appointments)
        appointments = sandbox_with_appointments.get_booked_appointments()

        # Each appointment ID should be in the result
        for appt in appointments:
            assert appt.id in result

    def test_list_appointments_output_contains_all_required_fields(self, sandbox_with_appointments):
        """Test that listing includes all required fields: appointment_id, date, time, provider, reason."""
        _schedule_appointment(
            sandbox_with_appointments,
            appointment_type="in_person",
            reason="Annual checkup",
        )

        result = _list_appointments(sandbox_with_appointments)

        # Check all required fields are present
        assert "Appointment ID:" in result
        assert "Date:" in result
        assert "Time:" in result
        assert "Provider Name:" in result
        assert "Provider Credentials:" in result
        assert "Provider Specialty:" in result
        assert "Annual checkup" in result  # reason

    def test_list_tool_invocation(self, sandbox_with_appointments):
        """Test list_appointments tool can be invoked."""
        tools = get_appointment_tools(sandbox_with_appointments)
        list_tool = next(t for t in tools if t.name == "list_appointments")

        result = list_tool.invoke({})

        assert "appointment" in result.lower()


class TestAppointmentToolsIntegration:
    """Integration tests for appointment tools workflow."""

    def test_schedule_list_cancel_workflow(self, sandbox_with_appointments):
        """Test full workflow: schedule, list, cancel."""
        tools = get_appointment_tools(sandbox_with_appointments)
        schedule_tool = next(t for t in tools if t.name == "schedule_appointment")
        list_tool = next(t for t in tools if t.name == "list_appointments")
        cancel_tool = next(t for t in tools if t.name == "cancel_appointment")

        # Schedule
        schedule_result = schedule_tool.invoke({
            "appointment_type": "in_person",
            "reason": "Annual physical",
        })
        assert "✓" in schedule_result

        # Extract appointment ID from result
        for line in schedule_result.split("\n"):
            if "Appointment ID:" in line:
                appt_id = line.split(":")[1].strip()
                break

        # List
        list_result = list_tool.invoke({})
        assert appt_id in list_result
        assert "Annual physical" in list_result

        # Cancel
        cancel_result = cancel_tool.invoke({"appointment_id": appt_id})
        assert "✓" in cancel_result
        assert "cancelled" in cancel_result.lower()

        # List again - should be empty
        list_result_after = list_tool.invoke({})
        assert "no scheduled appointments" in list_result_after.lower()

    def test_multiple_appointments_different_providers(self, sandbox_with_appointments):
        """Test scheduling with different providers."""
        tools = get_appointment_tools(sandbox_with_appointments)
        schedule_tool = next(t for t in tools if t.name == "schedule_appointment")
        list_tool = next(t for t in tools if t.name == "list_appointments")

        # Schedule with PCP
        schedule_tool.invoke({
            "appointment_type": "in_person",
            "provider_name": "PCP",
        })

        # Schedule with specialist
        schedule_tool.invoke({
            "appointment_type": "in_person",
            "provider_name": "Dr. Michael Chen",
        })

        # List should show both
        list_result = list_tool.invoke({})
        assert "Dr. Sarah Johnson" in list_result
        assert "Dr. Michael Chen" in list_result

    def test_slot_availability_after_booking(self, sandbox_with_appointments):
        """Test that slots become unavailable after booking."""
        # Get initial slot count
        initial_slots = len(sandbox_with_appointments.get_available_slots(
            appointment_type="in_person"
        ))

        # Book multiple appointments
        for _ in range(2):
            _schedule_appointment(
                sandbox_with_appointments,
                appointment_type="in_person",
            )

        # Should have 2 fewer slots
        current_slots = len(sandbox_with_appointments.get_available_slots(
            appointment_type="in_person"
        ))
        assert current_slots == initial_slots - 2


# =============================================================================
# Property-Based Tests for List Doctors
# =============================================================================

from hypothesis import given, settings, strategies as st


# Hypothesis Strategies for sandbox generation
@st.composite
def doctor_strategy(draw):
    """Generate a random doctor."""
    specialties = [
        "Primary Care", "Cardiology", "Dermatology", "Endocrinology",
        "Orthopedics", "Psychiatry", "OB/GYN"
    ]
    credentials = ["MD", "DO", "NP", "PA-C", "FNP"]

    return Doctor(
        id=draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=20)),
        name=f"Dr. {draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=1, max_size=1))}"
             f"{draw(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=2, max_size=15))}",
        specialty=draw(st.sampled_from(specialties)),
        credentials=draw(st.sampled_from(credentials)),
        office_id="office_001",
    )


@st.composite
def sandbox_with_doctors_strategy(draw, min_doctors=0, max_doctors=10):
    """Generate a sandbox with random doctors."""
    sandbox = HealthcareSandbox()

    # Add a default office
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

    # Generate unique doctors
    num_doctors = draw(st.integers(min_value=min_doctors, max_value=max_doctors))
    for i in range(num_doctors):
        doc = draw(doctor_strategy())
        # Ensure unique ID
        doc_id = f"doc_{i:03d}"
        doc = Doctor(
            id=doc_id,
            name=doc.name,
            specialty=doc.specialty,
            credentials=doc.credentials,
            office_id=doc.office_id,
        )
        sandbox.doctors[doc_id] = doc

    sandbox._initialized = True
    return sandbox


class TestListDoctorsProperties:
    """Property-based tests for list_doctors functionality."""

    @given(sandbox=sandbox_with_doctors_strategy(min_doctors=0, max_doctors=10))
    @settings(max_examples=100)
    def test_property_1_list_doctors_returns_all_doctors(self, sandbox):
        """
        Feature: sandbox-tool-improvements
        Property 1: List doctors returns all doctors

        For any sandbox containing doctors, calling list_doctors without a filter
        SHALL return all doctors in the sandbox.

        **Validates: Requirements 1.1**
        """
        result = _list_doctors(sandbox)
        doctors = sandbox.get_doctors()

        if not doctors:
            assert "No doctors are currently available" in result
        else:
            # Every doctor name should appear in the result
            for doctor in doctors:
                assert doctor.name in result, f"Doctor {doctor.name} not found in result"

    @given(sandbox=sandbox_with_doctors_strategy(min_doctors=1, max_doctors=10))
    @settings(max_examples=100)
    def test_property_2_list_doctors_output_contains_required_fields(self, sandbox):
        """
        Feature: sandbox-tool-improvements
        Property 2: List doctors output contains required fields

        For any doctor returned by list_doctors, the output SHALL contain
        the doctor's name, specialty, credentials, and office location.

        **Validates: Requirements 1.2**
        """
        result = _list_doctors(sandbox)
        doctors = sandbox.get_doctors()

        for doctor in doctors:
            # Check name appears
            assert doctor.name in result, f"Doctor name {doctor.name} not in result"
            # Check specialty appears
            assert doctor.specialty in result, f"Specialty {doctor.specialty} not in result"
            # Check credentials appear (as part of name line)
            assert doctor.credentials in result, f"Credentials {doctor.credentials} not in result"

        # Check location info appears
        assert "Location:" in result or "Test Medical Center" in result


class TestListDoctorsSpecialtyFilterProperties:
    """Property-based tests for list_doctors specialty filter."""

    @given(sandbox=sandbox_with_doctors_strategy(min_doctors=2, max_doctors=10))
    @settings(max_examples=100)
    def test_property_3_list_doctors_specialty_filter_correctness(self, sandbox):
        """
        Feature: sandbox-tool-improvements
        Property 3: List doctors specialty filter correctness

        For any sandbox containing doctors of various specialties, calling
        list_doctors with a specialty filter SHALL return only doctors
        matching that specialty.

        **Validates: Requirements 1.4**
        """
        doctors = sandbox.get_doctors()
        if not doctors:
            return  # Skip if no doctors

        # Get all unique specialties in the sandbox
        specialties = set(d.specialty for d in doctors)

        for specialty in specialties:
            result = _list_doctors(sandbox, specialty=specialty)

            # Count doctors with this specialty in sandbox
            matching_count = len([d for d in doctors if d.specialty == specialty])

            # Count occurrences of "Specialty: {specialty}" in result
            # This is more reliable than checking names which could be duplicated
            specialty_line = f"Specialty: {specialty}"
            result_count = result.count(specialty_line)

            assert result_count == matching_count, (
                f"Expected {matching_count} doctors with {specialty}, "
                f"but found {result_count} in result"
            )

            # Verify no other specialties appear in the result
            for other_specialty in specialties:
                if other_specialty != specialty:
                    other_line = f"Specialty: {other_specialty}"
                    assert other_line not in result, (
                        f"Found {other_specialty} in result when filtering for {specialty}"
                    )


# =============================================================================
# Property-Based Tests for Get Available Appointments
# =============================================================================


@st.composite
def slot_strategy(draw, doctor_id="doc_001", office_id="office_001"):
    """Generate a random appointment slot."""
    # Generate dates within a reasonable range
    year = 2026
    month = draw(st.integers(min_value=1, max_value=12))
    day = draw(st.integers(min_value=1, max_value=28))  # Safe for all months
    date = f"{year}-{month:02d}-{day:02d}"

    hour = draw(st.integers(min_value=8, max_value=17))
    minute = draw(st.sampled_from([0, 15, 30, 45]))
    time = f"{hour:02d}:{minute:02d}"

    return AppointmentSlot(
        id=draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=20)),
        doctor_id=doctor_id,
        office_id=office_id,
        date=date,
        time=time,
        duration_minutes=draw(st.sampled_from([15, 30, 45, 60])),
        available=True,  # Only generate available slots for this test
        appointment_type=draw(st.sampled_from(["in_person", "telehealth"])),
    )


@st.composite
def sandbox_with_slots_strategy(draw, min_slots=1, max_slots=20):
    """Generate a sandbox with random doctors and slots."""
    sandbox = HealthcareSandbox()

    # Add a default office
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

    # Generate doctors with different specialties
    specialties = [
        "Primary Care", "Cardiology", "Dermatology", "Endocrinology",
        "Orthopedics", "Psychiatry", "OB/GYN"
    ]
    num_doctors = draw(st.integers(min_value=1, max_value=4))
    for i in range(num_doctors):
        doc_id = f"doc_{i:03d}"
        sandbox.doctors[doc_id] = Doctor(
            id=doc_id,
            name=f"Dr. Test{i}",
            specialty=draw(st.sampled_from(specialties)),
            credentials="MD",
            office_id="office_001",
        )

    # Generate slots for each doctor
    doctor_ids = list(sandbox.doctors.keys())
    num_slots = draw(st.integers(min_value=min_slots, max_value=max_slots))
    for i in range(num_slots):
        doc_id = draw(st.sampled_from(doctor_ids))
        slot = draw(slot_strategy(doctor_id=doc_id))
        slot_id = f"slot_{i:03d}"
        slot = AppointmentSlot(
            id=slot_id,
            doctor_id=slot.doctor_id,
            office_id=slot.office_id,
            date=slot.date,
            time=slot.time,
            duration_minutes=slot.duration_minutes,
            available=slot.available,
            appointment_type=slot.appointment_type,
        )
        sandbox.slots[slot_id] = slot

    sandbox._initialized = True
    return sandbox


class TestGetAvailableAppointmentsProperties:
    """Property-based tests for get_available_appointments functionality."""

    @given(sandbox=sandbox_with_slots_strategy(min_slots=5, max_slots=20))
    @settings(max_examples=100)
    def test_property_4_date_range_filter_correctness(self, sandbox):
        """
        Feature: sandbox-tool-improvements
        Property 4: Get available appointments date range filter correctness

        For any sandbox containing appointment slots across multiple dates,
        calling get_available_appointments with a date range SHALL return
        only slots where start_date <= slot.date <= end_date.

        **Validates: Requirements 2.1**
        """
        all_slots = sandbox.get_available_slots()
        if not all_slots:
            return  # Skip if no slots

        # Get all unique dates
        dates = sorted(set(s.date for s in all_slots))
        if len(dates) < 2:
            return  # Need at least 2 dates to test range

        # Pick a date range that includes some but not all dates
        # Respect the 3-day max date range enforced by the function
        from datetime import date as date_cls, timedelta

        mid_idx = len(dates) // 2
        start_date = dates[0]
        end_date = dates[mid_idx]

        # Clamp to 3-day max range (matching the function's enforcement)
        parsed_start = date_cls.fromisoformat(start_date)
        parsed_end = date_cls.fromisoformat(end_date)
        if (parsed_end - parsed_start).days > 3:
            parsed_end = parsed_start + timedelta(days=3)
            end_date = parsed_end.isoformat()

        result = _get_available_appointments(sandbox, start_date, end_date)

        # Check that only slots within range are included
        for slot in all_slots:
            slot_in_range = start_date <= slot.date <= end_date

            if slot_in_range:
                # Slot should be in result (check by date since slot ID might not be unique)
                assert slot.date in result, f"Slot date {slot.date} should be in result"
            # Note: We can't easily check that out-of-range slots are NOT in result
            # because the date string might appear in other contexts

    @given(sandbox=sandbox_with_slots_strategy(min_slots=5, max_slots=20))
    @settings(max_examples=100)
    def test_property_5_output_contains_required_fields(self, sandbox):
        """
        Feature: sandbox-tool-improvements
        Property 5: Get available appointments output contains required fields

        For any slot returned by get_available_appointments, the output SHALL
        contain slot_id, date, time, appointment_type, office location, and
        doctor info (name, specialty, credentials).

        **Validates: Requirements 2.2, 2.3, 2.6**
        """
        all_slots = sandbox.get_available_slots()
        if not all_slots:
            return

        # Use a wide date range to get all slots
        dates = sorted(s.date for s in all_slots)
        start_date = dates[0]
        end_date = dates[-1]

        result = _get_available_appointments(sandbox, start_date, end_date)

        if "No available appointments" in result:
            return

        # Check that required fields are present in output
        assert "Slot ID:" in result
        assert "Date:" in result
        assert "Time:" in result
        assert "Type:" in result
        assert "Provider:" in result
        assert "Specialty:" in result
        assert "Location:" in result


class TestGetAvailableAppointmentsSpecialtyFilterProperties:
    """Property-based tests for get_available_appointments specialty filter."""

    @given(sandbox=sandbox_with_slots_strategy(min_slots=5, max_slots=20))
    @settings(max_examples=100)
    def test_property_6_specialty_filter_correctness(self, sandbox):
        """
        Feature: sandbox-tool-improvements
        Property 6: Get available appointments specialty filter correctness

        For any sandbox containing slots for doctors of various specialties,
        calling get_available_appointments with a specialty filter SHALL
        return only slots for doctors matching that specialty.

        **Validates: Requirements 2.4**
        """
        all_slots = sandbox.get_available_slots()
        if not all_slots:
            return

        # Get all unique specialties from doctors with slots
        doctor_specialties = {}
        for slot in all_slots:
            doctor = sandbox.get_doctor(slot.doctor_id)
            if doctor:
                doctor_specialties[slot.doctor_id] = doctor.specialty

        specialties = set(doctor_specialties.values())
        if len(specialties) < 2:
            return  # Need multiple specialties to test filter

        # Use a date range respecting the 3-day max enforced by the function
        from datetime import date as date_cls, timedelta

        dates = sorted(s.date for s in all_slots)
        start_date = dates[0]
        end_date = dates[-1]

        # Clamp to 3-day max range (matching the function's enforcement)
        parsed_start = date_cls.fromisoformat(start_date)
        parsed_end = date_cls.fromisoformat(end_date)
        if (parsed_end - parsed_start).days > 3:
            parsed_end = parsed_start + timedelta(days=3)
            end_date = parsed_end.isoformat()

        for specialty in specialties:
            result = _get_available_appointments(
                sandbox, start_date, end_date, specialty=specialty
            )

            if "No available appointments" in result:
                continue

            # Count expected slots for this specialty
            expected_count = sum(
                1 for s in all_slots
                if doctor_specialties.get(s.doctor_id) == specialty
                and start_date <= s.date <= end_date
            )

            # Count "Specialty: {specialty}" occurrences in result
            specialty_line = f"Specialty: {specialty}"
            result_count = result.count(specialty_line)

            assert result_count == expected_count, (
                f"Expected {expected_count} slots with {specialty}, "
                f"but found {result_count}"
            )

            # Verify no other specialties appear
            for other_specialty in specialties:
                if other_specialty != specialty:
                    other_line = f"Specialty: {other_specialty}"
                    assert other_line not in result, (
                        f"Found {other_specialty} when filtering for {specialty}"
                    )
