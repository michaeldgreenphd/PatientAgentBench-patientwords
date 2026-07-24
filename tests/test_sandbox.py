# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for Healthcare Sandbox.

Tests the sandbox data models and HealthcareSandbox class functionality.
"""

import threading
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed

from patient_agent_bench.sandbox import (
    HealthcareSandbox,
    OfficeLocation,
    Doctor,
    AppointmentSlot,
    BookedAppointment,
    Medication,
    MedicationStatus,
    initialize_sandbox,
)
from patient_agent_bench.patient import PatientProfile


def create_test_patient_profile(**kwargs) -> PatientProfile:
    """Helper to create a PatientProfile for testing with sensible defaults."""
    # Pop phone first to check if it exists
    phone = kwargs.pop("phone", None)
    insurance_provider = kwargs.pop("insurance_provider", None)
    pharmacy_name = kwargs.pop("pharmacy_name", None)

    defaults = {
        "personal_info": {
            "first_name": kwargs.pop("first_name", "John"),
            "last_name": kwargs.pop("last_name", "Doe"),
            "preferred_name": kwargs.pop("preferred_name", ""),
            "dob": kwargs.pop("dob", ""),
            "sex": kwargs.pop("sex", ""),
            "pronouns": kwargs.pop("pronouns", ""),
        },
        "account_info": {
            "age_in_years": str(kwargs.pop("age", 45)),
        },
        "addresses": {
            "address": {
                "address1": kwargs.pop("address", ""),
                "city": kwargs.pop("city", "Boston"),
                "state": kwargs.pop("state", "MA"),
                "zip": kwargs.pop("zip_code", "02101"),
            }
        },
        "phone_numbers": {"phone": {"number": phone}} if phone else {},
        "insurances": {
            "insurance": {
                "name": insurance_provider,
                "plan_type": kwargs.pop("insurance_plan_type", ""),
            }
        } if insurance_provider else {},
        "pharmacies": {
            "pharmacy": {
                "name": pharmacy_name,
                "address": kwargs.pop("pharmacy_address", ""),
                "phone": kwargs.pop("pharmacy_phone", ""),
            }
        } if pharmacy_name else {},
        "emergency_contacts": {},
        "care_team": [],  # List format for care_team
        "medications": kwargs.pop("medications", []),
    }
    # Consume legacy fields if passed (for backward compat)
    kwargs.pop("insurance_member_id", None)
    kwargs.pop("insurance_group", None)

    # Handle pcp_id and pcp_name - add to care_team list
    pcp_id = kwargs.pop("pcp_id", None)
    pcp_name = kwargs.pop("pcp_name", None)

    profile = PatientProfile.from_dict(defaults)

    # Use update_pcp method to add PCP to care_team
    if pcp_id and pcp_name:
        profile.update_pcp(pcp_id, pcp_name)

    return profile


# =============================================================================
# Data Model Tests (Task 1)
# =============================================================================


class TestOfficeLocation:
    """Tests for OfficeLocation dataclass."""

    def test_create_office(self):
        """Test creating an office location."""
        office = OfficeLocation(
            id="office_001",
            name="Downtown Primary Care",
            address="123 Main St",
            city="Boston",
            state="MA",
            zip_code="02101",
            phone="555-123-4567",
            hours="Mon-Fri 8AM-6PM",
        )
        assert office.id == "office_001"
        assert office.name == "Downtown Primary Care"
        assert office.city == "Boston"

    def test_office_to_dict(self):
        """Test converting office to dictionary."""
        office = OfficeLocation(
            id="office_001",
            name="Test Office",
            address="123 Main",
            city="Boston",
            state="MA",
            zip_code="02101",
            phone="555-1234",
            hours="Mon-Fri",
        )
        data = office.to_dict()
        assert data["id"] == "office_001"
        assert data["name"] == "Test Office"
        assert "city" in data

    def test_office_from_dict(self):
        """Test creating office from dictionary."""
        data = {
            "id": "office_002",
            "name": "Specialist Center",
            "address": "456 Oak Ave",
            "city": "Cambridge",
            "state": "MA",
            "zip_code": "02139",
            "phone": "555-9876",
            "hours": "Mon-Sat 9AM-5PM",
        }
        office = OfficeLocation.from_dict(data)
        assert office.id == "office_002"
        assert office.city == "Cambridge"

    def test_office_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = OfficeLocation(
            id="o1", name="Test", address="123", city="Boston",
            state="MA", zip_code="02101", phone="555", hours="9-5"
        )
        restored = OfficeLocation.from_dict(original.to_dict())
        assert original.id == restored.id
        assert original.name == restored.name


class TestDoctor:
    """Tests for Doctor dataclass."""

    def test_create_doctor(self):
        """Test creating a doctor."""
        doctor = Doctor(
            id="doc_001",
            name="Dr. Sarah Johnson",
            specialty="Primary Care",
            credentials="MD",
            office_id="office_001",
        )
        assert doctor.id == "doc_001"
        assert doctor.name == "Dr. Sarah Johnson"
        assert doctor.specialty == "Primary Care"

    def test_doctor_to_dict(self):
        """Test converting doctor to dictionary."""
        doctor = Doctor(
            id="doc_001",
            name="Dr. Smith",
            specialty="Cardiology",
            credentials="MD",
            office_id="office_001",
        )
        data = doctor.to_dict()
        assert data["specialty"] == "Cardiology"
        assert data["credentials"] == "MD"

    def test_doctor_from_dict(self):
        """Test creating doctor from dictionary."""
        data = {
            "id": "doc_002",
            "name": "Dr. Jane Doe",
            "specialty": "Dermatology",
            "credentials": "DO",
            "office_id": "office_002",
        }
        doctor = Doctor.from_dict(data)
        assert doctor.credentials == "DO"


class TestAppointmentSlot:
    """Tests for AppointmentSlot dataclass."""

    def test_create_slot(self):
        """Test creating an appointment slot."""
        slot = AppointmentSlot(
            id="slot_001",
            doctor_id="doc_001",
            office_id="office_001",
            date="2026-01-25",
            time="09:00",
            duration_minutes=30,
            available=True,
            appointment_type="in_person",
        )
        assert slot.available is True
        assert slot.duration_minutes == 30

    def test_slot_to_dict(self):
        """Test converting slot to dictionary."""
        slot = AppointmentSlot(
            id="s1", doctor_id="d1", office_id="o1",
            date="2026-01-25", time="10:00", duration_minutes=30,
            available=True, appointment_type="telehealth"
        )
        data = slot.to_dict()
        assert data["appointment_type"] == "telehealth"
        assert data["available"] is True

    def test_slot_from_dict_defaults(self):
        """Test slot from_dict with missing fields uses defaults."""
        data = {"id": "s1", "doctor_id": "d1", "office_id": "o1"}
        slot = AppointmentSlot.from_dict(data)
        assert slot.duration_minutes == 30  # default
        assert slot.available is True  # default
        assert slot.appointment_type == "in_person"  # default


class TestBookedAppointment:
    """Tests for BookedAppointment dataclass."""

    def test_create_booked_appointment(self):
        """Test creating a booked appointment."""
        appt = BookedAppointment(
            id="appt_001",
            slot_id="slot_001",
            doctor_id="doc_001",
            office_id="office_001",
            date="2026-01-25",
            time="09:00",
            appointment_type="in_person",
            reason="Annual checkup",
        )
        assert appt.reason == "Annual checkup"

    def test_booked_appointment_optional_reason(self):
        """Test booked appointment with no reason."""
        appt = BookedAppointment(
            id="appt_001",
            slot_id="slot_001",
            doctor_id="doc_001",
            office_id="office_001",
            date="2026-01-25",
            time="09:00",
            appointment_type="in_person",
        )
        assert appt.reason is None


class TestMedication:
    """Tests for Medication dataclass."""

    def test_create_medication(self):
        """Test creating a medication."""
        med = Medication(
            id="med_001",
            name="Metformin",
            dosage="500mg",
            frequency="twice daily",
            status=MedicationStatus.ACTIVE,
            prescribed_date="2025-01-01",
            pharmacy="CVS",
            refills_remaining=3,
            last_filled="2025-12-01",
            reason="Type 2 Diabetes",
        )
        assert med.status == MedicationStatus.ACTIVE
        assert med.refills_remaining == 3

    def test_medication_status_enum(self):
        """Test MedicationStatus enum values."""
        assert MedicationStatus.ACTIVE.value == "active"
        assert MedicationStatus.PAST.value == "past"
        assert MedicationStatus.UNDER_RENEWAL.value == "under_renewal"

    def test_medication_to_dict(self):
        """Test medication to_dict serializes status as string."""
        med = Medication(
            id="m1", name="Test", dosage="10mg", frequency="daily",
            status=MedicationStatus.ACTIVE, prescribed_date="2025-01-01",
            pharmacy="CVS", refills_remaining=2
        )
        data = med.to_dict()
        assert data["status"] == "active"  # string, not enum

    def test_medication_from_dict_string_status(self):
        """Test medication from_dict handles string status."""
        data = {
            "id": "m1", "name": "Test", "dosage": "10mg", "frequency": "daily",
            "status": "under_renewal", "prescribed_date": "2025-01-01",
            "pharmacy": "CVS", "refills_remaining": 0
        }
        med = Medication.from_dict(data)
        assert med.status == MedicationStatus.UNDER_RENEWAL


class TestPatientProfile:
    """Tests for PatientProfile dataclass property accessors."""

    def test_create_patient_profile(self):
        """Test creating a patient profile with helper."""
        profile = create_test_patient_profile(
            first_name="John",
            last_name="Doe",
            age=45,
            city="Boston",
            state="MA",
        )
        assert profile.personal_info.first_name == "John"
        assert int(profile.account_info.age_in_years) == 45

    def test_patient_profile_defaults(self):
        """Test patient profile has sensible defaults."""
        profile = PatientProfile()
        assert profile.personal_info.first_name == ""
        assert profile.get_pcp() is None
        assert profile.account_info.age_in_years == ""

    def test_patient_profile_to_xml(self):
        """Test patient profile XML generation."""
        profile = create_test_patient_profile(
            first_name="John",
            last_name="Doe",
            age=45,
            phone="555-1234",
            insurance_provider="Blue Cross",
        )
        xml = profile.to_xml()
        assert "John" in xml
        assert "45" in xml
        assert "Blue Cross" in xml


# =============================================================================
# HealthcareSandbox Tests (Task 2)
# =============================================================================


class TestHealthcareSandboxInit:
    """Tests for HealthcareSandbox initialization."""

    def test_create_sandbox(self):
        """Test creating an empty sandbox."""
        sandbox = HealthcareSandbox()
        assert sandbox.initialized is False
        assert len(sandbox.offices) == 0
        assert len(sandbox.doctors) == 0
        assert sandbox.patient_profile is None

    def test_sandbox_has_lock(self):
        """Test sandbox has threading lock."""
        sandbox = HealthcareSandbox()
        assert hasattr(sandbox, "_lock")
        assert isinstance(sandbox._lock, type(threading.RLock()))

    def test_get_sandbox_returns_self(self):
        """Test get_sandbox returns the sandbox instance."""
        sandbox = HealthcareSandbox()
        assert sandbox.get_sandbox() is sandbox


class TestHealthcareSandboxOffices:
    """Tests for office-related sandbox methods."""

    @pytest.fixture
    def sandbox_with_offices(self):
        """Create sandbox with test offices."""
        sandbox = HealthcareSandbox()
        sandbox.offices["o1"] = OfficeLocation(
            "o1", "Downtown Clinic", "123 Main", "Boston", "MA", "02101", "555-1111", "Mon-Fri"
        )
        sandbox.offices["o2"] = OfficeLocation(
            "o2", "Suburban Center", "456 Oak", "Cambridge", "MA", "02139", "555-2222", "Mon-Sat"
        )
        return sandbox

    def test_get_offices(self, sandbox_with_offices):
        """Test getting all offices."""
        offices = sandbox_with_offices.get_offices()
        assert len(offices) == 2

    def test_get_office_by_id(self, sandbox_with_offices):
        """Test getting office by ID."""
        office = sandbox_with_offices.get_office("o1")
        assert office is not None
        assert office.name == "Downtown Clinic"

    def test_get_office_not_found(self, sandbox_with_offices):
        """Test getting non-existent office."""
        office = sandbox_with_offices.get_office("nonexistent")
        assert office is None


class TestHealthcareSandboxDoctors:
    """Tests for doctor-related sandbox methods."""

    @pytest.fixture
    def sandbox_with_doctors(self):
        """Create sandbox with test doctors."""
        sandbox = HealthcareSandbox()
        sandbox.doctors["d1"] = Doctor("d1", "Dr. Sarah Johnson", "Primary Care", "MD", "o1")
        sandbox.doctors["d2"] = Doctor("d2", "Dr. Michael Chen", "Cardiology", "MD", "o1")
        sandbox.doctors["d3"] = Doctor("d3", "Dr. Emily Brown", "Dermatology", "DO", "o2")
        sandbox.pcp_id = "d1"
        return sandbox

    def test_get_doctors(self, sandbox_with_doctors):
        """Test getting all doctors."""
        doctors = sandbox_with_doctors.get_doctors()
        assert len(doctors) == 3

    def test_get_doctor_by_id(self, sandbox_with_doctors):
        """Test getting doctor by ID."""
        doctor = sandbox_with_doctors.get_doctor("d2")
        assert doctor is not None
        assert doctor.specialty == "Cardiology"

    def test_get_doctor_by_name(self, sandbox_with_doctors):
        """Test finding doctor by name."""
        doctor = sandbox_with_doctors.get_doctor_by_name("Chen")
        assert doctor is not None
        assert doctor.id == "d2"

    def test_get_doctor_by_name_case_insensitive(self, sandbox_with_doctors):
        """Test doctor name search is case-insensitive."""
        doctor = sandbox_with_doctors.get_doctor_by_name("SARAH")
        assert doctor is not None
        assert doctor.id == "d1"

    def test_get_doctor_by_name_not_found(self, sandbox_with_doctors):
        """Test doctor name search returns None when not found."""
        doctor = sandbox_with_doctors.get_doctor_by_name("Nonexistent")
        assert doctor is None

    def test_get_pcp(self, sandbox_with_doctors):
        """Test getting PCP."""
        pcp = sandbox_with_doctors.get_pcp()
        assert pcp is not None
        assert pcp.name == "Dr. Sarah Johnson"

    def test_get_pcp_when_none(self):
        """Test getting PCP when not assigned."""
        sandbox = HealthcareSandbox()
        assert sandbox.get_pcp() is None

    def test_resolve_doctor_pcp(self, sandbox_with_doctors):
        """Test resolving 'PCP' to actual doctor."""
        doctor = sandbox_with_doctors.resolve_doctor("PCP")
        assert doctor is not None
        assert doctor.id == "d1"

    def test_resolve_doctor_by_name(self, sandbox_with_doctors):
        """Test resolving doctor by name."""
        doctor = sandbox_with_doctors.resolve_doctor("Emily Brown")
        assert doctor is not None
        assert doctor.specialty == "Dermatology"


class TestHealthcareSandboxAppointments:
    """Tests for appointment-related sandbox methods."""

    @pytest.fixture
    def sandbox_with_slots(self):
        """Create sandbox with test appointment slots."""
        sandbox = HealthcareSandbox()
        sandbox.doctors["d1"] = Doctor("d1", "Dr. Smith", "Primary Care", "MD", "o1")
        sandbox.slots["s1"] = AppointmentSlot(
            "s1", "d1", "o1", "2026-01-25", "09:00", 30, True, "in_person"
        )
        sandbox.slots["s2"] = AppointmentSlot(
            "s2", "d1", "o1", "2026-01-25", "10:00", 30, True, "telehealth"
        )
        sandbox.slots["s3"] = AppointmentSlot(
            "s3", "d1", "o1", "2026-01-26", "09:00", 30, False, "in_person"
        )
        return sandbox

    def test_get_available_slots(self, sandbox_with_slots):
        """Test getting all available slots."""
        slots = sandbox_with_slots.get_available_slots()
        assert len(slots) == 2  # s3 is unavailable

    def test_get_available_slots_by_date(self, sandbox_with_slots):
        """Test filtering slots by date."""
        slots = sandbox_with_slots.get_available_slots(date="2026-01-25")
        assert len(slots) == 2

    def test_get_available_slots_by_type(self, sandbox_with_slots):
        """Test filtering slots by appointment type."""
        slots = sandbox_with_slots.get_available_slots(appointment_type="telehealth")
        assert len(slots) == 1
        assert slots[0].id == "s2"

    def test_book_appointment(self, sandbox_with_slots):
        """Test booking an appointment."""
        appt = sandbox_with_slots.book_appointment("s1", reason="Checkup")
        
        assert appt is not None
        assert appt.slot_id == "s1"
        assert appt.reason == "Checkup"
        assert sandbox_with_slots.slots["s1"].available is False

    def test_book_appointment_unavailable_slot(self, sandbox_with_slots):
        """Test booking unavailable slot returns None."""
        appt = sandbox_with_slots.book_appointment("s3")
        assert appt is None

    def test_book_appointment_nonexistent_slot(self, sandbox_with_slots):
        """Test booking nonexistent slot returns None."""
        appt = sandbox_with_slots.book_appointment("nonexistent")
        assert appt is None

    def test_cancel_appointment(self, sandbox_with_slots):
        """Test cancelling an appointment."""
        # First book
        appt = sandbox_with_slots.book_appointment("s1")
        assert sandbox_with_slots.slots["s1"].available is False
        
        # Then cancel
        result = sandbox_with_slots.cancel_appointment(appt.id)
        assert result is True
        assert sandbox_with_slots.slots["s1"].available is True
        assert appt.id not in sandbox_with_slots.booked_appointments

    def test_cancel_nonexistent_appointment(self, sandbox_with_slots):
        """Test cancelling nonexistent appointment returns False."""
        result = sandbox_with_slots.cancel_appointment("nonexistent")
        assert result is False

    def test_get_booked_appointments(self, sandbox_with_slots):
        """Test getting booked appointments."""
        sandbox_with_slots.book_appointment("s1")
        sandbox_with_slots.book_appointment("s2")
        
        booked = sandbox_with_slots.get_booked_appointments()
        assert len(booked) == 2


class TestHealthcareSandboxMedications:
    """Tests for medication-related sandbox methods."""

    @pytest.fixture
    def sandbox_with_meds(self):
        """Create sandbox with test medications via patient_profile."""
        sandbox = HealthcareSandbox()
        sandbox.patient_profile = PatientProfile()
        sandbox.patient_profile.add_medication(Medication(
            "m1", "Metformin", "500mg", "twice daily", MedicationStatus.ACTIVE,
            "2025-01-01", "CVS", 3, "2025-12-01", "Diabetes"
        ))
        sandbox.patient_profile.add_medication(Medication(
            "m2", "Lisinopril", "10mg", "once daily", MedicationStatus.ACTIVE,
            "2025-06-01", "CVS", 5, "2025-11-01", "Blood pressure"
        ))
        sandbox.patient_profile.add_medication(Medication(
            "m3", "Amoxicillin", "500mg", "three times daily", MedicationStatus.PAST,
            "2024-01-01", "Walgreens", 0, "2024-01-01", "Infection"
        ))
        return sandbox

    def test_get_all_medications(self, sandbox_with_meds):
        """Test getting all medications."""
        meds = sandbox_with_meds.get_medications()
        assert len(meds) == 3

    def test_get_medications_by_status(self, sandbox_with_meds):
        """Test filtering medications by status."""
        active = sandbox_with_meds.get_medications(MedicationStatus.ACTIVE)
        assert len(active) == 2

        past = sandbox_with_meds.get_medications(MedicationStatus.PAST)
        assert len(past) == 1

    def test_request_refill_success(self, sandbox_with_meds):
        """Test successful refill request."""
        med = sandbox_with_meds.patient_profile.get_medication("m1")
        original_refills = med.refills_remaining
        result = sandbox_with_meds.request_refill("m1")

        assert result is True
        assert med.refills_remaining == original_refills - 1
        assert med.last_filled is not None

    def test_request_refill_past_medication(self, sandbox_with_meds):
        """Test refill request for past medication fails."""
        result = sandbox_with_meds.request_refill("m3")
        assert result is False

    def test_request_refill_no_refills(self, sandbox_with_meds):
        """Test refill request with no refills remaining fails."""
        sandbox_with_meds.patient_profile.get_medication("m1").refills_remaining = 0
        result = sandbox_with_meds.request_refill("m1")
        assert result is False

    def test_request_refill_nonexistent(self, sandbox_with_meds):
        """Test refill request for nonexistent medication fails."""
        result = sandbox_with_meds.request_refill("nonexistent")
        assert result is False

    def test_add_medication(self, sandbox_with_meds):
        """Test adding a new medication."""
        new_med = Medication(
            "m4", "NewMed", "25mg", "daily", MedicationStatus.UNDER_RENEWAL,
            "2026-01-20", "CVS", 2
        )
        sandbox_with_meds.add_medication(new_med)
        
        assert sandbox_with_meds.patient_profile.get_medication("m4") is not None
        assert sandbox_with_meds.patient_profile.get_medication("m4").name == "NewMed"


class TestHealthcareSandboxProfile:
    """Tests for profile-related sandbox methods."""

    @pytest.fixture
    def sandbox_with_profile(self):
        """Create sandbox with test profile."""
        sandbox = HealthcareSandbox()
        sandbox.patient_profile = create_test_patient_profile(
            first_name="John",
            last_name="Doe",
            phone="555-1234",
            city="Boston",
            state="MA",
        )
        return sandbox

    def test_get_profile(self, sandbox_with_profile):
        """Test getting patient profile."""
        profile = sandbox_with_profile.get_profile()
        assert profile is not None
        assert profile.personal_info.first_name == "John"

    def test_get_profile_when_none(self):
        """Test getting profile when not set."""
        sandbox = HealthcareSandbox()
        assert sandbox.get_profile() is None

    def test_update_profile(self, sandbox_with_profile):
        """Test updating profile fields."""
        result = sandbox_with_profile.update_profile(phone="555-9999")
        
        assert result is True
        assert sandbox_with_profile.patient_profile.phone.number == "555-9999"

    def test_update_profile_when_none(self):
        """Test updating profile when not set returns False."""
        sandbox = HealthcareSandbox()
        result = sandbox.update_profile(phone="555-1234")
        assert result is False


class TestHealthcareSandboxThreadSafety:
    """Tests for thread safety of sandbox operations."""

    def test_concurrent_slot_booking(self):
        """Test that concurrent booking attempts are handled safely."""
        sandbox = HealthcareSandbox()
        sandbox.doctors["d1"] = Doctor("d1", "Dr. Smith", "Primary Care", "MD", "o1")
        
        # Create a single slot
        sandbox.slots["s1"] = AppointmentSlot(
            "s1", "d1", "o1", "2026-01-25", "09:00", 30, True, "in_person"
        )
        
        results = []
        
        def try_book():
            result = sandbox.book_appointment("s1", reason="Test")
            results.append(result)
        
        # Try to book the same slot from multiple threads
        threads = [threading.Thread(target=try_book) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one should succeed
        successful = [r for r in results if r is not None]
        assert len(successful) == 1

    def test_concurrent_refill_requests(self):
        """Test that concurrent refill requests are handled safely."""
        sandbox = HealthcareSandbox()
        sandbox.patient_profile = PatientProfile()
        sandbox.patient_profile.add_medication(Medication(
            "m1", "Test", "10mg", "daily", MedicationStatus.ACTIVE,
            "2025-01-01", "CVS", 3
        ))

        results = []

        def try_refill():
            result = sandbox.request_refill("m1")
            results.append(result)

        # Try to refill from multiple threads
        threads = [threading.Thread(target=try_refill) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have exactly 3 successes (initial refills_remaining)
        successful = [r for r in results if r is True]
        assert len(successful) == 3
        med = sandbox.patient_profile.get_medication("m1")
        assert med.refills_remaining == 0


# =============================================================================
# Sandbox Generator Tests (Task 3)
# =============================================================================


class TestSandboxGeneratorHelpers:
    """Tests for sandbox generator helper functions."""

    def test_parse_llm_response_valid_json(self):
        """Test parsing valid JSON response."""
        from patient_agent_bench.sandbox.generator import _parse_llm_response

        response = '{"offices": [], "doctors": []}'
        result = _parse_llm_response(response)
        assert result == {"offices": [], "doctors": []}

    def test_parse_llm_response_with_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        from patient_agent_bench.sandbox.generator import _parse_llm_response

        response = '```json\n{"offices": [], "doctors": []}\n```'
        result = _parse_llm_response(response)
        assert result == {"offices": [], "doctors": []}

    def test_parse_llm_response_invalid_json(self):
        """Test parsing invalid JSON raises error."""
        from patient_agent_bench.sandbox.generator import _parse_llm_response
        import json

        with pytest.raises(json.JSONDecodeError):
            _parse_llm_response("not valid json")

    def test_generate_appointment_slots(self):
        """Test appointment slot generation."""
        from patient_agent_bench.sandbox.generator import _generate_appointment_slots

        doctors = [
            Doctor("d1", "Dr. Smith", "Primary Care", "MD", "o1"),
            Doctor("d2", "Dr. Jones", "Cardiology", "MD", "o1"),
        ]

        slots = _generate_appointment_slots(doctors, current_date="2026-02-11")

        # Should generate slots for 14 days (minus weekends) for each doctor
        assert len(slots) > 0

        # All slots should have valid structure
        for slot in slots:
            assert slot.doctor_id in ["d1", "d2"]
            assert slot.duration_minutes == 30
            assert slot.appointment_type in ["in_person", "telehealth"]

    def test_parse_offices(self):
        """Test parsing office data using generic parser."""
        from patient_agent_bench.sandbox.generator import _parse_model_list
        from patient_agent_bench.sandbox.sandbox import OfficeLocation

        data = [
            {
                "id": "o1",
                "name": "Test Office",
                "address": "123 Main",
                "city": "Boston",
                "state": "MA",
                "zip_code": "02101",
                "phone": "555-1234",
                "hours": "Mon-Fri",
            }
        ]

        offices = _parse_model_list(data, OfficeLocation)
        assert "o1" in offices
        assert offices["o1"].name == "Test Office"

    def test_parse_doctors(self):
        """Test parsing doctor data using generic parser."""
        from patient_agent_bench.sandbox.generator import _parse_model_list
        from patient_agent_bench.sandbox.sandbox import Doctor

        data = [
            {
                "id": "d1",
                "name": "Dr. Smith",
                "specialty": "Primary Care",
                "credentials": "MD",
                "office_id": "o1",
            }
        ]

        doctors = _parse_model_list(data, Doctor)
        assert "d1" in doctors
        assert doctors["d1"].specialty == "Primary Care"

    def test_parse_medications(self):
        """Test parsing medication data using generic parser."""
        from patient_agent_bench.sandbox.generator import _parse_model_list
        from patient_agent_bench.sandbox.sandbox import Medication

        data = [
            {
                "id": "m1",
                "name": "Metformin",
                "dosage": "500mg",
                "frequency": "twice daily",
                "status": "active",
                "prescribed_date": "2025-01-01",
                "pharmacy": "CVS",
                "refills_remaining": 3,
            }
        ]

        medications = _parse_model_list(data, Medication)
        assert "m1" in medications
        assert medications["m1"].status == MedicationStatus.ACTIVE

    def test_assign_pcp_with_primary_care(self):
        """Test PCP assignment prefers primary care doctors."""
        from patient_agent_bench.sandbox.generator import _assign_pcp
        import random

        random.seed(42)  # For reproducibility

        doctors = {
            "d1": Doctor("d1", "Dr. Smith", "Primary Care", "MD", "o1"),
            "d2": Doctor("d2", "Dr. Jones", "Cardiology", "MD", "o1"),
        }

        # Run multiple times to check distribution
        pcp_ids = [_assign_pcp(doctors) for _ in range(100)]

        # Should mostly assign primary care doctor
        primary_care_count = sum(1 for p in pcp_ids if p == "d1")
        assert primary_care_count > 50  # Should be majority

    def test_assign_pcp_empty_doctors(self):
        """Test PCP assignment with no doctors returns None."""
        from patient_agent_bench.sandbox.generator import _assign_pcp

        assert _assign_pcp({}) is None

    def test_build_generation_prompt(self):
        """Test prompt building includes patient context."""
        from patient_agent_bench.sandbox.generator import _build_generation_prompt

        prompt = _build_generation_prompt(city="Boston", state="MA", current_date="2026-02-11")

        assert "Boston" in prompt
        assert "MA" in prompt


class TestSandboxInitialize:
    """Tests for HealthcareSandbox.initialize() method."""

    @pytest.fixture
    def mock_patient_profile(self):
        """Create a mock patient profile with medications."""
        return create_test_patient_profile(
            first_name="John",
            last_name="Doe",
            sex="Male",
            age=45,
            city="Boston",
            state="MA",
            zip_code="02101",
            medications=[
                {
                    "id": "med_001",
                    "name": "Metformin",
                    "dosage": "500mg",
                    "frequency": "twice daily",
                    "status": "active",
                    "prescribed_date": "2025-01-01",
                    "pharmacy": "CVS",
                    "refills_remaining": 3,
                    "last_filled": "2025-12-01",
                    "reason": "Type 2 Diabetes",
                }
            ],
        )

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response (offices and doctors only - medications from profile)."""
        return {
            "offices": [
                {
                    "id": "office_001",
                    "name": "Downtown Clinic",
                    "address": "123 Main St",
                    "city": "Boston",
                    "state": "MA",
                    "zip_code": "02101",
                    "phone": "555-1234",
                    "hours": "Mon-Fri 8AM-6PM",
                }
            ],
            "doctors": [
                {
                    "id": "doc_001",
                    "name": "Dr. Sarah Johnson",
                    "specialty": "Primary Care",
                    "credentials": "MD",
                    "office_id": "office_001",
                },
                {
                    "id": "doc_002",
                    "name": "Dr. Michael Chen",
                    "specialty": "Cardiology",
                    "credentials": "MD",
                    "office_id": "office_001",
                },
            ],
            # Note: medications removed - now come from patient_profile per refactor
        }

    @pytest.mark.asyncio
    async def test_initialize_with_mock_llm(self, mock_patient_profile, mock_llm_response):
        """Test sandbox initialization with mocked LLM."""
        import json
        from unittest.mock import AsyncMock, MagicMock

        # Create mock LLM client
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_llm_response)
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        sandbox = HealthcareSandbox()

        await initialize_sandbox(
            sandbox=sandbox,
            patient_profile=mock_patient_profile,
            llm_client=mock_llm,
            current_datetime="Wednesday, February 11, 2026 at 10:30 AM",
        )

        # Verify sandbox was initialized
        assert sandbox.initialized is True
        assert len(sandbox.offices) == 1
        assert len(sandbox.doctors) == 2
        assert len(sandbox.patient_profile.medications) == 1
        assert len(sandbox.slots) > 0  # Generated slots

        # Verify patient profile was set
        assert sandbox.patient_profile is not None
        assert sandbox.patient_profile.personal_info.first_name == "John"

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, mock_patient_profile, mock_llm_response):
        """Test that re-initialization is skipped."""
        import json
        from unittest.mock import AsyncMock, MagicMock

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_llm_response)
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        sandbox = HealthcareSandbox()

        # First initialization
        await initialize_sandbox(
            sandbox=sandbox,
            patient_profile=mock_patient_profile,
            llm_client=mock_llm,
            current_datetime="Wednesday, February 11, 2026 at 10:30 AM",
        )

        # Second initialization should be skipped
        await initialize_sandbox(
            sandbox=sandbox,
            patient_profile=mock_patient_profile,
            llm_client=mock_llm,
            current_datetime="Wednesday, February 11, 2026 at 10:30 AM",
        )

        # LLM should only be called once
        assert mock_llm.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_initialize_retry_on_invalid_json(self, mock_patient_profile, mock_llm_response):
        """Test that initialization retries on invalid JSON."""
        import json
        from unittest.mock import AsyncMock, MagicMock

        mock_llm = MagicMock()

        # First call returns invalid JSON, second returns valid
        invalid_response = MagicMock()
        invalid_response.content = "not valid json"

        valid_response = MagicMock()
        valid_response.content = json.dumps(mock_llm_response)

        mock_llm.ainvoke = AsyncMock(side_effect=[invalid_response, valid_response])

        sandbox = HealthcareSandbox()

        await initialize_sandbox(
            sandbox=sandbox,
            patient_profile=mock_patient_profile,
            llm_client=mock_llm,
            current_datetime="Wednesday, February 11, 2026 at 10:30 AM",
        )

        # Should have retried and succeeded
        assert sandbox.initialized is True
        assert mock_llm.ainvoke.call_count == 2


# =============================================================================
# Property-Based Tests for Sandbox Medication Passthrough (Task 8.4)
# =============================================================================

from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Strategy for generating valid medication status
_medication_status_strategy = st.sampled_from(["active", "past", "under_renewal"])

# Strategy for generating a valid medication dictionary
_medication_strategy = st.fixed_dictionaries({
    "id": st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_",
        min_size=1,
        max_size=20
    ).map(lambda s: f"med_{s}"),
    "name": st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
    "dosage": st.text(min_size=1, max_size=20).filter(lambda s: s.strip()),
    "frequency": st.text(min_size=1, max_size=30).filter(lambda s: s.strip()),
    "status": _medication_status_strategy,
    "prescribed_date": st.dates().map(lambda d: d.isoformat()),
    "pharmacy": st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
    "refills_remaining": st.integers(min_value=0, max_value=12),
    "last_filled": st.one_of(
        st.none(),
        st.dates().map(lambda d: d.isoformat())
    ),
    "reason": st.one_of(
        st.none(),
        st.text(min_size=1, max_size=100).filter(lambda s: s.strip())
    ),
})


# Strategy for generating a list of medications with unique IDs
@st.composite
def _medications_list_strategy(draw, min_size=1, max_size=5):
    """Generate a list of medications with unique IDs."""
    num_meds = draw(st.integers(min_value=min_size, max_value=max_size))
    medications = []
    used_ids = set()

    for i in range(num_meds):
        med = draw(_medication_strategy)
        # Ensure unique ID
        base_id = med["id"]
        unique_id = base_id
        counter = 0
        while unique_id in used_ids:
            counter += 1
            unique_id = f"{base_id}_{counter}"
        med["id"] = unique_id
        used_ids.add(unique_id)
        medications.append(med)

    return medications


class TestPropertySandboxMedicationPassthrough:
    """
    Property-based tests for sandbox medication passthrough.

    **Feature: benchmark-seed-refactor, Property 9: Sandbox Medication Passthrough**

    Property 9: Sandbox Medication Passthrough
    *For any* PatientProfile containing a non-empty medications list, after sandbox
    initialization, the sandbox's medications dictionary SHALL contain Medication
    objects with matching ids and names from the patient_profile.

    **Validates: Requirements 5.5**
    """

    @given(medications_list=_medications_list_strategy())
    @settings(max_examples=100)
    def test_property_9_sandbox_medication_passthrough(self, medications_list):
        """
        Property 9: Sandbox Medication Passthrough

        *For any* PatientProfile containing a non-empty medications list, after sandbox
        initialization, the patient_profile.medications list SHALL contain Medication
        objects with matching ids and names.

        **Validates: Requirements 5.5**
        """
        from patient_agent_bench.patient import PatientProfile, Medication
        import asyncio
        import json
        from unittest.mock import AsyncMock, MagicMock

        # Assume we have at least one medication
        assume(len(medications_list) > 0)

        # Create PatientProfile with the generated medications
        patient_profile = create_test_patient_profile(
            first_name="Test",
            last_name="Patient",
            sex="Male",
            age=30,
            city="Boston",
            state="MA",
            medications=medications_list,
        )

        # Create mock LLM client that returns minimal valid response
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "offices": [{
                "id": "office_001",
                "name": "Test Clinic",
                "address": "123 Main St",
                "city": "Boston",
                "state": "MA",
                "zip_code": "02101",
                "phone": "555-1234",
                "hours": "Mon-Fri 9-5",
            }],
            "doctors": [{
                "id": "doc_001",
                "name": "Dr. Test",
                "specialty": "Primary Care",
                "credentials": "MD",
                "office_id": "office_001",
            }],
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Run the async function - test via initialize_sandbox()
        async def run_test():
            sandbox = HealthcareSandbox()
            await initialize_sandbox(
                sandbox=sandbox,
                patient_profile=patient_profile,
                llm_client=mock_llm,
                current_datetime="Wednesday, February 11, 2026 at 10:30 AM",
            )
            return sandbox

        sandbox = asyncio.run(run_test())

        # Verify medications from patient_profile are accessible via sandbox
        profile_medications = sandbox.patient_profile.medications

        # Property assertion: patient_profile medications should match input
        assert len(profile_medications) == len(medications_list), (
            f"Expected {len(medications_list)} medications, got {len(profile_medications)}"
        )

        # Verify each medication has matching id and name
        for i, med_data in enumerate(medications_list):
            profile_med = profile_medications[i]
            assert isinstance(profile_med, Medication), (
                f"Expected Medication object, got {type(profile_med)}"
            )
            assert profile_med.id == med_data["id"], (
                f"Medication id mismatch: expected '{med_data['id']}', got '{profile_med.id}'"
            )
            assert profile_med.name == med_data["name"], (
                f"Medication name mismatch: expected '{med_data['name']}', got '{profile_med.name}'"
            )

    @given(medications_list=_medications_list_strategy(min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_property_9_medication_fields_preserved(self, medications_list):
        """
        Additional property test: All medication fields from patient_profile
        should be preserved in Medication objects.

        **Validates: Requirements 5.5**
        """
        from patient_agent_bench.patient import PatientProfile, Medication, MedicationStatus
        import asyncio
        import json
        from unittest.mock import AsyncMock, MagicMock

        # Create PatientProfile with the generated medications
        patient_profile = create_test_patient_profile(
            first_name="Test",
            last_name="Patient",
            city="Boston",
            state="MA",
            medications=medications_list,
        )

        # Create mock LLM client
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "offices": [{"id": "o1", "name": "Clinic", "address": "123 St",
                        "city": "Boston", "state": "MA", "zip_code": "02101",
                        "phone": "555-0000", "hours": "9-5"}],
            "doctors": [{"id": "d1", "name": "Dr. X", "specialty": "Primary Care",
                        "credentials": "MD", "office_id": "o1"}],
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        async def run_test():
            sandbox = HealthcareSandbox()
            await initialize_sandbox(
                sandbox=sandbox,
                patient_profile=patient_profile,
                llm_client=mock_llm,
                current_datetime="Wednesday, February 11, 2026 at 10:30 AM",
            )
            return sandbox

        sandbox = asyncio.run(run_test())
        profile_medications = sandbox.patient_profile.medications

        # Verify all fields are preserved
        for i, med_data in enumerate(medications_list):
            profile_med = profile_medications[i]

            # Check all required fields
            assert profile_med.dosage == med_data["dosage"]
            assert profile_med.frequency == med_data["frequency"]
            assert profile_med.status == MedicationStatus(med_data["status"])
            assert profile_med.prescribed_date == med_data["prescribed_date"]
            assert profile_med.pharmacy == med_data["pharmacy"]
            assert profile_med.refills_remaining == med_data["refills_remaining"]
            assert profile_med.last_filled == med_data["last_filled"]
            assert profile_med.reason == med_data["reason"]


# =============================================================================
# Unit Tests for Empty Medications Handling (Task 8.5)
# =============================================================================


class TestEmptyMedicationsHandling:
    """
    Unit tests for sandbox initialization with empty medications.

    **Validates: Requirements 5.6**

    Requirement 5.6: IF patient_profile does not contain medications,
    THE Sandbox SHALL initialize with an empty medications dictionary.
    """

    @pytest.mark.asyncio
    async def test_sandbox_empty_medications_when_not_in_profile(self):
        """
        Test sandbox initializes with empty medications when patient_profile
        has no medications (empty list).

        **Validates: Requirements 5.6**
        """
        import json
        from unittest.mock import AsyncMock, MagicMock

        # Create PatientProfile with empty medications list
        patient_profile = create_test_patient_profile(
            first_name="John",
            last_name="Doe",
            sex="Male",
            age=45,
            city="Boston",
            state="MA",
            medications=[],  # Explicitly empty
        )

        # Create mock LLM client
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "offices": [{
                "id": "office_001",
                "name": "Test Clinic",
                "address": "123 Main St",
                "city": "Boston",
                "state": "MA",
                "zip_code": "02101",
                "phone": "555-1234",
                "hours": "Mon-Fri 9-5",
            }],
            "doctors": [{
                "id": "doc_001",
                "name": "Dr. Test",
                "specialty": "Primary Care",
                "credentials": "MD",
                "office_id": "office_001",
            }],
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Initialize sandbox
        sandbox = HealthcareSandbox()
        await initialize_sandbox(
            sandbox=sandbox,
            patient_profile=patient_profile,
            llm_client=mock_llm,
            current_datetime="Wednesday, February 11, 2026 at 10:30 AM",
        )

        # Verify medications list is empty
        assert sandbox.patient_profile.medications == []
        assert len(sandbox.patient_profile.medications) == 0

    @pytest.mark.asyncio
    async def test_sandbox_empty_medications_when_field_not_set(self):
        """
        Test sandbox initializes with empty medications when patient_profile
        medications field uses default (empty list).

        **Validates: Requirements 5.6**
        """
        import json
        from unittest.mock import AsyncMock, MagicMock

        # Create PatientProfile without explicitly setting medications
        # (relies on default_factory=list)
        patient_profile = create_test_patient_profile(
            first_name="Jane",
            last_name="Smith",
            sex="Female",
            age=30,
            city="Cambridge",
            state="MA",
            # medications not set - uses default empty list
        )

        # Create mock LLM client
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "offices": [{
                "id": "office_001",
                "name": "Cambridge Health",
                "address": "456 Oak Ave",
                "city": "Cambridge",
                "state": "MA",
                "zip_code": "02139",
                "phone": "555-5678",
                "hours": "Mon-Sat 8-6",
            }],
            "doctors": [{
                "id": "doc_001",
                "name": "Dr. Smith",
                "specialty": "Family Medicine",
                "credentials": "MD",
                "office_id": "office_001",
            }],
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Initialize sandbox
        sandbox = HealthcareSandbox()
        await initialize_sandbox(
            sandbox=sandbox,
            patient_profile=patient_profile,
            llm_client=mock_llm,
            current_datetime="Wednesday, February 11, 2026 at 10:30 AM",
        )

        # Verify medications list is empty
        assert sandbox.patient_profile.medications == []
        assert len(sandbox.patient_profile.medications) == 0

    @pytest.mark.asyncio
    async def test_sandbox_other_data_generated_with_empty_medications(self):
        """
        Test that sandbox generates offices, doctors, and slots correctly
        even when patient_profile has no medications.

        **Validates: Requirements 5.6**
        """
        import json
        from unittest.mock import AsyncMock, MagicMock

        # Create PatientProfile with empty medications
        patient_profile = create_test_patient_profile(
            first_name="Bob",
            last_name="Jones",
            sex="Male",
            age=55,
            city="Boston",
            state="MA",
            medications=[],
        )

        # Create mock LLM client with multiple offices and doctors
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "offices": [
                {
                    "id": "office_001",
                    "name": "Downtown Clinic",
                    "address": "123 Main St",
                    "city": "Boston",
                    "state": "MA",
                    "zip_code": "02101",
                    "phone": "555-1111",
                    "hours": "Mon-Fri 8-5",
                },
                {
                    "id": "office_002",
                    "name": "Suburban Center",
                    "address": "456 Oak Ave",
                    "city": "Boston",
                    "state": "MA",
                    "zip_code": "02102",
                    "phone": "555-2222",
                    "hours": "Mon-Sat 9-6",
                },
            ],
            "doctors": [
                {
                    "id": "doc_001",
                    "name": "Dr. Primary",
                    "specialty": "Primary Care",
                    "credentials": "MD",
                    "office_id": "office_001",
                },
                {
                    "id": "doc_002",
                    "name": "Dr. Specialist",
                    "specialty": "Cardiology",
                    "credentials": "MD",
                    "office_id": "office_002",
                },
            ],
        })
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Initialize sandbox
        sandbox = HealthcareSandbox()
        await initialize_sandbox(
            sandbox=sandbox,
            patient_profile=patient_profile,
            llm_client=mock_llm,
            current_datetime="Wednesday, February 11, 2026 at 10:30 AM",
        )

        # Verify other sandbox data is generated correctly
        assert len(sandbox.offices) == 2
        assert len(sandbox.doctors) == 2
        assert len(sandbox.slots) > 0  # Slots are generated for doctors

        # Verify medications is still empty
        assert sandbox.patient_profile.medications == []
