# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Healthcare Sandbox for PatientAgentBench.

Provides a simulated healthcare environment with:
- Office locations with doctors of various specialties
- Appointment slots for the next two weeks
- Patient profile (from benchmark entry, mutable during conversation)
- Medications (stored in patient_profile)

The sandbox is generated at the start of each conversation using LLM
and persists for the duration of that conversation.
"""

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from patient_agent_bench.logging_config import get_logger
from patient_agent_bench.patient import PatientProfile, Medication, MedicationStatus

logger = get_logger(__name__)


@dataclass
class OfficeLocation:
    """A healthcare office location."""

    SCHEMA = {
        "id": "string (e.g., office_001)",
        "name": "string (e.g., Downtown Primary Care)",
        "address": "string (e.g., 123 Main St)",
        "city": "string",
        "state": "string (2-letter code)",
        "zip_code": "string",
        "phone": "string (e.g., 555-123-4567)",
        "hours": "string (e.g., Mon-Fri 8AM-6PM)",
    }

    id: str
    name: str
    address: str
    city: str
    state: str
    zip_code: str
    phone: str
    hours: str  # e.g., "Mon-Fri 8AM-6PM"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "zip_code": self.zip_code,
            "phone": self.phone,
            "hours": self.hours,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OfficeLocation":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            address=data.get("address", ""),
            city=data.get("city", ""),
            state=data.get("state", ""),
            zip_code=data.get("zip_code", ""),
            phone=data.get("phone", ""),
            hours=data.get("hours", ""),
        )


@dataclass
class Doctor:
    """A doctor in the healthcare system."""

    SCHEMA = {
        "id": "string (e.g., doc_001)",
        "name": "string (e.g., Dr. Sarah Johnson)",
        "specialty": "string (Primary Care|Cardiology|Dermatology|Endocrinology|Orthopedics|Psychiatry|OB/GYN)",
        "credentials": "string (MD|DO|NP|PA-C|FNP)",
        "office_id": "string (must match an office id)",
    }

    id: str
    name: str
    specialty: str
    credentials: str  # e.g., "MD", "DO", "NP", "PA-C"
    office_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "specialty": self.specialty,
            "credentials": self.credentials,
            "office_id": self.office_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Doctor":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            specialty=data.get("specialty", ""),
            credentials=data.get("credentials", ""),
            office_id=data.get("office_id", ""),
        )


@dataclass
class AppointmentSlot:
    """An appointment slot."""

    id: str
    doctor_id: str
    office_id: str
    date: str  # YYYY-MM-DD
    time: str  # HH:MM (24-hour)
    duration_minutes: int
    available: bool
    appointment_type: str  # e.g., "in_person", "telehealth"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "doctor_id": self.doctor_id,
            "office_id": self.office_id,
            "date": self.date,
            "time": self.time,
            "duration_minutes": self.duration_minutes,
            "available": self.available,
            "appointment_type": self.appointment_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppointmentSlot":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            doctor_id=data.get("doctor_id", ""),
            office_id=data.get("office_id", ""),
            date=data.get("date", ""),
            time=data.get("time", ""),
            duration_minutes=data.get("duration_minutes", 30),
            available=data.get("available", True),
            appointment_type=data.get("appointment_type", "in_person"),
        )


@dataclass
class BookedAppointment:
    """A booked appointment for the user."""

    id: str
    slot_id: str
    doctor_id: str
    office_id: str
    date: str
    time: str
    appointment_type: str
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "slot_id": self.slot_id,
            "doctor_id": self.doctor_id,
            "office_id": self.office_id,
            "date": self.date,
            "time": self.time,
            "appointment_type": self.appointment_type,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BookedAppointment":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            slot_id=data.get("slot_id", ""),
            doctor_id=data.get("doctor_id", ""),
            office_id=data.get("office_id", ""),
            date=data.get("date", ""),
            time=data.get("time", ""),
            appointment_type=data.get("appointment_type", "in_person"),
            reason=data.get("reason"),
        )


# PatientProfile is imported from patient module and used directly
# Medication and MedicationStatus are also imported from patient module


class HealthcareSandbox:
    """
    Thread-safe healthcare simulation environment.

    Manages healthcare inventory (offices, doctors, appointment slots),
    patient profile, and medications. The sandbox is generated at conversation
    start using LLM and persists throughout with thread-safe access.
    """

    def __init__(self) -> None:
        """Initialize an empty sandbox with threading lock."""
        self._lock = threading.RLock()
        self.offices: Dict[str, OfficeLocation] = {}
        self.doctors: Dict[str, Doctor] = {}
        self.slots: Dict[str, AppointmentSlot] = {}
        self.booked_appointments: Dict[str, BookedAppointment] = {}
        self.patient_profile: Optional[PatientProfile] = None
        self.pcp_id: Optional[str] = None
        self.current_date: str = ""  # YYYY-MM-DD from conversation start
        self._initialized: bool = False

    @property
    def initialized(self) -> bool:
        """Check if sandbox has been initialized."""
        with self._lock:
            return self._initialized

    def get_sandbox(self) -> "HealthcareSandbox":
        """
        Get sandbox instance (for tool access).

        Returns self - this method exists for API consistency with
        tools that need a sandbox reference.
        """
        return self

    # -------------------------------------------------------------------------
    # Office Methods
    # -------------------------------------------------------------------------

    def get_offices(self) -> List[OfficeLocation]:
        """Get all office locations."""
        with self._lock:
            return list(self.offices.values())

    def get_office(self, office_id: str) -> Optional[OfficeLocation]:
        """Get a specific office by ID."""
        with self._lock:
            return self.offices.get(office_id)

    # -------------------------------------------------------------------------
    # Doctor Methods
    # -------------------------------------------------------------------------

    def get_doctors(self) -> List[Doctor]:
        """Get all doctors."""
        with self._lock:
            return list(self.doctors.values())

    def get_doctor(self, doctor_id: str) -> Optional[Doctor]:
        """Get a specific doctor by ID."""
        with self._lock:
            return self.doctors.get(doctor_id)

    def get_doctor_by_name(self, name: str) -> Optional[Doctor]:
        """
        Find a doctor by name (case-insensitive partial match).

        Args:
            name: Doctor name to search for

        Returns:
            First matching doctor or None
        """
        with self._lock:
            name_lower = name.lower()
            for doctor in self.doctors.values():
                if name_lower in doctor.name.lower():
                    return doctor
            return None

    def get_pcp(self) -> Optional[Doctor]:
        """Get the patient's primary care provider."""
        with self._lock:
            if self.pcp_id:
                return self.doctors.get(self.pcp_id)
            return None

    def resolve_doctor(self, name_or_pcp: str) -> Optional[Doctor]:
        """
        Resolve a doctor reference to an actual Doctor object.

        Args:
            name_or_pcp: Either "PCP" to get the patient's PCP, or a doctor name

        Returns:
            The resolved Doctor or None if not found
        """
        with self._lock:
            if name_or_pcp.upper() == "PCP":
                return self.get_pcp()
            return self.get_doctor_by_name(name_or_pcp)

    # -------------------------------------------------------------------------
    # Appointment Slot Methods
    # -------------------------------------------------------------------------

    def get_available_slots(
        self,
        doctor_id: Optional[str] = None,
        date: Optional[str] = None,
        appointment_type: Optional[str] = None,
    ) -> List[AppointmentSlot]:
        """
        Get available appointment slots with optional filtering.

        Args:
            doctor_id: Filter by specific doctor
            date: Filter by specific date (YYYY-MM-DD)
            appointment_type: Filter by type ("in_person" or "telehealth")

        Returns:
            List of available slots matching criteria
        """
        with self._lock:
            slots = []
            for slot in self.slots.values():
                if not slot.available:
                    continue
                if doctor_id and slot.doctor_id != doctor_id:
                    continue
                if date and slot.date != date:
                    continue
                if appointment_type and slot.appointment_type != appointment_type:
                    continue
                slots.append(slot)
            return slots

    def book_appointment(
        self,
        slot_id: str,
        reason: Optional[str] = None,
    ) -> Optional[BookedAppointment]:
        """
        Book an appointment slot.

        Args:
            slot_id: ID of the slot to book
            reason: Optional reason for the appointment

        Returns:
            BookedAppointment if successful, None if slot not found or unavailable
        """
        with self._lock:
            slot = self.slots.get(slot_id)
            if not slot or not slot.available:
                return None

            # Mark slot as unavailable
            slot.available = False

            # Create booked appointment
            appointment_id = f"appt_{len(self.booked_appointments) + 1:03d}"
            appointment = BookedAppointment(
                id=appointment_id,
                slot_id=slot_id,
                doctor_id=slot.doctor_id,
                office_id=slot.office_id,
                date=slot.date,
                time=slot.time,
                appointment_type=slot.appointment_type,
                reason=reason,
            )
            self.booked_appointments[appointment_id] = appointment
            return appointment

    def cancel_appointment(self, appointment_id: str) -> bool:
        """
        Cancel a booked appointment.

        Args:
            appointment_id: ID of the appointment to cancel

        Returns:
            True if cancelled successfully, False if not found
        """
        with self._lock:
            appointment = self.booked_appointments.get(appointment_id)
            if not appointment:
                return False

            # Mark slot as available again
            slot = self.slots.get(appointment.slot_id)
            if slot:
                slot.available = True

            # Remove the booked appointment
            del self.booked_appointments[appointment_id]
            return True

    def get_booked_appointments(self) -> List[BookedAppointment]:
        """Get all booked appointments."""
        with self._lock:
            return list(self.booked_appointments.values())

    # -------------------------------------------------------------------------
    # Medication Methods (delegate to patient_profile)
    # -------------------------------------------------------------------------

    def get_medications(
        self,
        status: Optional[MedicationStatus] = None,
    ) -> List[Medication]:
        """
        Get medications with optional status filter.

        Args:
            status: Filter by medication status (active, past, under_renewal)

        Returns:
            List of medications matching criteria
        """
        with self._lock:
            if not self.patient_profile:
                return []
            return self.patient_profile.get_medications(status)

    def request_refill(self, medication_id: str) -> bool:
        """
        Request a refill for a medication.

        Updates the medication's last_filled date and decrements refills_remaining.

        Args:
            medication_id: ID of the medication to refill

        Returns:
            True if refill requested successfully, False if not found or no refills
        """
        with self._lock:
            if not self.patient_profile:
                return False
            return self.patient_profile.request_refill(medication_id, self.current_date)

    def add_medication(self, medication: Medication) -> None:
        """
        Add a new medication to the patient's record.

        Args:
            medication: The medication to add
        """
        with self._lock:
            if self.patient_profile:
                self.patient_profile.add_medication(medication)

    # -------------------------------------------------------------------------
    # Profile Methods
    # -------------------------------------------------------------------------

    def get_profile(self) -> Optional[PatientProfile]:
        """Get the current patient profile."""
        with self._lock:
            return self.patient_profile

    def update_profile(self, **kwargs: Any) -> bool:
        """
        Update patient profile fields.

        Args:
            **kwargs: Field names and values to update

        Returns:
            True if updated successfully, False if no profile exists
        """
        with self._lock:
            if not self.patient_profile:
                return False

            # Handle special update methods
            if "pcp_id" in kwargs and "pcp_name" in kwargs:
                self.patient_profile.update_pcp(kwargs.pop("pcp_id"), kwargs.pop("pcp_name"))
            if "pharmacy_name" in kwargs:
                self.patient_profile.update_pharmacy(
                    name=kwargs.pop("pharmacy_name"),
                    address=kwargs.pop("pharmacy_address", None),
                    phone=kwargs.pop("pharmacy_phone", None),
                )
            if "insurance_name" in kwargs:
                self.patient_profile.update_insurance(
                    name=kwargs.pop("insurance_name"),
                    plan_type=kwargs.pop("insurance_plan_type", None),
                )
            if "phone" in kwargs or "email" in kwargs or "address" in kwargs:
                self.patient_profile.update_contact(
                    phone=kwargs.pop("phone", None),
                    email=kwargs.pop("email", None),
                    address=kwargs.pop("address", None),
                )
            return True
