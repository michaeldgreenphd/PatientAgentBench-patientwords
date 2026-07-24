# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Appointment Management Tools for PatientAgentBench.

Provides tools for scheduling and canceling appointments using the healthcare sandbox.
These tools interact with the sandbox state to manage appointment slots.
"""

from datetime import date, timedelta
from typing import List, Optional

from langchain_core.tools import BaseTool, StructuredTool

from patient_agent_bench.sandbox import HealthcareSandbox

# Valid specialty values for filtering
VALID_SPECIALTIES = [
    "Primary Care",
    "Cardiology",
    "Dermatology",
    "Endocrinology",
    "Orthopedics",
    "Psychiatry",
    "OB/GYN",
]


def _list_doctors(
    sandbox: HealthcareSandbox,
    specialty: Optional[str] = None,
) -> str:
    """
    List all doctors in the healthcare system.

    Args:
        sandbox: The healthcare sandbox instance
        specialty: Optional filter by specialty (case-insensitive)

    Returns:
        Formatted list of doctors with name, specialty, credentials,
        and office location
    """
    # Validate specialty if provided
    if specialty:
        specialty_lower = specialty.lower()
        valid_lower = [s.lower() for s in VALID_SPECIALTIES]
        if specialty_lower not in valid_lower:
            valid_list = ", ".join(VALID_SPECIALTIES)
            return f"Invalid specialty '{specialty}'. Valid values: {valid_list}"

    # Get all doctors from sandbox
    doctors = sandbox.get_doctors()

    if not doctors:
        return "No doctors are currently available in the system."

    # Filter by specialty if provided (case-insensitive)
    if specialty:
        specialty_lower = specialty.lower()
        doctors = [d for d in doctors if d.specialty.lower() == specialty_lower]

        if not doctors:
            return f"No doctors found with specialty '{specialty}'."

    # Format output
    lines = ["Available Doctors:", ""]

    for i, doctor in enumerate(doctors, 1):
        office = sandbox.get_office(doctor.office_id)

        lines.append(f"{i}. {doctor.name}, {doctor.credentials}")
        lines.append(f"   Specialty: {doctor.specialty}")
        if office:
            lines.append(f"   Location: {office.name}")
            lines.append(f"   Address: {office.address}, {office.city}, {office.state}")
        lines.append("")

    return "\n".join(lines)


def _get_available_appointments(
    sandbox: HealthcareSandbox,
    start_date: str,
    end_date: str,
    specialty: Optional[str] = None,
    preferred_time: Optional[str] = None,
) -> str:
    """
    Get available appointment slots within a date range.

    Args:
        sandbox: The healthcare sandbox instance
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        specialty: Optional filter by doctor specialty (case-insensitive)
        preferred_time: Optional time-of-day filter. Accepts "morning" (6AM-12PM),
            "afternoon" (12PM-5PM), or "evening" (5PM-9PM).

    Returns:
        Formatted list of available slots with slot_id, date, time,
        doctor info, and office location
    """
    # Enforce max 3-day date range to keep output manageable
    MAX_DATE_RANGE_DAYS = 3
    try:
        parsed_start = date.fromisoformat(start_date)
        parsed_end = date.fromisoformat(end_date)
        if (parsed_end - parsed_start).days > MAX_DATE_RANGE_DAYS:
            parsed_end = parsed_start + timedelta(days=MAX_DATE_RANGE_DAYS)
            end_date = parsed_end.isoformat()
    except ValueError:
        return "Invalid date format. Use YYYY-MM-DD."

    # Validate specialty if provided
    if specialty:
        specialty_lower = specialty.lower()
        valid_lower = [s.lower() for s in VALID_SPECIALTIES]
        if specialty_lower not in valid_lower:
            valid_list = ", ".join(VALID_SPECIALTIES)
            return f"Invalid specialty '{specialty}'. Valid values: {valid_list}"

    # Validate preferred_time if provided
    valid_time_periods = ("morning", "afternoon", "evening")
    if preferred_time and preferred_time.lower() not in valid_time_periods:
        valid_list = ", ".join(valid_time_periods)
        return f"Invalid preferred_time '{preferred_time}'. Valid values: {valid_list}"

    # Get all available slots from sandbox
    all_slots = sandbox.get_available_slots()

    if not all_slots:
        return "No available appointments found for the specified criteria."

    # Filter by date range
    filtered_slots = []
    for slot in all_slots:
        if start_date <= slot.date <= end_date:
            filtered_slots.append(slot)

    # Filter by specialty if provided
    if specialty:
        specialty_lower = specialty.lower()
        specialty_filtered = []
        for slot in filtered_slots:
            doctor = sandbox.get_doctor(slot.doctor_id)
            if doctor and doctor.specialty.lower() == specialty_lower:
                specialty_filtered.append(slot)
        filtered_slots = specialty_filtered

    # Filter by preferred time period if provided
    if preferred_time:
        time_lower = preferred_time.lower()
        time_filtered = []
        for slot in filtered_slots:
            slot_hour = int(slot.time.split(":")[0])
            if time_lower == "morning" and 6 <= slot_hour < 12:
                time_filtered.append(slot)
            elif time_lower == "afternoon" and 12 <= slot_hour < 17:
                time_filtered.append(slot)
            elif time_lower == "evening" and 17 <= slot_hour < 21:
                time_filtered.append(slot)
        filtered_slots = time_filtered

    if not filtered_slots:
        return "No available appointments found for the specified criteria."

    # Sort by date and time
    filtered_slots.sort(key=lambda s: (s.date, s.time))

    # Format output
    lines = ["Available Appointments:", ""]

    for i, slot in enumerate(filtered_slots, 1):
        doctor = sandbox.get_doctor(slot.doctor_id)
        office = sandbox.get_office(slot.office_id)

        lines.append(f"{i}. Slot ID: {slot.id}")
        lines.append(f"   Date: {slot.date}")
        lines.append(f"   Time: {slot.time}")
        lines.append(f"   Type: {slot.appointment_type.replace('_', ' ').title()}")
        lines.append(f"   Duration: {slot.duration_minutes} minutes")
        if doctor:
            lines.append(f"   Provider: {doctor.name}, {doctor.credentials}")
            lines.append(f"   Specialty: {doctor.specialty}")
        if office:
            lines.append(f"   Location: {office.name}")
            lines.append(f"   Address: {office.address}, {office.city}, {office.state}")
        lines.append("")

    return "\n".join(lines)


def _schedule_appointment(
    sandbox: HealthcareSandbox,
    appointment_type: str,
    preferred_date: Optional[str] = None,
    preferred_time: Optional[str] = None,
    provider_name: Optional[str] = None,
    reason: Optional[str] = None,
) -> str:
    """
    Schedule a new appointment for the patient.

    Finds an available slot matching the criteria and books it via the sandbox.
    """
    # Resolve provider if specified
    doctor = None
    doctor_id = None
    if provider_name:
        doctor = sandbox.resolve_doctor(provider_name)
        if doctor:
            doctor_id = doctor.id
        else:
            return f"Error: Could not find provider '{provider_name}'. Please check the name and try again."

    # Normalize appointment type to sandbox format (in_person or telehealth)
    appt_lower = appointment_type.lower().replace("-", "_").replace(" ", "_")
    if appt_lower in ("telehealth", "virtual", "video", "video_call"):
        sandbox_appt_type = "telehealth"
    else:
        # Default to in_person for: in_person, office, primary_care, specialist, urgent_care, follow_up, etc.
        sandbox_appt_type = "in_person"

    # Get available slots with filters
    available_slots = sandbox.get_available_slots(
        doctor_id=doctor_id,
        date=preferred_date,
        appointment_type=sandbox_appt_type,
    )

    if not available_slots:
        # Try without date filter if no slots found
        if preferred_date:
            available_slots = sandbox.get_available_slots(
                doctor_id=doctor_id,
                appointment_type=sandbox_appt_type,
            )
            if not available_slots:
                return "No available appointment slots found matching your criteria. Please try different options."
        else:
            return "No available appointment slots found matching your criteria. Please try different options."

    # Filter by preferred time if specified
    if preferred_time and available_slots:
        time_lower = preferred_time.lower()
        filtered_slots = []

        for slot in available_slots:
            slot_hour = int(slot.time.split(":")[0])

            if time_lower == "morning" and 6 <= slot_hour < 12:
                filtered_slots.append(slot)
            elif time_lower == "afternoon" and 12 <= slot_hour < 17:
                filtered_slots.append(slot)
            elif time_lower == "evening" and 17 <= slot_hour < 21:
                filtered_slots.append(slot)
            elif ":" in preferred_time or preferred_time.replace(":", "").isdigit():
                # Specific time requested - exact match only
                if slot.time == preferred_time:
                    filtered_slots.append(slot)

        if filtered_slots:
            available_slots = filtered_slots
        elif ":" in preferred_time or preferred_time.replace(":", "").isdigit():
            # Specific time was requested but no exact match found
            return (
                f"No available appointment slot at {preferred_time} matching your criteria. "
                "Use get_available_appointments to see available time slots, then retry "
                "with an available time."
            )

    # Sort by date and time to get earliest available
    available_slots.sort(key=lambda s: (s.date, s.time))

    # Book the first available slot
    slot = available_slots[0]
    appointment = sandbox.book_appointment(slot.id, reason=reason)

    if not appointment:
        return "Failed to book the appointment. The slot may no longer be available."

    # Get doctor and office info for confirmation
    booked_doctor = sandbox.get_doctor(slot.doctor_id)
    office = sandbox.get_office(slot.office_id)

    # Format confirmation message
    lines = ["✓ Appointment scheduled successfully.", ""]
    lines.append(f"Appointment ID: {appointment.id}")
    if booked_doctor:
        lines.append(f"Provider: {booked_doctor.name}, {booked_doctor.credentials} ({booked_doctor.specialty})")
    lines.append(f"Date: {appointment.date}")
    lines.append(f"Time: {appointment.time}")
    lines.append(f"Type: {appointment.appointment_type.replace('_', ' ').title()}")
    if office:
        lines.append(f"Location: {office.name}")
        lines.append(f"Address: {office.address}, {office.city}, {office.state} {office.zip_code}")
    if reason:
        lines.append(f"Reason: {reason}")

    # Add patient guidance
    lines.append("")
    lines.append("What to expect:")
    if appointment.appointment_type == "in_person":
        lines.append("- Please arrive 15 minutes early to complete check-in")
        lines.append("- Bring your insurance card and photo ID")
        lines.append("- Bring a list of current medications")
    else:
        lines.append("- You will receive a video call link before your appointment")
        lines.append("- Ensure you have a stable internet connection")
        lines.append("- Find a quiet, private space for your appointment")

    return "\n".join(lines)



def _cancel_appointment(
    sandbox: HealthcareSandbox,
    appointment_id: str,
    reason_for_cancellation: Optional[str] = None,
) -> str:
    """
    Cancel an existing appointment.

    Releases the slot back to available status via the sandbox.
    """
    # Get appointment details before cancelling
    appointments = sandbox.get_booked_appointments()
    appointment = None
    for appt in appointments:
        if appt.id == appointment_id:
            appointment = appt
            break

    if not appointment:
        return f"Error: Appointment '{appointment_id}' not found. Please check the appointment ID and try again."

    # Get doctor info for confirmation
    doctor = sandbox.get_doctor(appointment.doctor_id)

    # Cancel the appointment
    success = sandbox.cancel_appointment(appointment_id)

    if not success:
        return f"Failed to cancel appointment {appointment_id}. Please try again."

    # Format confirmation message
    lines = [f"✓ Appointment {appointment_id} has been cancelled."]
    if doctor:
        lines.append(f"Provider: {doctor.name}")
    lines.append(f"Original Date: {appointment.date}")
    lines.append(f"Original Time: {appointment.time}")
    if reason_for_cancellation:
        lines.append(f"Reason: {reason_for_cancellation}")
    lines.append("")
    lines.append("The time slot is now available for other patients.")
    lines.append("")
    lines.append("Need to rebook?")
    lines.append("- Use get_available_appointments to find new available slots")
    lines.append("- Use schedule_appointment to book a new appointment")

    return "\n".join(lines)


def _list_appointments(
    sandbox: HealthcareSandbox,
    status: Optional[str] = "upcoming",
) -> str:
    """
    List patient's appointments.

    Returns booked appointments from the sandbox state.
    Note: The status parameter is accepted for API compatibility but currently
    all appointments in the sandbox are considered "upcoming" since we don't
    track appointment completion.
    """
    _ = status  # Currently unused - all sandbox appointments are upcoming
    appointments = sandbox.get_booked_appointments()

    if not appointments:
        return "You have no scheduled appointments."

    # Sort by date and time
    appointments.sort(key=lambda a: (a.date, a.time))

    lines = ["Your Scheduled Appointments:", ""]

    for i, appt in enumerate(appointments, 1):
        doctor = sandbox.get_doctor(appt.doctor_id)
        office = sandbox.get_office(appt.office_id)

        lines.append(f"{i}. Appointment ID: {appt.id}")
        if doctor:
            lines.append(f"   Provider Name: {doctor.name}")
            lines.append(f"   Provider Credentials: {doctor.credentials}")
            lines.append(f"   Provider Specialty: {doctor.specialty}")
        lines.append(f"   Date: {appt.date}")
        lines.append(f"   Time: {appt.time}")
        lines.append(f"   Type: {appt.appointment_type.replace('_', ' ').title()}")
        if office:
            lines.append(f"   Location: {office.name}")
        if appt.reason:
            lines.append(f"   Reason: {appt.reason}")
        lines.append("")

    return "\n".join(lines)


def get_appointment_tools(sandbox: HealthcareSandbox) -> List[BaseTool]:
    """
    Get all appointment management tools configured with the sandbox.

    Args:
        sandbox: The healthcare sandbox instance for state management

    Returns:
        List of appointment-related tools
    """
    specialty_list = ", ".join(VALID_SPECIALTIES)

    list_doctors_tool = StructuredTool.from_function(
        func=lambda specialty=None: _list_doctors(sandbox, specialty),
        name="list_doctors",
        description=f"""List all doctors in the healthcare system.

Args:
    specialty: Optional filter by specialty. Valid values: {specialty_list}

Returns:
    List of doctors with name, specialty, credentials, and office location""",
    )

    get_available_appointments_tool = StructuredTool.from_function(
        func=lambda start_date, end_date, specialty=None, preferred_time=None: _get_available_appointments(
            sandbox, start_date, end_date, specialty, preferred_time
        ),
        name="get_available_appointments",
        description=f"""Get available appointment slots within a date range.

Search in short windows (up to 3 days) to keep results manageable. If a range longer than
3 days is provided, it will be automatically clamped to 3 days from start_date. Use specialty
and preferred_time filters to narrow results further.

Args:
    start_date: Start date in YYYY-MM-DD format
    end_date: End date in YYYY-MM-DD format (max 3 days from start_date)
    specialty: Optional filter by doctor specialty. Valid values: {specialty_list}
    preferred_time: Optional time-of-day filter. Valid values: "morning" (6AM-12PM),
        "afternoon" (12PM-5PM), "evening" (5PM-9PM). Only these three values are accepted.

Returns:
    List of available slots. Each slot includes: slot_id, date, time (HH:MM 24-hour),
    appointment_type, duration, provider name/credentials/specialty, and office location.
    Use the exact time from a returned slot when calling schedule_appointment.""",
    )

    schedule_tool = StructuredTool.from_function(
        func=lambda appointment_type, preferred_date=None, preferred_time=None, provider_name=None, reason=None: _schedule_appointment(
            sandbox, appointment_type, preferred_date, preferred_time, provider_name, reason
        ),
        name="schedule_appointment",
        description="""Schedule a new appointment for the patient.

IMPORTANT: Always call get_available_appointments first to find available slots.
Then use the exact time from a returned slot as preferred_time.

Args:
    appointment_type: Type of appointment ("in_person" or "telehealth")
    preferred_date: Date in YYYY-MM-DD format
    preferred_time: Either a time-of-day period ("morning", "afternoon", "evening") which
        books the earliest slot in that window, OR a specific time in HH:MM 24-hour format
        (e.g., "14:00") which requires an exact match to an available slot. If a specific
        time is given and no slot exists at that exact time, scheduling will fail with an
        error asking you to check available slots.
    provider_name: Provider name (partial match supported) or "PCP" for primary care provider
    reason: Reason for the appointment

Returns:
    On success: confirmation with appointment ID, provider, date, time, location.
    On failure: error message explaining why (no matching slot, provider not found, etc.)""",
    )

    cancel_tool = StructuredTool.from_function(
        func=lambda appointment_id, reason_for_cancellation=None: _cancel_appointment(
            sandbox, appointment_id, reason_for_cancellation
        ),
        name="cancel_appointment",
        description="""Cancel an existing appointment.

IMPORTANT: Check current appointments first using list_appointments to get the
appointment_id before canceling.

Args:
    appointment_id: The ID of the appointment to cancel (e.g., "appt_001")
    reason_for_cancellation: Reason for canceling the appointment

Returns:
    Confirmation message for the cancellation with guidance on rebooking if needed""",
    )

    list_tool = StructuredTool.from_function(
        func=lambda status="upcoming": _list_appointments(sandbox, status),
        name="list_appointments",
        description="""List patient's scheduled appointments.

Args:
    status: Filter by status - "upcoming", "past", or "all" (default: "upcoming")

Returns:
    List of appointments. Each entry includes: Appointment ID, Provider Name,
    Provider Credentials, Provider Specialty, Date, Time, Type, Location, and Reason.""",
    )

    return [list_doctors_tool, get_available_appointments_tool, schedule_tool, cancel_tool, list_tool]
