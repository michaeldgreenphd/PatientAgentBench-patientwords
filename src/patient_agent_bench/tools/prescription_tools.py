# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Prescription Management Tools for PatientAgentBench.

Provides tools for prescription refills, new prescriptions, and medication management.
These tools interact with the sandbox state to manage medications.
"""

from datetime import date
from typing import List
from uuid import uuid4

from langchain_core.tools import BaseTool, StructuredTool

from patient_agent_bench.sandbox import (
    HealthcareSandbox,
    Medication,
    MedicationStatus,
)


def _list_medications(sandbox: HealthcareSandbox) -> str:
    """
    List the patient's medications grouped by status.

    Returns medications from sandbox state organized by active, under_renewal, and past.
    """
    all_meds = sandbox.get_medications()

    if not all_meds:
        return "No medications on file."

    # Group by status
    active = [m for m in all_meds if m.status == MedicationStatus.ACTIVE]
    under_renewal = [m for m in all_meds if m.status == MedicationStatus.UNDER_RENEWAL]
    past = [m for m in all_meds if m.status == MedicationStatus.PAST]

    lines = ["Your Medications:", ""]

    def format_med(med: Medication, index: int) -> List[str]:
        """Format a single medication for display."""
        med_lines = []
        med_lines.append(f"{index}. {med.name} {med.dosage} - {med.frequency}")
        if med.reason:
            med_lines.append(f"   Prescribed for: {med.reason}")
        if med.last_filled:
            med_lines.append(f"   Last filled: {med.last_filled}")
        med_lines.append(f"   Refills remaining: {med.refills_remaining}")
        med_lines.append(f"   Pharmacy: {med.pharmacy}")
        med_lines.append(f"   Medication ID: {med.id}")
        return med_lines

    idx = 1
    if active:
        lines.append("ACTIVE MEDICATIONS:")
        for med in active:
            lines.extend(format_med(med, idx))
            lines.append("")
            idx += 1

    if under_renewal:
        lines.append("PENDING RENEWAL:")
        for med in under_renewal:
            lines.extend(format_med(med, idx))
            lines.append("")
            idx += 1

    if past:
        lines.append("PAST MEDICATIONS:")
        for med in past:
            lines.extend(format_med(med, idx))
            lines.append("")
            idx += 1

    return "\n".join(lines)


def _request_refill(
    sandbox: HealthcareSandbox,
    medication_name: str,
    urgent: bool = False,
) -> str:
    """
    Request a refill for an existing prescription.

    Validates medication exists and is active or under_renewal, then updates sandbox.
    Uses pharmacy from user profile.
    """
    # Get pharmacy from profile
    profile = sandbox.get_profile()
    pharmacy = profile.pharmacy.name if profile else None

    if not pharmacy:
        return (
            "Error: No pharmacy is configured. Please set your preferred pharmacy "
            "using update_pharmacy before requesting refills."
        )

    # Find medication by name (case-insensitive partial match)
    all_meds = sandbox.get_medications()
    medication = None
    name_lower = medication_name.lower()

    for med in all_meds:
        if name_lower in med.name.lower():
            medication = med
            break

    if not medication:
        return (
            f"Error: Medication '{medication_name}' not found in your records. "
            "Please check the medication name and try again."
        )

    # Check if medication can be refilled
    if medication.status == MedicationStatus.PAST:
        return (
            f"Error: {medication.name} is a past medication and cannot be refilled. "
            "Please request a new prescription if you need this medication."
        )

    # Check refills remaining
    if medication.refills_remaining <= 0:
        return (
            f"Error: {medication.name} has no refills remaining. "
            "Please request a new prescription or contact your provider."
        )

    # Process the refill via sandbox
    success = sandbox.request_refill(medication.id)

    if not success:
        return f"Failed to process refill for {medication.name}. Please try again."

    urgency = "URGENT " if urgent else ""
    ready_time = "4-8 hours" if urgent else "24-48 hours"

    lines = [f"✓ {urgency}Refill request submitted successfully."]
    lines.append("")
    lines.append(f"Medication: {medication.name} {medication.dosage}")
    lines.append(f"Pharmacy: {pharmacy}")
    lines.append(f"Expected pickup time: {ready_time}")
    lines.append(f"Refills remaining after this fill: {medication.refills_remaining}")
    lines.append("")
    lines.append("You will receive a notification when your prescription is ready for pickup.")

    return "\n".join(lines)


def _request_new_prescription(
    sandbox: HealthcareSandbox,
    medication_name: str,
    reason: str,
) -> str:
    """
    Request a new prescription (requires provider review).

    Adds medication to sandbox with under_renewal status.
    Uses pharmacy from user profile.
    """
    # Get user profile for pharmacy info
    profile = sandbox.get_profile()
    pharmacy = profile.pharmacy.name if profile else None

    if not pharmacy:
        return (
            "Error: No pharmacy is configured. Please set your preferred pharmacy "
            "using update_pharmacy before requesting prescriptions."
        )

    # Create new medication with under_renewal status
    med_id = f"med_{uuid4().hex[:8]}"
    new_med = Medication(
        id=med_id,
        name=medication_name,
        dosage="TBD",  # To be determined by provider
        frequency="TBD",
        status=MedicationStatus.UNDER_RENEWAL,
        prescribed_date=date.today().isoformat(),
        pharmacy=pharmacy,
        refills_remaining=0,  # Will be set by provider
        last_filled=None,
        reason=reason,
    )

    sandbox.add_medication(new_med)

    lines = ["✓ New prescription request submitted successfully."]
    lines.append("")
    lines.append(f"Medication: {medication_name}")
    lines.append(f"Reason: {reason}")
    lines.append(f"Pharmacy: {pharmacy}")
    lines.append("")
    lines.append("What happens next:")
    lines.append("- Your provider will review this request (typically 1-2 business days)")
    lines.append("- You will be notified when the prescription is approved")
    lines.append("- Once approved, the prescription will be sent to your pharmacy")

    return "\n".join(lines)


def get_prescription_tools(sandbox: HealthcareSandbox) -> List[BaseTool]:
    """
    Get all prescription management tools configured with the sandbox.

    Args:
        sandbox: The healthcare sandbox instance for state management

    Returns:
        List of prescription-related tools
    """
    list_tool = StructuredTool.from_function(
        func=lambda: _list_medications(sandbox),
        name="list_medications",
        description="""List the patient's medications grouped by status.

Returns:
    List of medications organized by status (active, pending renewal, past) with details
    including medication ID, dosage, frequency, refills remaining, and pharmacy.""",
    )

    refill_tool = StructuredTool.from_function(
        func=lambda medication_name, urgent=False: _request_refill(
            sandbox, medication_name, urgent
        ),
        name="request_refill",
        description="""Request a refill for an existing prescription.

IMPORTANT: Before calling this tool, use list_medications to verify the medication
exists in the patient's records and check refills remaining.

NOTE: A pharmacy must be set in your profile before requesting refills.
Use update_pharmacy to set your preferred pharmacy if not already configured.

Args:
    medication_name: Name of the medication to refill (must match a medication from list_medications)
    urgent: Whether this is an urgent refill request (faster processing)

Returns:
    Confirmation with pharmacy info and expected pickup timeline""",
    )

    new_rx_tool = StructuredTool.from_function(
        func=lambda medication_name, reason: _request_new_prescription(
            sandbox, medication_name, reason
        ),
        name="request_new_prescription",
        description="""Request a new prescription (requires provider review).

NOTE: A pharmacy must be set in your profile before requesting prescriptions.
Use update_pharmacy to set your preferred pharmacy if not already configured.

Args:
    medication_name: Name of the medication being requested
    reason: Reason for the prescription request

Returns:
    Confirmation with provider review timeline and next steps""",
    )

    return [list_tool, refill_tool, new_rx_tool]
