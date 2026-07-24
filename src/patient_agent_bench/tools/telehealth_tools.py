# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Telehealth Tools for PatientAgentBench.

Provides tools for virtual care interactions including messaging PCP
and joining virtual call queues. These tools interact with the sandbox
to resolve doctor references and return confirmation messages.
"""

from typing import List, Optional

from langchain_core.tools import BaseTool, StructuredTool

from patient_agent_bench.sandbox import HealthcareSandbox


def _message_pcp(
    sandbox: HealthcareSandbox,
    reason_for_consultation: str,
    message_body: str,
    symptom_onset: Optional[str] = None,
    symptom_severity: Optional[str] = None,
    current_medications_relevant: Optional[str] = None,
    additional_context: Optional[str] = None,
    urgency: str = "routine",
) -> str:
    """
    Send a secure message to the patient's Primary Care Provider (PCP).

    Args:
        sandbox: The healthcare sandbox instance
        reason_for_consultation: Brief reason for contacting PCP (required)
        message_body: The detailed message content (required)
        symptom_onset: When symptoms started (optional)
        symptom_severity: Severity level - mild, moderate, severe (optional)
        current_medications_relevant: Relevant current medications (optional)
        additional_context: Any additional relevant information (optional)
        urgency: Message urgency - "routine" (24-48h) or "urgent" (4h)

    Returns:
        Confirmation with expected response timeline, or error if no PCP assigned
    """
    # Get PCP from sandbox
    pcp = sandbox.get_pcp()

    if not pcp:
        return (
            "Error: You do not have a Primary Care Provider (PCP) assigned. "
            "Please update your PCP in your profile before sending messages."
        )

    # Determine response time based on urgency
    urgency_lower = urgency.lower()
    if urgency_lower == "urgent":
        response_time = "4 hours"
    else:
        response_time = "24-48 hours"

    # Format confirmation message
    lines = [
        "✓ Message sent successfully to your PCP.",
        "",
        f"To: {pcp.name}, {pcp.credentials} ({pcp.specialty})",
        f"Reason: {reason_for_consultation}",
        f"Urgency: {urgency.title()}",
        "",
        "Message:",
        f"  {message_body}",
    ]

    # Add optional fields if provided
    if symptom_onset:
        lines.append(f"  Symptom onset: {symptom_onset}")
    if symptom_severity:
        lines.append(f"  Severity: {symptom_severity}")
    if current_medications_relevant:
        lines.append(f"  Relevant medications: {current_medications_relevant}")
    if additional_context:
        lines.append(f"  Additional context: {additional_context}")

    lines.extend([
        "",
        f"Expected response time: {response_time}",
        "",
        "You will receive a notification when your PCP responds.",
    ])

    return "\n".join(lines)


def _join_virtual_call_queue(
    sandbox: HealthcareSandbox,
    doctor_name: str,
    reason: str,
) -> str:
    """
    Join a virtual call queue for a video visit.

    Resolves the doctor reference (name or "PCP") and returns a confirmation
    with estimated wait time.
    """
    # Resolve doctor
    doctor = sandbox.resolve_doctor(doctor_name)
    
    if not doctor:
        if doctor_name.upper() == "PCP":
            return (
                "Error: You do not have a Primary Care Provider (PCP) assigned. "
                "Please select a specific doctor or update your PCP in your profile."
            )
        return f"Error: Could not find doctor '{doctor_name}'. Please check the name and try again."

    # Get office info for the doctor
    office = sandbox.get_office(doctor.office_id)

    # Simulated wait time (would be dynamic in real system)
    wait_time = "10-15 minutes"

    # Format confirmation message
    lines = [
        "✓ You have been added to the virtual call queue.",
        "",
        f"Provider: {doctor.name}, {doctor.credentials} ({doctor.specialty})",
    ]
    
    if office:
        lines.append(f"Practice: {office.name}")
    
    lines.extend([
        f"Reason for visit: {reason}",
        "",
        f"Estimated wait time: {wait_time}",
        "",
        "Please stay on this page. You will be connected when the provider is ready.",
        "Make sure your camera and microphone are enabled.",
    ])

    return "\n".join(lines)


def get_telehealth_tools(sandbox: HealthcareSandbox) -> List[BaseTool]:
    """
    Get all telehealth tools configured with the sandbox.

    Args:
        sandbox: The healthcare sandbox instance for state management

    Returns:
        List of telehealth-related tools
    """
    message_pcp_tool = StructuredTool.from_function(
        func=lambda reason_for_consultation, message_body, symptom_onset=None, symptom_severity=None, current_medications_relevant=None, additional_context=None, urgency="routine": _message_pcp(
            sandbox, reason_for_consultation, message_body, symptom_onset, symptom_severity,
            current_medications_relevant, additional_context, urgency
        ),
        name="message_pcp",
        description="""Send a secure message to your Primary Care Provider (PCP).

NOTE: Messaging is restricted to your assigned PCP only. To message other
providers, please schedule an appointment or use the virtual call queue.

Args:
    reason_for_consultation: (Required) Brief reason for contacting your PCP
    message_body: (Required) The detailed message content
    symptom_onset: (Optional) When symptoms started
    symptom_severity: (Optional) Severity level (mild, moderate, severe)
    current_medications_relevant: (Optional) Relevant current medications
    additional_context: (Optional) Any additional relevant information
    urgency: Message urgency - "routine" (24-48h response) or "urgent" (4h response)

Returns:
    Confirmation with expected response timeline""",
    )

    queue_tool = StructuredTool.from_function(
        func=lambda doctor_name, reason: _join_virtual_call_queue(
            sandbox, doctor_name, reason
        ),
        name="join_virtual_call_queue",
        description="""Join a virtual video call queue to speak with a doctor.

Args:
    doctor_name: Name of the doctor to call, or "PCP" to call your Primary Care Provider
    reason: Reason for the virtual visit

Returns:
    Confirmation message with estimated wait time and instructions""",
    )

    return [message_pcp_tool, queue_tool]
