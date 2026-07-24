# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Profile Tools for PatientAgentBench.

Provides tools for reading and updating patient profile information.
All tools operate on the sandbox state for the current conversation.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool

if TYPE_CHECKING:
    from patient_agent_bench.sandbox import HealthcareSandbox


def create_get_profile(sandbox: "HealthcareSandbox") -> BaseTool:
    """Create get_profile tool with sandbox reference."""

    def get_profile() -> str:
        """
        Retrieve the patient's current profile information.

        Returns:
            Current profile details including personal info, PCP, insurance, and pharmacy
        """
        profile = sandbox.get_profile()
        if not profile:
            return "Error: Patient profile not available."
        return profile.to_xml()

    return StructuredTool.from_function(
        func=get_profile,
        name="get_profile",
        description=(
            "Retrieve the patient's current profile information including personal info, "
            "contact details, PCP, insurance, pharmacy, and emergency contact."
        ),
    )


def create_update_pcp(sandbox: "HealthcareSandbox") -> BaseTool:
    """Create update_pcp tool with sandbox reference."""

    def update_pcp(
        new_pcp_name: str,
        reason: Optional[str] = None,
    ) -> str:
        """
        Update the patient's Primary Care Provider (PCP).

        Args:
            new_pcp_name: Name of the new PCP (must exist in healthcare inventory)
            reason: Optional reason for changing PCP

        Returns:
            Confirmation of PCP update or error if doctor not found
        """
        # Validate doctor exists in inventory
        doctor = sandbox.get_doctor_by_name(new_pcp_name)
        if not doctor:
            available_doctors = sandbox.get_doctors()
            pcp_doctors = [d for d in available_doctors if d.specialty == "Primary Care"]
            if pcp_doctors:
                names = ", ".join(d.name for d in pcp_doctors[:5])
                return (
                    f"Error: Doctor '{new_pcp_name}' not found in our system. "
                    f"Available primary care providers include: {names}"
                )
            return f"Error: Doctor '{new_pcp_name}' not found in our system."

        # Update sandbox
        sandbox.pcp_id = doctor.id
        sandbox.update_profile(pcp_id=doctor.id, pcp_name=doctor.name)

        msg = f"""✓ Primary Care Provider updated successfully.

New PCP: {doctor.name}, {doctor.credentials}
Specialty: {doctor.specialty}
Effective Date: Immediately"""

        if reason:
            msg += f"\nReason: {reason}"

        # Get office info
        office = sandbox.get_office(doctor.office_id)
        if office:
            msg += f"""

Office Location:
  {office.name}
  {office.address}, {office.city}, {office.state} {office.zip_code}
  Phone: {office.phone}
  Hours: {office.hours}"""

        msg += """

Your medical records will be accessible to your new provider. Please schedule an \
introductory appointment with your new PCP at your earliest convenience."""

        return msg

    return StructuredTool.from_function(
        func=update_pcp,
        name="update_pcp",
        description=(
            "Update the patient's Primary Care Provider (PCP). "
            "The new PCP must be a doctor in the healthcare system."
        ),
    )


def create_update_pharmacy(sandbox: "HealthcareSandbox") -> BaseTool:
    """Create update_pharmacy tool with sandbox reference."""

    def update_pharmacy(
        pharmacy_name: str,
        pharmacy_address: Optional[str] = None,
        pharmacy_phone: Optional[str] = None,
    ) -> str:
        """
        Update the patient's preferred pharmacy.

        Args:
            pharmacy_name: Name of the pharmacy
            pharmacy_address: Address of the pharmacy
            pharmacy_phone: Phone number of the pharmacy

        Returns:
            Confirmation of pharmacy update
        """
        # Build update dict
        updates = {"pharmacy_name": pharmacy_name}
        if pharmacy_address:
            updates["pharmacy_address"] = pharmacy_address
        if pharmacy_phone:
            updates["pharmacy_phone"] = pharmacy_phone

        # Update sandbox profile
        sandbox.update_profile(**updates)

        # Build confirmation message
        details = [f"Pharmacy: {pharmacy_name}"]
        if pharmacy_address:
            details.append(f"Address: {pharmacy_address}")
        if pharmacy_phone:
            details.append(f"Phone: {pharmacy_phone}")

        return (
            "✓ Pharmacy information updated successfully.\n\n"
            + "\n".join(details)
            + "\n\nThis pharmacy has been set as your default. "
            "Future prescriptions will be sent to this pharmacy unless otherwise specified."
        )

    return StructuredTool.from_function(
        func=update_pharmacy,
        name="update_pharmacy",
        description="Update the patient's preferred pharmacy for prescriptions.",
    )


def create_update_insurance(sandbox: "HealthcareSandbox") -> BaseTool:
    """Create update_insurance tool with sandbox reference."""

    def update_insurance(
        insurance_name: str,
        plan_type: Optional[str] = None,
    ) -> str:
        """
        Update the patient's insurance information.

        Args:
            insurance_name: Name of the insurance provider
            plan_type: Type of insurance plan (e.g., PPO, HMO)

        Returns:
            Confirmation of insurance update
        """
        # Build update dict
        updates: Dict[str, str] = {"insurance_name": insurance_name}
        if plan_type:
            updates["insurance_plan_type"] = plan_type

        # Update sandbox profile
        sandbox.update_profile(**updates)

        # Build confirmation message
        details = [f"Insurance Provider: {insurance_name}"]
        if plan_type:
            details.append(f"Plan Type: {plan_type}")

        return (
            "✓ Insurance information updated successfully.\n\n"
            + "\n".join(details)
            + "\n\nPlease bring your insurance card to your next appointment for verification. "
            "If you have any copay or coverage questions, please contact your insurance "
            "provider directly."
        )

    return StructuredTool.from_function(
        func=update_insurance,
        name="update_insurance",
        description="Update the patient's insurance information.",
    )


def create_update_contact_info(sandbox: "HealthcareSandbox") -> BaseTool:
    """Create update_contact_info tool with sandbox reference."""

    def update_contact_info(
        phone: Optional[str] = None,
        email: Optional[str] = None,
        address: Optional[str] = None,
    ) -> str:
        """
        Update the patient's contact information.

        Args:
            phone: New phone number
            email: New email address
            address: New mailing address

        Returns:
            Confirmation of contact information update
        """
        if not phone and not email and not address:
            return "No contact information provided to update."

        # Build update dict
        updates = {}
        update_msgs = []
        if phone:
            updates["phone"] = phone
            update_msgs.append(f"Phone: {phone}")
        if email:
            updates["email"] = email
            update_msgs.append(f"Email: {email}")
        if address:
            updates["address"] = address
            update_msgs.append(f"Address: {address}")

        # Update sandbox profile
        sandbox.update_profile(**updates)

        return (
            "✓ Contact information updated successfully.\n\n"
            "Updated Fields:\n"
            + "\n".join(update_msgs)
        )

    return StructuredTool.from_function(
        func=update_contact_info,
        name="update_contact_info",
        description="Update the patient's contact information (phone, email, address).",
    )


def get_profile_tools(sandbox: "HealthcareSandbox") -> List[BaseTool]:
    """
    Get all profile tools configured with sandbox reference.

    Args:
        sandbox: The HealthcareSandbox instance for this conversation

    Returns:
        List of profile-related tools
    """
    return [
        create_get_profile(sandbox),
        create_update_pcp(sandbox),
        create_update_pharmacy(sandbox),
        create_update_insurance(sandbox),
        create_update_contact_info(sandbox),
    ]
