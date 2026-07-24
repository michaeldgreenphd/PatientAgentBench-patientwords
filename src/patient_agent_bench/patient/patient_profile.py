# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Patient Profile model for PatientAgentBench.

Contains dataclasses for patient profile components with SCHEMA definitions
for LLM prompt generation and JSON serialization.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MedicationStatus(str, Enum):
    """Status of a medication."""

    ACTIVE = "active"
    PAST = "past"
    UNDER_RENEWAL = "under_renewal"


@dataclass
class Medication:
    """A medication in the patient's record."""

    SCHEMA = {
        "id": "string (e.g., med_001)",
        "name": "string (e.g., Metformin)",
        "dosage": "string (e.g., 500mg)",
        "frequency": "string (e.g., twice daily)",
        "status": "string (active|past|under_renewal)",
        "prescribed_date": "string (YYYY-MM-DD)",
        "pharmacy": "string",
        "refills_remaining": "number (0-12)",
        "last_filled": "string (YYYY-MM-DD) or null",
        "reason": "string (condition being treated)",
    }

    id: str = ""
    name: str = ""
    dosage: str = ""
    frequency: str = ""
    status: MedicationStatus = MedicationStatus.ACTIVE
    prescribed_date: str = ""
    pharmacy: str = ""
    refills_remaining: int = 0
    last_filled: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "dosage": self.dosage,
            "frequency": self.frequency,
            "status": self.status.value,
            "prescribed_date": self.prescribed_date,
            "pharmacy": self.pharmacy,
            "refills_remaining": self.refills_remaining,
            "last_filled": self.last_filled,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Medication":
        """Create from dictionary."""
        status = data.get("status", "active")
        if isinstance(status, str):
            try:
                status = MedicationStatus(status)
            except ValueError:
                status = MedicationStatus.ACTIVE
        
        # Handle None values for optional fields
        refills = data.get("refills_remaining")
        refills_remaining = int(refills) if refills is not None else 0
        
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            dosage=str(data.get("dosage", "")),
            frequency=str(data.get("frequency", "")),
            status=status,
            prescribed_date=str(data.get("prescribed_date", "")),
            pharmacy=str(data.get("pharmacy", "")),
            refills_remaining=refills_remaining,
            last_filled=data.get("last_filled"),
            reason=data.get("reason"),
        )


@dataclass
class AccountInfo:
    """Patient account information."""

    SCHEMA = {
        "age_in_years": "string (numeric)",
        "timezone": "string (e.g., America/New_York)",
    }

    age_in_years: str = ""
    timezone: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {"age_in_years": self.age_in_years, "timezone": self.timezone}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AccountInfo":
        """Create from dictionary."""
        return cls(
            age_in_years=str(data.get("age_in_years", "")),
            timezone=str(data.get("timezone", "")),
        )


@dataclass
class PersonalInfo:
    """Patient personal information."""

    SCHEMA = {
        "first_name": "string",
        "last_name": "string",
        "preferred_name": "string (optional)",
        "dob": "string (YYYY-MM-DD)",
        "sex": "string (male/female)",
        "gender": "string",
        "pronouns": "string (e.g., he/him, she/her, they/them)",
        "email": "string (optional)",
    }

    first_name: str = ""
    last_name: str = ""
    preferred_name: str = ""
    dob: str = ""
    sex: str = ""
    gender: str = ""
    pronouns: str = ""
    email: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "first_name": self.first_name,
            "last_name": self.last_name,
            "preferred_name": self.preferred_name,
            "dob": self.dob,
            "sex": self.sex,
            "gender": self.gender,
            "pronouns": self.pronouns,
            "email": self.email,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalInfo":
        """Create from dictionary."""
        return cls(
            first_name=str(data.get("first_name", "")),
            last_name=str(data.get("last_name", "")),
            preferred_name=str(data.get("preferred_name", "")),
            dob=str(data.get("dob", "")),
            sex=str(data.get("sex", "")),
            gender=str(data.get("gender", "")),
            pronouns=str(data.get("pronouns", "")),
            email=str(data.get("email", "")),
        )


@dataclass
class Address:
    """Patient address."""

    SCHEMA = {
        "address1": "string",
        "address2": "string (optional)",
        "city": "string",
        "state": "string (2-letter code)",
        "zip": "string (5 digits)",
        "is_preferred": "string (true/false)",
    }

    address1: str = ""
    address2: str = ""
    city: str = ""
    state: str = ""
    zip: str = ""
    is_preferred: str = "true"

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "address1": self.address1,
            "address2": self.address2,
            "city": self.city,
            "state": self.state,
            "zip": self.zip,
            "is_preferred": self.is_preferred,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Address":
        """Create from dictionary."""
        return cls(
            address1=str(data.get("address1", "")),
            address2=str(data.get("address2", "")),
            city=str(data.get("city", "")),
            state=str(data.get("state", "")),
            zip=str(data.get("zip", "")),
            is_preferred=str(data.get("is_preferred", "true")),
        )


@dataclass
class PhoneNumber:
    """Patient phone number."""

    SCHEMA = {
        "number": "string (10 digits)",
        "kind": "string (mobile/home/work)",
        "is_preferred": "string (true/false)",
        "extension": "string (optional)",
    }

    number: str = ""
    kind: str = "mobile"
    is_preferred: str = "true"
    extension: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "number": self.number,
            "kind": self.kind,
            "is_preferred": self.is_preferred,
            "extension": self.extension,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhoneNumber":
        """Create from dictionary."""
        return cls(
            number=str(data.get("number", "")),
            kind=str(data.get("kind", "mobile")),
            is_preferred=str(data.get("is_preferred", "true")),
            extension=str(data.get("extension", "")),
        )


@dataclass
class EmergencyContact:
    """Patient emergency contact."""

    SCHEMA = {
        "first_name": "string",
        "last_name": "string",
        "relationship": "string (e.g., spouse, parent, sibling)",
        "phone": "string (10 digits)",
    }

    first_name: str = ""
    last_name: str = ""
    relationship: str = ""
    phone: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "first_name": self.first_name,
            "last_name": self.last_name,
            "relationship": self.relationship,
            "phone": self.phone,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmergencyContact":
        """Create from dictionary."""
        return cls(
            first_name=str(data.get("first_name", "")),
            last_name=str(data.get("last_name", "")),
            relationship=str(data.get("relationship", "")),
            phone=str(data.get("phone", "")),
        )


@dataclass
class Insurance:
    """Patient insurance information."""

    SCHEMA = {
        "name": "string (insurance provider name)",
        "plan_type": "string (e.g., PPO, HMO, EPO)",
    }

    name: str = ""
    plan_type: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {"name": self.name, "plan_type": self.plan_type}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Insurance":
        """Create from dictionary."""
        return cls(
            name=str(data.get("name", "")),
            plan_type=str(data.get("plan_type", "")),
        )


@dataclass
class Pharmacy:
    """Patient preferred pharmacy."""

    SCHEMA = {
        "name": "string",
        "address": "string",
        "phone": "string",
    }

    name: str = ""
    address: str = ""
    phone: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {"name": self.name, "address": self.address, "phone": self.phone}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pharmacy":
        """Create from dictionary."""
        return cls(
            name=str(data.get("name", "")),
            address=str(data.get("address", "")),
            phone=str(data.get("phone", "")),
        )


@dataclass
class CareTeamMember:
    """A member of the patient's care team."""

    SCHEMA = {
        "role": "string (pcp|specialist|nurse)",
        "id": "string",
        "name": "string",
    }

    role: str = ""
    id: str = ""
    name: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {"role": self.role, "id": self.id, "name": self.name}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CareTeamMember":
        """Create from dictionary."""
        return cls(
            role=str(data.get("role", "")),
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
        )



@dataclass
class PatientProfile:
    """
    Patient profile data from benchmark entry.

    Contains typed nested structures for patient information.
    The care_team list is updated by sandbox when a PCP is assigned.
    """

    account_info: AccountInfo = field(default_factory=AccountInfo)
    personal_info: PersonalInfo = field(default_factory=PersonalInfo)
    address: Address = field(default_factory=Address)
    phone: PhoneNumber = field(default_factory=PhoneNumber)
    emergency_contact: EmergencyContact = field(default_factory=EmergencyContact)
    insurance: Insurance = field(default_factory=Insurance)
    pharmacy: Pharmacy = field(default_factory=Pharmacy)
    care_team: List[CareTeamMember] = field(default_factory=list)
    medications: List[Medication] = field(default_factory=list)

    # Class-level schema combining all nested schemas
    SCHEMA = {
        "account_info": AccountInfo.SCHEMA,
        "personal_info": PersonalInfo.SCHEMA,
        "addresses": {"address": Address.SCHEMA},
        "phone_numbers": {"phone": PhoneNumber.SCHEMA},
        "emergency_contacts": {"contact": EmergencyContact.SCHEMA},
        "insurances": {"insurance": Insurance.SCHEMA},
        "pharmacies": {"pharmacy": Pharmacy.SCHEMA},
        "medications": [Medication.SCHEMA],
    }

    @classmethod
    def get_schema_json(cls) -> str:
        """Get the SCHEMA as a formatted JSON string for prompts."""
        return json.dumps(cls.SCHEMA, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatientProfile":
        """Create PatientProfile from dictionary."""
        # Parse account_info
        account_info = AccountInfo.from_dict(data.get("account_info", {}))

        # Parse personal_info
        personal_info = PersonalInfo.from_dict(data.get("personal_info", {}))

        # Parse address (nested under addresses.address)
        addresses_data = data.get("addresses", {})
        address_data = addresses_data.get("address", {}) if addresses_data else {}
        address = Address.from_dict(address_data)

        # Parse phone (nested under phone_numbers.phone, can be list or dict)
        phone_numbers_data = data.get("phone_numbers", {})
        if isinstance(phone_numbers_data, dict):
            phone_data = phone_numbers_data.get("phone", {})
            if isinstance(phone_data, list) and phone_data:
                # Find preferred or use first
                for p in phone_data:
                    if p.get("is_preferred") == "true":
                        phone_data = p
                        break
                else:
                    phone_data = phone_data[0]
            phone = PhoneNumber.from_dict(phone_data) if isinstance(phone_data, dict) else PhoneNumber()
        else:
            phone = PhoneNumber()

        # Parse emergency contact (nested under emergency_contacts.contact)
        ec_data = data.get("emergency_contacts", {})
        if isinstance(ec_data, dict):
            contact_data = ec_data.get("contact", {})
            if isinstance(contact_data, list) and contact_data:
                contact_data = contact_data[0]
            emergency_contact = EmergencyContact.from_dict(contact_data) if isinstance(contact_data, dict) else EmergencyContact()
        else:
            emergency_contact = EmergencyContact()

        # Parse insurance (nested under insurances.insurance)
        insurances_data = data.get("insurances", {})
        insurance_data = insurances_data.get("insurance", {}) if isinstance(insurances_data, dict) else {}
        insurance = Insurance.from_dict(insurance_data) if isinstance(insurance_data, dict) else Insurance()

        # Parse pharmacy (nested under pharmacies.pharmacy)
        pharmacies_data = data.get("pharmacies", {})
        pharmacy_data = pharmacies_data.get("pharmacy", {}) if isinstance(pharmacies_data, dict) else {}
        pharmacy = Pharmacy.from_dict(pharmacy_data) if isinstance(pharmacy_data, dict) else Pharmacy()

        # Parse care_team (list of dicts or legacy dict format)
        care_team_data = data.get("care_team", [])
        if isinstance(care_team_data, dict):
            care_team_data = []  # Convert legacy dict format to empty list
        care_team = [CareTeamMember.from_dict(m) for m in care_team_data]

        # Parse medications as Medication objects
        medications_data = data.get("medications", [])
        medications = []
        for med_data in medications_data:
            if isinstance(med_data, dict):
                medications.append(Medication.from_dict(med_data))

        return cls(
            account_info=account_info,
            personal_info=personal_info,
            address=address,
            phone=phone,
            emergency_contact=emergency_contact,
            insurance=insurance,
            pharmacy=pharmacy,
            care_team=care_team,
            medications=medications,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (matches original nested structure for compatibility)."""
        return {
            "account_info": self.account_info.to_dict(),
            "personal_info": self.personal_info.to_dict(),
            "addresses": {"address": self.address.to_dict()},
            "phone_numbers": {"phone": self.phone.to_dict()},
            "emergency_contacts": {"contact": self.emergency_contact.to_dict()},
            "insurances": {"insurance": self.insurance.to_dict()},
            "pharmacies": {"pharmacy": self.pharmacy.to_dict()},
            "care_team": [m.to_dict() for m in self.care_team],
            "medications": [m.to_dict() for m in self.medications],
        }

    def to_xml(self) -> str:
        """Convert patient profile to XML format."""
        return _dict_to_xml(self.to_dict(), root_tag=None)

    # -------------------------------------------------------------------------
    # Convenience accessors for commonly used fields
    # -------------------------------------------------------------------------

    def get_pcp(self) -> Optional[CareTeamMember]:
        """Get PCP from care_team if assigned."""
        for member in self.care_team:
            if member.role == "pcp":
                return member
        return None

    # -------------------------------------------------------------------------
    # Update methods for sandbox tools
    # -------------------------------------------------------------------------

    def update_pcp(self, pcp_id: str, pcp_name: str) -> None:
        """Update PCP assignment in care_team."""
        # Remove existing PCP if any
        self.care_team = [m for m in self.care_team if m.role != "pcp"]
        # Add new PCP
        self.care_team.append(CareTeamMember(role="pcp", id=pcp_id, name=pcp_name))

    def update_pharmacy(
        self,
        name: str,
        address: Optional[str] = None,
        phone: Optional[str] = None,
    ) -> None:
        """Update pharmacy information."""
        self.pharmacy.name = name
        if address:
            self.pharmacy.address = address
        if phone:
            self.pharmacy.phone = phone

    def update_insurance(
        self,
        name: str,
        plan_type: Optional[str] = None,
    ) -> None:
        """Update insurance information."""
        self.insurance.name = name
        if plan_type:
            self.insurance.plan_type = plan_type

    def update_contact(
        self,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        address: Optional[str] = None,
    ) -> None:
        """Update contact information."""
        if phone:
            self.phone.number = phone
        if email:
            self.personal_info.email = email
        if address:
            self.address.address1 = address

    # -------------------------------------------------------------------------
    # Medication methods
    # -------------------------------------------------------------------------

    def get_medications(
        self,
        status: Optional[MedicationStatus] = None,
    ) -> List["Medication"]:
        """
        Get medications with optional status filter.

        Args:
            status: Filter by medication status (active, past, under_renewal)

        Returns:
            List of medications matching criteria
        """
        if status is None:
            return list(self.medications)
        return [m for m in self.medications if m.status == status]

    def get_medication(self, medication_id: str) -> Optional["Medication"]:
        """Get a specific medication by ID."""
        for med in self.medications:
            if med.id == medication_id:
                return med
        return None

    def add_medication(self, medication: "Medication") -> None:
        """Add a new medication to the patient's record."""
        self.medications.append(medication)

    def request_refill(self, medication_id: str, current_date: str) -> bool:
        """
        Request a refill for a medication.

        Updates the medication's last_filled date and decrements refills_remaining.

        Args:
            medication_id: ID of the medication to refill
            current_date: Date string (YYYY-MM-DD) to use as fill date

        Returns:
            True if refill requested successfully, False if not found or no refills
        """
        medication = self.get_medication(medication_id)
        if not medication:
            return False

        # Can only refill active or under_renewal medications
        if medication.status == MedicationStatus.PAST:
            return False

        # Check if refills available
        if medication.refills_remaining <= 0:
            return False

        # Update medication with fill date
        medication.last_filled = current_date
        medication.refills_remaining -= 1
        return True


def _dict_to_xml(data: Any, root_tag: Optional[str] = None, indent: int = 0) -> str:
    """
    Recursively convert a dictionary to XML string.

    Args:
        data: Dictionary, list, or primitive value to convert
        root_tag: Optional root tag name (None for top-level)
        indent: Current indentation level

    Returns:
        XML formatted string
    """
    indent_str = "  " * indent
    lines: List[str] = []

    if isinstance(data, dict):
        for key, value in data.items():
            if value == "" or value is None:
                continue
            if isinstance(value, dict):
                inner = _dict_to_xml(value, root_tag=None, indent=indent + 1)
                if inner.strip():
                    lines.append(f"{indent_str}<{key}>")
                    lines.append(inner)
                    lines.append(f"{indent_str}</{key}>")
            elif isinstance(value, list):
                for item in value:
                    inner = _dict_to_xml(item, root_tag=None, indent=indent + 1)
                    lines.append(f"{indent_str}<{key}>")
                    lines.append(inner)
                    lines.append(f"{indent_str}</{key}>")
            else:
                lines.append(f"{indent_str}<{key}>{_escape_xml(str(value))}</{key}>")
    elif isinstance(data, list):
        for item in data:
            lines.append(_dict_to_xml(item, root_tag=None, indent=indent))
    else:
        if data != "" and data is not None:
            lines.append(f"{indent_str}{_escape_xml(str(data))}")

    result = "\n".join(lines)

    if root_tag:
        return f"<{root_tag}>\n{result}\n</{root_tag}>"
    return result


def _escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
