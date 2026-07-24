# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Patient module for PatientAgentBench.

This module contains patient-related models and generation logic:
- PatientProfile: Data model for patient information with typed nested structures
- Component dataclasses: AccountInfo, PersonalInfo, Address, Medication, etc.
- Patient enrichment generation for benchmark seeds

Exports:
    PatientProfile: Patient profile dataclass
    AccountInfo: Account information dataclass
    PersonalInfo: Personal information dataclass
    Address: Address dataclass
    PhoneNumber: Phone number dataclass
    EmergencyContact: Emergency contact dataclass
    Insurance: Insurance dataclass
    Pharmacy: Pharmacy dataclass
    CareTeamMember: Care team member dataclass
    Medication: Medication dataclass
    MedicationStatus: Medication status enum
    ENRICHMENT_PROMPT: Prompt template for patient generation
"""

from patient_agent_bench.patient.patient_profile import (
    PatientProfile,
    AccountInfo,
    PersonalInfo,
    Address,
    PhoneNumber,
    EmergencyContact,
    Insurance,
    Pharmacy,
    CareTeamMember,
    Medication,
    MedicationStatus,
    _dict_to_xml,
    _escape_xml,
)
from patient_agent_bench.patient.prompts import ENRICHMENT_PROMPT

__all__ = [
    "PatientProfile",
    "AccountInfo",
    "PersonalInfo",
    "Address",
    "PhoneNumber",
    "EmergencyContact",
    "Insurance",
    "Pharmacy",
    "CareTeamMember",
    "Medication",
    "MedicationStatus",
    "_dict_to_xml",
    "_escape_xml",
    "ENRICHMENT_PROMPT",
]
