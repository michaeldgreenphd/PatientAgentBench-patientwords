# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Healthcare Sandbox Module.

This module provides the healthcare simulation environment for PatientAgentBench,
including state management, data models, and LLM-based sandbox generation.

Public exports:
- HealthcareSandbox: Main sandbox class managing healthcare state
- Data models: OfficeLocation, Doctor, AppointmentSlot, BookedAppointment
- Medication, MedicationStatus: Re-exported from patient module for convenience
- create_sandbox_llm: Factory function for sandbox LLM client
- initialize_sandbox: Main entry point for sandbox initialization

Note: PatientProfile, Medication, MedicationStatus are in patient_agent_bench.patient
"""

from patient_agent_bench.sandbox.sandbox import (
    AppointmentSlot,
    BookedAppointment,
    Doctor,
    HealthcareSandbox,
    OfficeLocation,
)
from patient_agent_bench.sandbox.generator import create_sandbox_llm, initialize_sandbox
# Re-export Medication and MedicationStatus from patient module for backward compatibility
from patient_agent_bench.patient import Medication, MedicationStatus

__all__ = [
    "HealthcareSandbox",
    "OfficeLocation",
    "Doctor",
    "AppointmentSlot",
    "BookedAppointment",
    "Medication",
    "MedicationStatus",
    "create_sandbox_llm",
    "initialize_sandbox",
]
