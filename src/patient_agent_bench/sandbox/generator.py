# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
LLM-based sandbox data generator for Healthcare Sandbox.

Generates realistic healthcare inventory data (offices, doctors, appointment slots)
using an LLM. Medications come from patient_profile during sandbox initialization.

Also provides:
- create_sandbox_llm: Factory function for creating sandbox LLM clients
- initialize_sandbox: Main entry point for sandbox initialization
"""

import json
import random
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Type

from langchain_core.language_models import BaseChatModel

from patient_agent_bench.config import ModelConfig, create_chat_model
from patient_agent_bench.logging_config import get_logger
from patient_agent_bench.patient import PatientProfile
from patient_agent_bench.sandbox.prompts import GENERATION_PROMPT
from patient_agent_bench.utils.llm_content import flatten_response_content
from patient_agent_bench.sandbox.sandbox import (
    AppointmentSlot,
    Doctor,
    HealthcareSandbox,
    OfficeLocation,
)

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Sandbox LLM Creation
# -----------------------------------------------------------------------------


def create_sandbox_llm(model_config: ModelConfig, bedrock_client) -> BaseChatModel:
    """
    Create an LLM client for sandbox generation.

    Args:
        model_config: Model configuration for sandbox LLM (required).
        bedrock_client: boto3 bedrock-runtime client (required for Bedrock models,
                       ignored for OpenAI API models).

    Returns:
        BaseChatModel instance for sandbox generation

    Raises:
        ValueError: If model_config fields are missing or bedrock_client is None for Bedrock models.
    """
    if not model_config.model_id:
        raise ValueError("sandbox_model.model_id is required in config")
    # temperature is optional: some models (e.g. Sonnet 5, Opus 4.8) reject the
    # `temperature` param, so None is legitimate and create_chat_model() omits it.
    if model_config.max_tokens is None:
        raise ValueError("sandbox_model.max_tokens is required in config")
    if bedrock_client is None and model_config.requires_bedrock:
        raise ValueError("bedrock_client is required for Bedrock models")

    return create_chat_model(model_config, bedrock_client)


# -----------------------------------------------------------------------------
# Dynamic Schema Generation
# -----------------------------------------------------------------------------


# Model classes used for LLM generation (order matters for schema output)
# Note: Medication removed - medications now come from patient_profile
GENERATION_MODELS: List[tuple[str, Type]] = [
    ("offices", OfficeLocation),
    ("doctors", Doctor),
]


def _build_schema_from_models() -> str:
    """
    Dynamically build JSON schema string from model class SCHEMA attributes.

    Returns:
        JSON schema string for LLM prompt
    """
    schema_dict = {}
    for key, model_class in GENERATION_MODELS:
        if hasattr(model_class, "SCHEMA"):
            schema_dict[key] = [model_class.SCHEMA]

    return json.dumps(schema_dict, indent=2)



# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _build_generation_prompt(
    city: str,
    state: str,
    current_date: str,
) -> str:
    """Build the LLM prompt for generating sandbox data."""
    schema = _build_schema_from_models()

    return GENERATION_PROMPT.format(
        city=city,
        state=state,
        schema=schema,
        today=current_date,
    )


def _generate_appointment_slots(
    doctors: List[Doctor],
    current_date: str,
) -> List[AppointmentSlot]:
    """
    Generate appointment slots for each doctor for the next 14 days.

    Args:
        doctors: List of doctors to generate slots for
        current_date: Date string (YYYY-MM-DD) to use as start date

    Returns:
        List of AppointmentSlot objects with 40-70% marked as available
    """
    slots = []
    slot_counter = 1

    # Parse current_date
    start_date = date.fromisoformat(current_date)

    # Standard appointment times
    times = ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
             "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00"]

    for doctor in doctors:
        for day_offset in range(14):
            slot_date = start_date + timedelta(days=day_offset)

            # Skip weekends
            if slot_date.weekday() >= 5:
                continue

            # Generate 3-6 slots per doctor per day
            num_slots = random.randint(3, 6)
            day_times = random.sample(times, min(num_slots, len(times)))

            for time in day_times:
                # 40-70% available
                available = random.random() < random.uniform(0.4, 0.7)

                # Mix of in_person and telehealth
                appt_type = random.choice(["in_person", "in_person", "telehealth"])

                slot = AppointmentSlot(
                    id=f"slot_{slot_counter:04d}",
                    doctor_id=doctor.id,
                    office_id=doctor.office_id,
                    date=slot_date.isoformat(),
                    time=time,
                    duration_minutes=30,
                    available=available,
                    appointment_type=appt_type,
                )
                slots.append(slot)
                slot_counter += 1

    return slots


def _parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM response text into JSON.

    Handles common issues like markdown code blocks.

    Args:
        response_text: Raw LLM response

    Returns:
        Parsed JSON dictionary

    Raises:
        json.JSONDecodeError: If response is not valid JSON
    """
    text = response_text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    result: Dict[str, Any] = json.loads(text)
    return result


def _parse_model_list(data: List[Dict[str, Any]], model_class: Type) -> Dict[str, Any]:
    """
    Parse a list of dicts into model instances keyed by id.

    Args:
        data: List of dictionaries from LLM response
        model_class: Class with from_dict() method (e.g., OfficeLocation, Doctor)

    Returns:
        Dictionary mapping id -> model instance
    """
    result = {}
    for item in data:
        instance = model_class.from_dict(item)
        result[instance.id] = instance
    return result


def _assign_pcp(doctors: Dict[str, Doctor]) -> str | None:
    """Randomly assign a PCP from available doctors."""
    if not doctors:
        return None

    primary_care_docs = [d for d in doctors.values() if "primary" in d.specialty.lower()]
    if primary_care_docs and random.random() < 0.8:  # 80% chance of having PCP
        return random.choice(primary_care_docs).id
    if doctors and random.random() < 0.5:  # 50% chance of any doctor as PCP
        return random.choice(list(doctors.values())).id
    return None



# -----------------------------------------------------------------------------
# Main Generation Function
# -----------------------------------------------------------------------------


async def generate_sandbox_data(
    llm_client: Any,
    city: str,
    state: str,
    current_date: str,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Generate sandbox data using LLM.

    Args:
        llm_client: LangChain LLM client (ChatBedrockConverse)
        city: City for office locations
        state: State for office locations
        current_date: Date string (YYYY-MM-DD) for grounding slots
        max_retries: Maximum retry attempts for invalid JSON

    Returns:
        Dictionary with keys: offices, doctors, slots, pcp_id

    Raises:
        ValueError: If LLM fails to generate valid data after max_retries
    """
    prompt = _build_generation_prompt(city=city, state=state, current_date=current_date)

    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            logger.debug(
                "Generating sandbox data (attempt %d/%d)", attempt + 1, max_retries
            )

            # Invoke LLM
            response = await llm_client.ainvoke(prompt)
            # Normalize str-or-list content (thinking/reasoning models return a
            # list of content blocks) to plain text before JSON parsing.
            response_text = flatten_response_content(
                getattr(response, "content", response)
            )

            # Parse JSON response
            data = _parse_llm_response(response_text)

            # Parse into data structures using generic parser
            offices = _parse_model_list(data.get("offices", []), OfficeLocation)
            doctors = _parse_model_list(data.get("doctors", []), Doctor)
            # Note: medications come from patient_profile, not LLM

            # Validate we got minimum data
            if len(offices) < 1:
                raise ValueError("LLM generated no offices")
            if len(doctors) < 1:
                raise ValueError("LLM generated no doctors")

            # Generate appointment slots (not from LLM - too verbose)
            slots_list = _generate_appointment_slots(
                list(doctors.values()),
                current_date=current_date,
            )
            slots = {s.id: s for s in slots_list}

            # Randomly assign PCP (or None)
            pcp_id = _assign_pcp(doctors)

            logger.info(
                "Generated sandbox: %d offices, %d doctors, %d slots",
                len(offices), len(doctors), len(slots)
            )

            return {
                "offices": offices,
                "doctors": doctors,
                "slots": slots,
                "pcp_id": pcp_id,
            }

        except json.JSONDecodeError as e:
            last_error = e
            logger.warning("Invalid JSON from LLM (attempt %d): %s", attempt + 1, e)
        except ValueError as e:
            last_error = e
            logger.warning("Validation error (attempt %d): %s", attempt + 1, e)

    raise ValueError(
        f"Failed to generate sandbox data after {max_retries} attempts: {last_error}"
    )


# -----------------------------------------------------------------------------
# Sandbox Initialization
# -----------------------------------------------------------------------------


async def initialize_sandbox(
    sandbox: HealthcareSandbox,
    patient_profile: PatientProfile,
    llm_client: Any,
    current_datetime: str,
) -> None:
    """
    Initialize sandbox with LLM-generated data.

    Args:
        sandbox: The HealthcareSandbox instance to initialize
        patient_profile: The benchmark patient profile (contains medications)
        llm_client: LangChain LLM client for generation
        current_datetime: Datetime string for temporal grounding (required)

    Raises:
        ValueError: If initialization fails after retries
    """
    if sandbox._initialized:
        logger.warning("Sandbox already initialized, skipping")
        return

    # Store patient profile directly (medications already parsed)
    sandbox.patient_profile = patient_profile

    # Parse current_date from datetime string
    # Format: "Wednesday, February 11, 2026 at 10:30 AM"
    try:
        dt = datetime.strptime(current_datetime, "%A, %B %d, %Y at %I:%M %p")
        current_date = dt.date().isoformat()
    except ValueError:
        # Try ISO format as fallback
        dt = datetime.fromisoformat(current_datetime.replace(" at ", "T"))
        current_date = dt.date().isoformat()

    # Store current_date for use by tools (e.g., medication refills)
    sandbox.current_date = current_date

    # Extract city/state for office generation
    city = patient_profile.address.city or "Boston"
    state = patient_profile.address.state or "MA"

    # Generate sandbox data using LLM (offices, doctors, slots)
    data = await generate_sandbox_data(
        llm_client=llm_client,
        city=city,
        state=state,
        current_date=current_date,
    )

    # Populate sandbox with generated data
    sandbox.offices = data["offices"]
    sandbox.doctors = data["doctors"]
    sandbox.slots = data["slots"]
    sandbox.pcp_id = data["pcp_id"]

    # Update patient profile with PCP info if assigned
    if sandbox.pcp_id and sandbox.pcp_id in sandbox.doctors:
        pcp = sandbox.doctors[sandbox.pcp_id]
        patient_profile.update_pcp(pcp.id, pcp.name)

    sandbox._initialized = True
    logger.info(
        "Sandbox initialized: %d offices, %d doctors, %d slots, %d medications",
        len(sandbox.offices),
        len(sandbox.doctors),
        len(sandbox.slots),
        len(patient_profile.medications),
    )
