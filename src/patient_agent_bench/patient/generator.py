# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
LLM-based patient profile generator for benchmark seed enrichment.

Generates realistic patient profiles, stories, and queries using an LLM
based on benchmark seed attributes (condition, severity, task, demographics).

Note: This module generates patient profile data that is used by the benchmark
seed generator. The sandbox module generates additional data (care_team)
at conversation start.
"""

import json
from typing import Any, Dict

from patient_agent_bench.benchmark_seed.benchmark_entry import BenchmarkSeed
from patient_agent_bench.logging_config import get_logger
from patient_agent_bench.patient.patient_profile import PatientProfile
from patient_agent_bench.patient.prompts import ENRICHMENT_PROMPT
from patient_agent_bench.utils.llm_content import flatten_response_content

logger = get_logger(__name__)


def _build_enrichment_prompt(seed: BenchmarkSeed) -> str:
    """Build the LLM prompt for generating patient enrichment data.

    Args:
        seed: The benchmark seed with demographic and condition info

    Returns:
        Formatted prompt string for LLM
    """
    # Get schema from PatientProfile and format it for the prompt
    schema_json = PatientProfile.get_schema_json()

    # Escape curly braces in schema JSON so they don't get interpreted
    # as format placeholders in the second .format() call
    schema_json_escaped = schema_json.replace("{", "{{").replace("}", "}}")

    # First fill the schema placeholder, then fill the seed attributes
    # Note: ENRICHMENT_PROMPT uses double braces for seed attributes ({{condition_name}})
    # to allow two-stage formatting
    prompt_with_schema = ENRICHMENT_PROMPT.format(patient_profile_schema=schema_json_escaped)

    # Now fill the seed attributes (single braces after first format)
    return prompt_with_schema.format(
        condition_name=seed.condition_name,
        severity_level=seed.severity_level,
        task_category=seed.task_category,
        task_subcategory=seed.task_subcategory,
        age_group=seed.age_group,
        sex=seed.sex,
        gender=seed.gender,
        care_preference=seed.care_preference,
    )


def _parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM response text into JSON.

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


def _validate_enrichment_data(data: Dict[str, Any]) -> None:
    """Validate that the enrichment data has all required fields.

    Args:
        data: Parsed JSON data from LLM

    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Check top-level required fields (query removed, scenario_complexity added)
    required_top_level = ["patient_profile", "patient_story", "scenario_complexity"]
    for field in required_top_level:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Validate scenario_complexity is one of valid values
    valid_complexities = ["infeasible", "regular", "chronic", "complicated"]
    if data["scenario_complexity"] not in valid_complexities:
        raise ValueError(
            f"Invalid scenario_complexity: {data['scenario_complexity']}. "
            f"Must be one of: {valid_complexities}"
        )

    # Check patient_profile structure
    profile = data["patient_profile"]
    required_profile_fields = [
        "account_info",
        "personal_info",
        "addresses",
        "phone_numbers",
        "emergency_contacts",
        "insurances",
        "medications",
    ]
    for field in required_profile_fields:
        if field not in profile:
            raise ValueError(f"Missing required patient_profile field: {field}")

    # Validate medications is a list
    medications = profile.get("medications", [])
    if not isinstance(medications, list):
        raise ValueError("medications must be a list")

    # Validate account_info has age_in_years
    account_info = profile.get("account_info", {})
    if "age_in_years" not in account_info:
        raise ValueError("Missing age_in_years in account_info")

    # Validate personal_info has required fields
    personal_info = profile.get("personal_info", {})
    required_personal = ["first_name", "last_name", "dob", "sex", "gender"]
    for field in required_personal:
        if field not in personal_info:
            raise ValueError(f"Missing required personal_info field: {field}")

    # Validate patient_story is non-empty string
    if not isinstance(data["patient_story"], str) or not data["patient_story"].strip():
        raise ValueError("patient_story must be a non-empty string")


async def generate_enrichment(
    seed: BenchmarkSeed,
    llm_client: Any,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Generate patient profile and story using LLM in a single call.

    This function takes a benchmark seed with demographic and condition information
    and uses an LLM to generate a complete patient profile, a clinical patient story,
    and scenario complexity classification.

    The generated profile includes:
    - account_info: age_in_years, timezone
    - personal_info: name, dob, sex, gender, pronouns
    - addresses: US address with city/state
    - phone_numbers: 10-digit US phone numbers
    - emergency_contacts: contact information
    - insurances: insurance provider details
    - insurance_status: verification status
    - medications: list of patient medications

    Note: care_team and pharmacies are NOT generated here - they are generated
    by the sandbox at conversation start.

    Args:
        seed: The benchmark seed with demographic and condition info
        llm_client: LangChain LLM client (e.g., ChatBedrockConverse)
        max_retries: Maximum retry attempts for invalid JSON responses

    Returns:
        Dictionary with 'patient_profile', 'patient_story', and 'scenario_complexity' keys

    Raises:
        ValueError: If generation fails after max_retries attempts
    """
    prompt = _build_enrichment_prompt(seed)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            logger.debug(
                "Generating patient enrichment (attempt %d/%d) for condition=%s, task=%s_%s",
                attempt + 1,
                max_retries,
                seed.condition_name,
                seed.task_category,
                seed.task_subcategory,
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

            # Validate the response has all required fields
            _validate_enrichment_data(data)

            logger.info(
                "Generated patient enrichment: %s %s, age=%s, condition=%s",
                data["patient_profile"]["personal_info"].get("first_name", "Unknown"),
                data["patient_profile"]["personal_info"].get("last_name", ""),
                data["patient_profile"]["account_info"].get("age_in_years", "?"),
                seed.condition_name,
            )

            return data

        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(
                "Invalid JSON from LLM (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                e,
            )
        except ValueError as e:
            last_error = e
            logger.warning(
                "Validation error (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                e,
            )

    raise ValueError(
        f"Failed to generate patient enrichment after {max_retries} attempts: {last_error}"
    )
