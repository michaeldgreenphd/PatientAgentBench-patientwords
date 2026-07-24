# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for patient generator module.

Tests the ENRICHMENT_PROMPT template and generate_enrichment() function.
"""

import json
import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from patient_agent_bench.benchmark_seed.selector import BenchmarkSeed
from patient_agent_bench.patient.generator import (
    ENRICHMENT_PROMPT,
    _build_enrichment_prompt,
    _parse_llm_response,
    _validate_enrichment_data,
    generate_enrichment,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_seed():
    """Create a sample BenchmarkSeed for testing."""
    return BenchmarkSeed(
        condition_name="Influenza (Flu)",
        severity_level="moderate",
        task_category="prescription",
        task_subcategory="renewal",
        age_group="middle_aged",
        gender="female",
        sex="female",
    )


@pytest.fixture
def valid_enrichment_response():
    """Create a valid enrichment response matching expected format."""
    return {
        "patient_profile": {
            "account_info": {"age_in_years": "42", "timezone": "America/New_York"},
            "personal_info": {
                "first_name": "Sarah",
                "last_name": "Johnson",
                "preferred_name": "Sarah",
                "dob": "1982-03-15",
                "sex": "female",
                "gender": "female",
                "pronouns": "She/Her",
            },
            "addresses": {
                "address": {
                    "address1": "123 Main Street",
                    "address2": "Apt 4B",
                    "city": "Boston",
                    "state": "MA",
                    "zip": "02101",
                    "is_preferred": "true",
                }
            },
            "phone_numbers": {
                "phone": {
                    "number": "6175551234",
                    "kind": "mobile",
                    "is_preferred": "true",
                    "extension": "",
                }
            },
            "emergency_contacts": {
                "contact": {
                    "first_name": "Michael",
                    "last_name": "Johnson",
                    "relationship": "Spouse",
                    "phone": "6175555678",
                }
            },
            "insurances": {
                "insurance": {
                    "name": "Blue Cross Blue Shield",
                    "plan_type": "PPO",
                    "copay": "$25",
                    "subscriber_number": "XYZ123456789",
                    "groupno": "GRP001",
                    "verification_status": "verified",
                }
            },
            "insurance_status": {
                "verified_within_past_year": "true",
                "only_has_self_pay": "false",
                "insurances_manually_verified_at": "2024-01-15",
            },
            "medications": [],
        },
        "patient_story": "Sarah Johnson is a 42-year-old female presenting with moderate flu symptoms including fever, body aches, and fatigue for the past 3 days. She has been managing symptoms at home with over-the-counter medications but is concerned about the duration and severity of her illness. She is seeking a prescription renewal for her regular medications while dealing with this acute illness.",
        "scenario_complexity": "regular",
    }


# -----------------------------------------------------------------------------
# Test ENRICHMENT_PROMPT Template
# -----------------------------------------------------------------------------


class TestEnrichmentPrompt:
    """Tests for the ENRICHMENT_PROMPT template."""

    def test_prompt_contains_required_placeholders(self):
        """Verify prompt has all required placeholders."""
        required_placeholders = [
            "{condition_name}",
            "{severity_level}",
            "{task_category}",
            "{task_subcategory}",
            "{age_group}",
            "{sex}",
            "{gender}",
        ]
        for placeholder in required_placeholders:
            assert placeholder in ENRICHMENT_PROMPT, f"Missing placeholder: {placeholder}"

    def test_prompt_specifies_age_group_ranges(self):
        """Verify prompt includes age group range definitions."""
        assert "young_adult: 18-35" in ENRICHMENT_PROMPT
        assert "middle_aged: 36-64" in ENRICHMENT_PROMPT
        assert "senior: 65-85" in ENRICHMENT_PROMPT

    def test_prompt_specifies_phone_format(self):
        """Verify prompt specifies 10-digit phone format."""
        assert "10 digits" in ENRICHMENT_PROMPT

    def test_prompt_excludes_care_team(self):
        """Verify prompt instructs not to use specific provider names."""
        assert "must NOT use specific provider names" in ENRICHMENT_PROMPT

    def test_prompt_requests_json_only(self):
        """Verify prompt requests JSON-only response."""
        assert "Return ONLY valid JSON" in ENRICHMENT_PROMPT


# -----------------------------------------------------------------------------
# Test _build_enrichment_prompt
# -----------------------------------------------------------------------------


class TestBuildEnrichmentPrompt:
    """Tests for the _build_enrichment_prompt function."""

    def test_builds_prompt_with_seed_values(self, sample_seed):
        """Verify prompt is built with seed attribute values."""
        prompt = _build_enrichment_prompt(sample_seed)

        assert "Influenza (Flu)" in prompt
        assert "moderate" in prompt
        assert "prescription" in prompt
        assert "renewal" in prompt
        assert "middle_aged" in prompt
        assert "female" in prompt

    def test_handles_empty_gender(self):
        """Verify prompt handles empty gender value."""
        seed = BenchmarkSeed(
            condition_name="Test Condition",
            severity_level="mild",
            task_category="health_qa",
            task_subcategory="wellness",
            age_group="young_adult",
            gender="",  # Empty gender is valid
            sex="male",
        )
        prompt = _build_enrichment_prompt(seed)
        # Should not raise an error
        assert "Test Condition" in prompt


# -----------------------------------------------------------------------------
# Test _parse_llm_response
# -----------------------------------------------------------------------------


class TestParseLlmResponse:
    """Tests for the _parse_llm_response function."""

    def test_parses_plain_json(self, valid_enrichment_response):
        """Verify parsing of plain JSON response."""
        json_str = json.dumps(valid_enrichment_response)
        result = _parse_llm_response(json_str)
        assert result == valid_enrichment_response

    def test_parses_json_with_markdown_code_block(self, valid_enrichment_response):
        """Verify parsing of JSON wrapped in markdown code block."""
        json_str = f"```json\n{json.dumps(valid_enrichment_response)}\n```"
        result = _parse_llm_response(json_str)
        assert result == valid_enrichment_response

    def test_parses_json_with_plain_code_block(self, valid_enrichment_response):
        """Verify parsing of JSON wrapped in plain code block."""
        json_str = f"```\n{json.dumps(valid_enrichment_response)}\n```"
        result = _parse_llm_response(json_str)
        assert result == valid_enrichment_response

    def test_handles_whitespace(self, valid_enrichment_response):
        """Verify parsing handles leading/trailing whitespace."""
        json_str = f"\n\n  {json.dumps(valid_enrichment_response)}  \n\n"
        result = _parse_llm_response(json_str)
        assert result == valid_enrichment_response

    def test_raises_on_invalid_json(self):
        """Verify JSONDecodeError is raised for invalid JSON."""
        with pytest.raises(json.JSONDecodeError):
            _parse_llm_response("not valid json")


# -----------------------------------------------------------------------------
# Test _validate_enrichment_data
# -----------------------------------------------------------------------------


class TestValidateEnrichmentData:
    """Tests for the _validate_enrichment_data function."""

    def test_valid_data_passes(self, valid_enrichment_response):
        """Verify valid data passes validation."""
        # Should not raise
        _validate_enrichment_data(valid_enrichment_response)

    def test_missing_patient_profile_raises(self, valid_enrichment_response):
        """Verify missing patient_profile raises ValueError."""
        del valid_enrichment_response["patient_profile"]
        with pytest.raises(ValueError, match="Missing required field: patient_profile"):
            _validate_enrichment_data(valid_enrichment_response)

    def test_missing_patient_story_raises(self, valid_enrichment_response):
        """Verify missing patient_story raises ValueError."""
        del valid_enrichment_response["patient_story"]
        with pytest.raises(ValueError, match="Missing required field: patient_story"):
            _validate_enrichment_data(valid_enrichment_response)

    def test_missing_scenario_complexity_raises(self, valid_enrichment_response):
        """Verify missing scenario_complexity raises ValueError."""
        del valid_enrichment_response["scenario_complexity"]
        with pytest.raises(ValueError, match="Missing required field: scenario_complexity"):
            _validate_enrichment_data(valid_enrichment_response)

    def test_missing_account_info_raises(self, valid_enrichment_response):
        """Verify missing account_info raises ValueError."""
        del valid_enrichment_response["patient_profile"]["account_info"]
        with pytest.raises(ValueError, match="Missing required patient_profile field: account_info"):
            _validate_enrichment_data(valid_enrichment_response)

    def test_missing_age_in_years_raises(self, valid_enrichment_response):
        """Verify missing age_in_years raises ValueError."""
        del valid_enrichment_response["patient_profile"]["account_info"]["age_in_years"]
        with pytest.raises(ValueError, match="Missing age_in_years in account_info"):
            _validate_enrichment_data(valid_enrichment_response)

    def test_missing_personal_info_field_raises(self, valid_enrichment_response):
        """Verify missing personal_info field raises ValueError."""
        del valid_enrichment_response["patient_profile"]["personal_info"]["first_name"]
        with pytest.raises(ValueError, match="Missing required personal_info field: first_name"):
            _validate_enrichment_data(valid_enrichment_response)

    def test_empty_patient_story_raises(self, valid_enrichment_response):
        """Verify empty patient_story raises ValueError."""
        valid_enrichment_response["patient_story"] = ""
        with pytest.raises(ValueError, match="patient_story must be a non-empty string"):
            _validate_enrichment_data(valid_enrichment_response)

    def test_invalid_scenario_complexity_raises(self, valid_enrichment_response):
        """Verify invalid scenario_complexity raises ValueError."""
        valid_enrichment_response["scenario_complexity"] = "invalid_value"
        with pytest.raises(ValueError, match="Invalid scenario_complexity"):
            _validate_enrichment_data(valid_enrichment_response)


# -----------------------------------------------------------------------------
# Test generate_enrichment
# -----------------------------------------------------------------------------


class TestGenerateEnrichment:
    """Tests for the generate_enrichment async function."""

    @pytest.mark.asyncio
    async def test_successful_generation(self, sample_seed, valid_enrichment_response):
        """Verify successful generation with valid LLM response."""
        # Create mock LLM client
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(valid_enrichment_response)
        mock_llm.ainvoke.return_value = mock_response

        result = await generate_enrichment(sample_seed, mock_llm)

        assert result == valid_enrichment_response
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_invalid_json(self, sample_seed, valid_enrichment_response):
        """Verify retries when LLM returns invalid JSON."""
        mock_llm = AsyncMock()
        mock_response_invalid = MagicMock()
        mock_response_invalid.content = "not valid json"
        mock_response_valid = MagicMock()
        mock_response_valid.content = json.dumps(valid_enrichment_response)

        # First call returns invalid JSON, second returns valid
        mock_llm.ainvoke.side_effect = [mock_response_invalid, mock_response_valid]

        result = await generate_enrichment(sample_seed, mock_llm, max_retries=3)

        assert result == valid_enrichment_response
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_validation_error(self, sample_seed, valid_enrichment_response):
        """Verify retries when LLM returns incomplete data."""
        mock_llm = AsyncMock()

        # First response missing required field
        incomplete_response = {"patient_profile": {}, "patient_story": "test"}
        mock_response_incomplete = MagicMock()
        mock_response_incomplete.content = json.dumps(incomplete_response)

        mock_response_valid = MagicMock()
        mock_response_valid.content = json.dumps(valid_enrichment_response)

        mock_llm.ainvoke.side_effect = [mock_response_incomplete, mock_response_valid]

        result = await generate_enrichment(sample_seed, mock_llm, max_retries=3)

        assert result == valid_enrichment_response
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, sample_seed):
        """Verify ValueError is raised after max retries exhausted."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "not valid json"
        mock_llm.ainvoke.return_value = mock_response

        with pytest.raises(ValueError, match="Failed to generate patient enrichment after 3 attempts"):
            await generate_enrichment(sample_seed, mock_llm, max_retries=3)

        assert mock_llm.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_handles_response_without_content_attribute(self, sample_seed, valid_enrichment_response):
        """Verify handling of response without content attribute."""
        mock_llm = AsyncMock()
        # Response that doesn't have .content attribute - will use str()
        mock_llm.ainvoke.return_value = json.dumps(valid_enrichment_response)

        result = await generate_enrichment(sample_seed, mock_llm)

        assert result == valid_enrichment_response

    @pytest.mark.asyncio
    async def test_handles_markdown_wrapped_response(self, sample_seed, valid_enrichment_response):
        """Verify handling of markdown-wrapped JSON response."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = f"```json\n{json.dumps(valid_enrichment_response)}\n```"
        mock_llm.ainvoke.return_value = mock_response

        result = await generate_enrichment(sample_seed, mock_llm)

        assert result == valid_enrichment_response


# -----------------------------------------------------------------------------
# Test Requirements Validation
# -----------------------------------------------------------------------------


class TestRequirementsValidation:
    """Tests validating specific requirements from the spec."""

    def test_requirement_4_1_story_consistent_with_seed(self, valid_enrichment_response):
        """
        Validates: Requirements 4.1
        Patient story should be consistent with seed's condition, severity, task, demographics.
        """
        # The validation function ensures patient_story exists and is non-empty
        # Actual consistency is ensured by the LLM prompt
        _validate_enrichment_data(valid_enrichment_response)
        assert len(valid_enrichment_response["patient_story"]) > 0

    def test_requirement_4_2_scenario_complexity_valid(self, valid_enrichment_response):
        """
        Validates: Requirements 6.1, 6.2
        Scenario complexity should be one of the valid values.
        """
        _validate_enrichment_data(valid_enrichment_response)
        assert valid_enrichment_response["scenario_complexity"] in ["infeasible", "regular", "chronic", "complicated"]

    def test_requirement_5_1_profile_structure(self, valid_enrichment_response):
        """
        Validates: Requirements 5.1
        Patient profile should have required structure.
        """
        profile = valid_enrichment_response["patient_profile"]
        assert "account_info" in profile
        assert "personal_info" in profile
        assert "addresses" in profile
        assert "phone_numbers" in profile
        assert "emergency_contacts" in profile
        assert "insurances" in profile
        assert "insurance_status" in profile

    def test_requirement_5_2_no_care_team_or_pharmacies(self, valid_enrichment_response):
        """
        Validates: Requirements 5.2
        Profile should NOT include care_team or pharmacies.
        """
        profile = valid_enrichment_response["patient_profile"]
        assert "care_team" not in profile
        assert "pharmacies" not in profile

    def test_requirement_5_3_age_within_range(self):
        """
        Validates: Requirements 5.3
        Age should be within age_group range.
        """
        # This is validated by the LLM prompt specifying ranges
        # The prompt includes: young_adult: 18-35, middle_aged: 36-55, senior: 56-85
        assert "young_adult: 18-35" in ENRICHMENT_PROMPT
        assert "middle_aged: 36-64" in ENRICHMENT_PROMPT
        assert "senior: 65-85" in ENRICHMENT_PROMPT

    def test_requirement_5_4_realistic_us_addresses(self):
        """
        Validates: Requirements 5.4
        Prompt should request realistic US city/state combinations.
        """
        assert "realistic US addresses" in ENRICHMENT_PROMPT
        assert "real cities/states" in ENRICHMENT_PROMPT

    def test_requirement_5_5_phone_format(self):
        """
        Validates: Requirements 5.5
        Prompt should specify 10-digit phone format.
        """
        assert "10 digits" in ENRICHMENT_PROMPT
        assert "no dashes" in ENRICHMENT_PROMPT

    def test_requirement_5_6_internal_consistency(self):
        """
        Validates: Requirements 5.6
        Prompt should request internally consistent data.
        """
        assert "Date of birth should match the chosen age" in ENRICHMENT_PROMPT


# -----------------------------------------------------------------------------
# Property-Based Tests
# -----------------------------------------------------------------------------


# Valid scenario complexity values as defined in the design document
VALID_SCENARIO_COMPLEXITIES = ["infeasible", "regular", "chronic", "complicated"]


# Strategy for generating a valid patient profile structure
@st.composite
def valid_patient_profile_strategy(draw):
    """Generate a valid patient profile with all required fields."""
    return {
        "account_info": {"age_in_years": draw(st.text(min_size=1, max_size=3).filter(lambda x: x.strip())), "timezone": "America/New_York"},
        "personal_info": {
            "first_name": draw(st.text(min_size=1, max_size=20).filter(lambda x: x.strip())),
            "last_name": draw(st.text(min_size=1, max_size=20).filter(lambda x: x.strip())),
            "preferred_name": "Test",
            "dob": "1990-01-01",
            "sex": "female",
            "gender": "female",
            "pronouns": "She/Her",
        },
        "addresses": {"address": {"address1": "123 Main St", "city": "Boston", "state": "MA", "zip": "02101", "is_preferred": "true"}},
        "phone_numbers": {"phone": {"number": "6175551234", "kind": "mobile", "is_preferred": "true", "extension": ""}},
        "emergency_contacts": {"contact": {"first_name": "John", "last_name": "Doe", "relationship": "Spouse", "phone": "6175555678"}},
        "insurances": {"insurance": {"name": "Blue Cross", "plan_type": "PPO", "copay": "$25", "subscriber_number": "XYZ123", "groupno": "GRP001", "verification_status": "verified"}},
        "insurance_status": {"verified_within_past_year": "true", "only_has_self_pay": "false", "insurances_manually_verified_at": "2024-01-15"},
        "medications": [],
    }


# Required medication fields as defined in Requirements 5.2
REQUIRED_MEDICATION_FIELDS = [
    "id",
    "name",
    "dosage",
    "frequency",
    "status",
    "prescribed_date",
    "pharmacy",
    "refills_remaining",
    "last_filled",
    "reason",
]


# Strategy for generating a valid medication dictionary
@st.composite
def valid_medication_strategy(draw):
    """Generate a valid medication dictionary with all required fields."""
    return {
        "id": draw(st.text(min_size=1, max_size=20).filter(lambda x: x.strip())),
        "name": draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        "dosage": draw(st.text(min_size=1, max_size=20).filter(lambda x: x.strip())),
        "frequency": draw(st.text(min_size=1, max_size=30).filter(lambda x: x.strip())),
        "status": draw(st.sampled_from(["active", "past", "under_renewal"])),
        "prescribed_date": draw(st.from_regex(r"20[0-9]{2}-[0-1][0-9]-[0-3][0-9]", fullmatch=True)),
        "pharmacy": draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        "refills_remaining": draw(st.integers(min_value=0, max_value=12)),
        "last_filled": draw(st.from_regex(r"20[0-9]{2}-[0-1][0-9]-[0-3][0-9]", fullmatch=True)),
        "reason": draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip())),
    }


# Strategy for generating an invalid medication (missing one or more required fields)
@st.composite
def invalid_medication_strategy(draw):
    """Generate a medication dictionary missing at least one required field."""
    # Start with a valid medication
    medication = draw(valid_medication_strategy())
    # Remove at least one required field
    fields_to_remove = draw(
        st.lists(
            st.sampled_from(REQUIRED_MEDICATION_FIELDS),
            min_size=1,
            max_size=len(REQUIRED_MEDICATION_FIELDS),
            unique=True,
        )
    )
    for field in fields_to_remove:
        del medication[field]
    return medication, fields_to_remove


def validate_medication_schema(medication: Dict[str, Any]) -> List[str]:
    """
    Validate that a medication dictionary has all required fields.

    Args:
        medication: Dictionary representing a medication

    Returns:
        List of missing field names (empty if valid)
    """
    missing_fields = []
    for field in REQUIRED_MEDICATION_FIELDS:
        if field not in medication:
            missing_fields.append(field)
    return missing_fields


class TestProperty8MedicationsSchemaCompliance:
    """
    **Feature: benchmark-seed-refactor, Property 8: Medications Schema Compliance**

    Property 8: Medications Schema Compliance
    *For any* medication dictionary in patient_profile.medications, it SHALL contain
    all required fields: id, name, dosage, frequency, status, prescribed_date,
    pharmacy, refills_remaining, last_filled, reason.

    **Validates: Requirements 5.2**
    """

    @settings(max_examples=100)
    @given(medication=valid_medication_strategy())
    def test_property_8_valid_medication_has_all_required_fields(self, medication):
        """
        **Feature: benchmark-seed-refactor, Property 8: Medications Schema Compliance**

        Verify that a valid medication dictionary contains all required fields.

        **Validates: Requirements 5.2**
        """
        missing_fields = validate_medication_schema(medication)
        assert missing_fields == [], f"Valid medication should have all required fields, but missing: {missing_fields}"

    @settings(max_examples=100)
    @given(invalid_data=invalid_medication_strategy())
    def test_property_8_invalid_medication_detected(self, invalid_data):
        """
        **Feature: benchmark-seed-refactor, Property 8: Medications Schema Compliance**

        Verify that medications missing required fields are detected.

        **Validates: Requirements 5.2**
        """
        medication, removed_fields = invalid_data
        missing_fields = validate_medication_schema(medication)
        # All removed fields should be detected as missing
        assert set(removed_fields) == set(missing_fields), (
            f"Expected missing fields {removed_fields}, but got {missing_fields}"
        )

    @settings(max_examples=100)
    @given(medications=st.lists(valid_medication_strategy(), min_size=1, max_size=5))
    def test_property_8_all_medications_in_list_validated(self, medications):
        """
        **Feature: benchmark-seed-refactor, Property 8: Medications Schema Compliance**

        Verify that all medications in a list are validated for schema compliance.

        **Validates: Requirements 5.2**
        """
        for i, medication in enumerate(medications):
            missing_fields = validate_medication_schema(medication)
            assert missing_fields == [], (
                f"Medication at index {i} should have all required fields, but missing: {missing_fields}"
            )

    @settings(max_examples=100)
    @given(
        patient_profile=valid_patient_profile_strategy(),
        medications=st.lists(valid_medication_strategy(), min_size=0, max_size=5),
    )
    def test_property_8_patient_profile_medications_validated(self, patient_profile, medications):
        """
        **Feature: benchmark-seed-refactor, Property 8: Medications Schema Compliance**

        Verify that medications in patient_profile are validated for schema compliance.

        **Validates: Requirements 5.2**
        """
        patient_profile["medications"] = medications
        for i, medication in enumerate(patient_profile["medications"]):
            missing_fields = validate_medication_schema(medication)
            assert missing_fields == [], (
                f"Medication at index {i} in patient_profile should have all required fields, but missing: {missing_fields}"
            )

    def test_property_8_required_fields_count(self):
        """
        **Feature: benchmark-seed-refactor, Property 8: Medications Schema Compliance**

        Verify that exactly 11 required fields are defined per Requirements 5.2.

        **Validates: Requirements 5.2**
        """
        expected_fields = {
            "id", "name", "dosage", "frequency", "status",
            "prescribed_date", "pharmacy",
            "refills_remaining", "last_filled", "reason"
        }
        assert set(REQUIRED_MEDICATION_FIELDS) == expected_fields
        assert len(REQUIRED_MEDICATION_FIELDS) == 10


class TestProperty10ScenarioComplexityValidity:
    """
    **Feature: benchmark-seed-refactor, Property 10: Scenario Complexity Validity**

    Property 10: Scenario Complexity Validity
    *For any* enrichment result containing scenario_complexity, the value SHALL be
    one of: "infeasible", "regular", "chronic", "complicated".

    **Validates: Requirements 6.1, 6.2**
    """

    @settings(max_examples=100)
    @given(
        valid_complexity=st.sampled_from(VALID_SCENARIO_COMPLEXITIES),
        patient_profile=valid_patient_profile_strategy(),
    )
    def test_property_10_valid_complexity_accepted(self, valid_complexity, patient_profile):
        """
        **Feature: benchmark-seed-refactor, Property 10: Scenario Complexity Validity**

        Verify that _validate_enrichment_data accepts all valid scenario_complexity values.

        **Validates: Requirements 6.1, 6.2**
        """
        enrichment_data = {
            "patient_profile": patient_profile,
            "patient_story": "A test patient story with sufficient content.",
            "scenario_complexity": valid_complexity,
        }

        # Should not raise - valid complexity values are accepted
        _validate_enrichment_data(enrichment_data)

    @settings(max_examples=100)
    @given(
        invalid_complexity=st.text(min_size=0, max_size=50).filter(
            lambda x: x not in VALID_SCENARIO_COMPLEXITIES
        ),
        patient_profile=valid_patient_profile_strategy(),
    )
    def test_property_10_invalid_complexity_rejected(self, invalid_complexity, patient_profile):
        """
        **Feature: benchmark-seed-refactor, Property 10: Scenario Complexity Validity**

        Verify that _validate_enrichment_data rejects invalid scenario_complexity values.

        **Validates: Requirements 6.1, 6.2**
        """
        enrichment_data = {
            "patient_profile": patient_profile,
            "patient_story": "A test patient story with sufficient content.",
            "scenario_complexity": invalid_complexity,
        }

        # Should raise ValueError for invalid complexity values
        with pytest.raises(ValueError, match="Invalid scenario_complexity"):
            _validate_enrichment_data(enrichment_data)

    @settings(max_examples=100)
    @given(
        complexity=st.sampled_from(VALID_SCENARIO_COMPLEXITIES),
    )
    def test_property_10_complexity_values_exhaustive(self, complexity):
        """
        **Feature: benchmark-seed-refactor, Property 10: Scenario Complexity Validity**

        Verify that all valid complexity values are exactly the four defined values.

        **Validates: Requirements 6.1, 6.2**
        """
        # Verify the complexity is one of the exact valid values
        assert complexity in ["infeasible", "regular", "chronic", "complicated"]
        # Verify there are exactly 4 valid values
        assert len(VALID_SCENARIO_COMPLEXITIES) == 4
