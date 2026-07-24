# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for data models.

Tests PatientProfile, BenchmarkEntry, and utility functions.
"""

import json
import pytest
from pathlib import Path

from patient_agent_bench.benchmark_seed import (
    BenchmarkEntry,
    load_benchmark_entries,
)
from patient_agent_bench.patient import (
    PatientProfile,
    _dict_to_xml,
    _escape_xml,
)


class TestPatientProfile:
    """Tests for PatientProfile dataclass."""

    def test_from_dict_complete(self):
        """Test creating PatientProfile from complete dictionary."""
        data = {
            "account_info": {"age_in_years": "44", "timezone": "America/New_York"},
            "personal_info": {"first_name": "John", "last_name": "Doe", "dob": "1980-01-15"},
            "addresses": {"address": {"city": "Boston", "state": "MA", "zip": "02101"}},
            "phone_numbers": {"phone": {"number": "5551234567", "kind": "mobile"}},
            "emergency_contacts": {"contact": {"first_name": "Jane", "relationship": "spouse"}},
            "insurances": {"insurance": {"name": "Blue Cross", "plan_type": "PPO"}},
            "pharmacies": {"pharmacy": {"name": "CVS Pharmacy", "address": "456 Oak Ave"}},
            "care_team": [{"role": "pcp", "id": "D001", "name": "Dr. Smith"}],
            "medications": [{"name": "Lisinopril", "dosage": "10mg"}],
        }
        profile = PatientProfile.from_dict(data)

        assert profile.account_info.age_in_years == "44"
        assert profile.personal_info.first_name == "John"
        assert profile.address.city == "Boston"
        assert profile.phone.number == "5551234567"
        assert profile.insurance.name == "Blue Cross"
        assert profile.pharmacy.name == "CVS Pharmacy"
        assert len(profile.care_team) == 1
        assert profile.get_pcp().name == "Dr. Smith"

    def test_from_dict_empty(self):
        """Test creating PatientProfile from empty dictionary."""
        profile = PatientProfile.from_dict({})

        # Nested dataclasses should have default empty values
        assert profile.account_info.age_in_years == ""
        assert profile.personal_info.first_name == ""
        assert profile.address.city == ""
        assert profile.care_team == []

    def test_from_dict_partial(self):
        """Test creating PatientProfile from partial dictionary."""
        partial_dict = {
            "account_info": {"age_in_years": "30"},
            "personal_info": {"first_name": "Jane"},
        }
        profile = PatientProfile.from_dict(partial_dict)

        assert profile.account_info.age_in_years == "30"
        assert profile.personal_info.first_name == "Jane"
        assert profile.address.city == ""  # Default empty

    def test_to_xml(self):
        """Test converting PatientProfile to XML."""
        data = {
            "account_info": {"age_in_years": "44"},
            "personal_info": {"first_name": "John", "last_name": "Doe"},
            "addresses": {"address": {"city": "Boston", "state": "MA"}},
        }
        profile = PatientProfile.from_dict(data)
        xml = profile.to_xml()

        assert "<age_in_years>44</age_in_years>" in xml
        assert "<first_name>John</first_name>" in xml
        assert "<city>Boston</city>" in xml

    def test_to_xml_empty_profile(self):
        """Test converting empty PatientProfile to XML."""
        profile = PatientProfile()
        xml = profile.to_xml()

        # Empty profile still has some default values (is_preferred, kind)
        # but should not have actual patient data
        assert "<first_name>" not in xml
        assert "<city>" not in xml
        assert "<age_in_years>" not in xml


class TestBenchmarkEntry:
    """Tests for BenchmarkEntry dataclass."""

    def test_from_dict_complete(self, sample_benchmark_entry_dict):
        """Test creating BenchmarkEntry from complete dictionary."""
        entry = BenchmarkEntry.from_dict(sample_benchmark_entry_dict)
        
        # Backward compatibility: query_id in dict maps to scenario_id
        assert entry.scenario_id == "test_001"
        assert entry.condition_name == "Hypertension"
        assert entry.severity_level == "low"
        assert isinstance(entry.patient_profile, PatientProfile)

    def test_from_dict_minimal(self):
        """Test creating BenchmarkEntry from minimal dictionary."""
        minimal_dict = {
            "query_id": "min_001",  # Legacy field name
            "patient_story": "Test story",
        }
        entry = BenchmarkEntry.from_dict(minimal_dict)
        
        # Backward compatibility: query_id maps to scenario_id
        assert entry.scenario_id == "min_001"
        assert entry.condition_name == ""  # Default empty
        assert entry.has_image == "False"  # Default

    def test_from_dict_with_scenario_id(self):
        """Test creating BenchmarkEntry with new scenario_id field."""
        data = {
            "scenario_id": "scenario_001",
            "patient_story": "Test story",
        }
        entry = BenchmarkEntry.from_dict(data)
        
        assert entry.scenario_id == "scenario_001"

    def test_from_dict_scenario_id_preferred_over_query_id(self):
        """Test that scenario_id is preferred when both are present."""
        data = {
            "scenario_id": "new_id",
            "query_id": "old_id",
            "patient_story": "Test story",
        }
        entry = BenchmarkEntry.from_dict(data)
        
        # scenario_id should be preferred
        assert entry.scenario_id == "new_id"

    def test_from_dict_ignores_removed_fields(self):
        """Test that removed fields are ignored for backward compatibility."""
        data = {
            "scenario_id": "test_001",
            "patient_story": "Test story",
            "condition_mapped_name": "Should be ignored",
            "interaction_type": "Should be ignored",
            "query": "Should be ignored",
        }
        entry = BenchmarkEntry.from_dict(data)
        
        # Entry should be created successfully
        assert entry.scenario_id == "test_001"
        # Removed fields should not be accessible
        assert not hasattr(entry, "condition_mapped_name")
        assert not hasattr(entry, "interaction_type")
        assert not hasattr(entry, "query")

    def test_from_dict_scenario_complexity_default(self):
        """Test that scenario_complexity defaults to empty string."""
        data = {
            "scenario_id": "test_001",
            "patient_story": "Test story",
        }
        entry = BenchmarkEntry.from_dict(data)
        
        assert entry.scenario_complexity == ""

    def test_from_dict_scenario_complexity_parsed(self):
        """Test that scenario_complexity is parsed correctly."""
        data = {
            "scenario_id": "test_001",
            "patient_story": "Test story",
            "scenario_complexity": "chronic",
        }
        entry = BenchmarkEntry.from_dict(data)
        
        assert entry.scenario_complexity == "chronic"

    def test_property_aliases(self, sample_benchmark_entry):
        """Test property aliases work correctly."""
        assert sample_benchmark_entry.id == sample_benchmark_entry.scenario_id
        assert sample_benchmark_entry.scenario == sample_benchmark_entry.patient_story

    def test_metadata_property(self, sample_benchmark_entry):
        """Test metadata property returns correct fields."""
        metadata = sample_benchmark_entry.metadata
        
        assert "condition_name" in metadata
        assert "severity_level" in metadata
        assert "task_type" in metadata
        assert metadata["condition_name"] == "Hypertension"

    def test_patient_profile_xml_property(self, sample_benchmark_entry):
        """Test patient_profile_xml property."""
        xml = sample_benchmark_entry.patient_profile_xml

        assert isinstance(xml, str)
        # XML should contain some patient profile data (structure may vary)
        assert "<" in xml or xml == ""  # Either has XML tags or is empty


class TestXmlUtilities:
    """Tests for XML conversion utilities."""

    def test_escape_xml_special_chars(self):
        """Test escaping special XML characters."""
        assert _escape_xml("&") == "&amp;"
        assert _escape_xml("<") == "&lt;"
        assert _escape_xml(">") == "&gt;"
        assert _escape_xml('"') == "&quot;"
        assert _escape_xml("'") == "&apos;"

    def test_escape_xml_combined(self):
        """Test escaping multiple special characters."""
        text = '<tag attr="value">content & more</tag>'
        escaped = _escape_xml(text)
        
        assert "&lt;" in escaped
        assert "&gt;" in escaped
        assert "&amp;" in escaped
        assert "&quot;" in escaped

    def test_escape_xml_normal_text(self):
        """Test that normal text is unchanged."""
        text = "Hello World 123"
        assert _escape_xml(text) == text

    def test_dict_to_xml_simple(self):
        """Test converting simple dictionary to XML."""
        data = {"name": "John", "age": "30"}
        xml = _dict_to_xml(data)
        
        assert "<name>John</name>" in xml
        assert "<age>30</age>" in xml

    def test_dict_to_xml_nested(self):
        """Test converting nested dictionary to XML."""
        data = {
            "person": {
                "name": "John",
                "address": {"city": "Boston"},
            }
        }
        xml = _dict_to_xml(data)
        
        assert "<person>" in xml
        assert "<name>John</name>" in xml
        assert "<city>Boston</city>" in xml

    def test_dict_to_xml_list(self):
        """Test converting dictionary with list to XML."""
        data = {
            "items": ["apple", "banana", "cherry"]
        }
        xml = _dict_to_xml(data)
        
        # Each item should be wrapped in <items> tag
        assert xml.count("<items>") == 3

    def test_dict_to_xml_empty_values(self):
        """Test that empty values are skipped."""
        data = {"name": "John", "empty": "", "none": None}
        xml = _dict_to_xml(data)
        
        assert "<name>John</name>" in xml
        assert "<empty>" not in xml
        assert "<none>" not in xml

    def test_dict_to_xml_with_root_tag(self):
        """Test XML with root tag."""
        data = {"name": "John"}
        xml = _dict_to_xml(data, root_tag="person")
        
        assert xml.startswith("<person>")
        assert xml.endswith("</person>")


class TestLoadBenchmarkEntries:
    """Tests for loading benchmark entries from file."""

    def test_load_single_entry(self, temp_benchmark_file):
        """Test loading a single benchmark entry."""
        entries = load_benchmark_entries(str(temp_benchmark_file))
        
        assert len(entries) == 1
        # Backward compatibility: query_id in file maps to scenario_id
        assert entries[0].scenario_id == "test_001"

    def test_load_multiple_entries(self, tmp_path, sample_benchmark_entry_dict):
        """Test loading multiple benchmark entries."""
        # Create file with multiple entries (using legacy query_id field)
        entries_data = [
            sample_benchmark_entry_dict,
            {**sample_benchmark_entry_dict, "query_id": "test_002"},
            {**sample_benchmark_entry_dict, "query_id": "test_003"},
        ]
        file_path = tmp_path / "multi_benchmark.json"
        file_path.write_text(json.dumps(entries_data))
        
        entries = load_benchmark_entries(str(file_path))
        
        assert len(entries) == 3
        # Backward compatibility: query_id maps to scenario_id
        assert entries[0].scenario_id == "test_001"
        assert entries[1].scenario_id == "test_002"
        assert entries[2].scenario_id == "test_003"

    def test_load_empty_file(self, tmp_path):
        """Test loading empty benchmark file."""
        file_path = tmp_path / "empty_benchmark.json"
        file_path.write_text("[]")
        
        entries = load_benchmark_entries(str(file_path))
        assert entries == []

    def test_load_file_not_found(self, tmp_path):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_benchmark_entries(str(tmp_path / "nonexistent.json"))


# =============================================================================
# Property-Based Tests for BenchmarkEntry
# =============================================================================

from hypothesis import given, settings, strategies as st, HealthCheck


# =============================================================================
# Hypothesis Strategies for BenchmarkEntry
# =============================================================================


@st.composite
def patient_profile_dict_strategy(draw):
    """Generate a valid patient profile dictionary."""
    return {
        "account_info": {"patient_id": draw(st.text(min_size=1, max_size=20))},
        "personal_info": {
            "first_name": draw(st.text(min_size=1, max_size=20)),
            "last_name": draw(st.text(min_size=1, max_size=20)),
        },
        "addresses": {},
        "phone_numbers": {},
        "emergency_contacts": {},
        "insurances": {},
        "care_team": {},
        "pharmacies": {},
        "insurance_status": {},
        "medications": [],
    }


@st.composite
def benchmark_entry_dict_with_legacy_fields(draw):
    """
    Generate a dictionary with required fields and various combinations of legacy fields.

    Legacy fields: condition_mapped_name, interaction_type, query, query_id
    """
    # Required fields for a valid BenchmarkEntry
    base_dict = {
        "scenario_id": draw(st.text(min_size=1, max_size=50)),
        "patient_story": draw(st.text(min_size=1, max_size=500)),
        "patient_profile": draw(patient_profile_dict_strategy()),
        "condition_name": draw(st.text(max_size=50)),
        "preferred_care_option": draw(st.text(max_size=50)),
        "severity_level": draw(st.text(max_size=20)),
        "task_type": draw(st.text(max_size=50)),
        "has_image": draw(st.sampled_from(["True", "False"])),
        "scenario_complexity": draw(st.sampled_from(["", "regular", "chronic", "complicated"])),
    }

    # Optionally add legacy fields
    if draw(st.booleans()):
        base_dict["condition_mapped_name"] = draw(st.text(max_size=100))
    if draw(st.booleans()):
        base_dict["interaction_type"] = draw(st.text(max_size=50))
    if draw(st.booleans()):
        base_dict["query"] = draw(st.text(max_size=500))
    if draw(st.booleans()):
        base_dict["query_id"] = draw(st.text(max_size=50))

    return base_dict


@st.composite
def benchmark_entry_dict_with_all_legacy_fields(draw):
    """
    Generate a dictionary that always includes ALL legacy fields.

    This ensures we test the case where all legacy fields are present.
    """
    return {
        "scenario_id": draw(st.text(min_size=1, max_size=50)),
        "patient_story": draw(st.text(min_size=1, max_size=500)),
        "patient_profile": draw(patient_profile_dict_strategy()),
        "condition_name": draw(st.text(max_size=50)),
        "preferred_care_option": draw(st.text(max_size=50)),
        "severity_level": draw(st.text(max_size=20)),
        "task_type": draw(st.text(max_size=50)),
        "has_image": draw(st.sampled_from(["True", "False"])),
        "scenario_complexity": draw(st.sampled_from(["", "regular", "chronic", "complicated"])),
        # All legacy fields
        "condition_mapped_name": draw(st.text(max_size=100)),
        "interaction_type": draw(st.text(max_size=50)),
        "query": draw(st.text(max_size=500)),
        "query_id": draw(st.text(max_size=50)),
    }


# =============================================================================
# Property 4: Backward Compatibility for Legacy Fields
# =============================================================================


class TestBackwardCompatibilityProperty:
    """
    **Feature: benchmark-seed-refactor, Property 4: Backward Compatibility for Legacy Fields**

    *For any* dictionary containing legacy fields (condition_mapped_name, interaction_type,
    query, query_id), BenchmarkEntry.from_dict() SHALL successfully create a BenchmarkEntry
    without raising an exception, and the legacy fields SHALL be ignored.

    **Validates: Requirements 2.5, 3.10, 4.6**
    """

    @given(data=benchmark_entry_dict_with_legacy_fields())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_from_dict_succeeds_with_legacy_fields(self, data):
        """
        Property 4: from_dict() succeeds with any combination of legacy fields.

        **Feature: benchmark-seed-refactor, Property 4: Backward Compatibility for Legacy Fields**
        **Validates: Requirements 2.5, 3.10, 4.6**
        """
        # from_dict() should not raise an exception
        entry = BenchmarkEntry.from_dict(data)

        # Entry should be created successfully
        assert entry is not None
        assert isinstance(entry, BenchmarkEntry)

        # Legacy fields should NOT be present as attributes
        assert not hasattr(entry, "condition_mapped_name")
        assert not hasattr(entry, "interaction_type")
        assert not hasattr(entry, "query")

        # scenario_id should be set (from scenario_id or query_id)
        assert isinstance(entry.scenario_id, str)

    @given(data=benchmark_entry_dict_with_all_legacy_fields())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_all_legacy_fields_ignored(self, data):
        """
        Property 4: All legacy fields are ignored when present.

        **Feature: benchmark-seed-refactor, Property 4: Backward Compatibility for Legacy Fields**
        **Validates: Requirements 2.5, 3.10, 4.6**
        """
        # Capture the legacy field values before parsing
        legacy_condition_mapped = data.get("condition_mapped_name")
        legacy_interaction_type = data.get("interaction_type")
        legacy_query = data.get("query")

        # from_dict() should not raise an exception
        entry = BenchmarkEntry.from_dict(data)

        # Entry should be created successfully
        assert entry is not None
        assert isinstance(entry, BenchmarkEntry)

        # Legacy fields should NOT be accessible on the entry
        assert not hasattr(entry, "condition_mapped_name")
        assert not hasattr(entry, "interaction_type")
        assert not hasattr(entry, "query")

        # Verify the entry's metadata doesn't contain legacy fields
        metadata = entry.metadata
        assert "condition_mapped_name" not in metadata
        assert "interaction_type" not in metadata
        assert "query" not in metadata

    @given(
        scenario_id=st.text(min_size=1, max_size=50),
        query_id=st.text(min_size=1, max_size=50),
        condition_mapped_name=st.text(max_size=100),
        interaction_type=st.text(max_size=50),
        query=st.text(max_size=500),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_legacy_fields_do_not_affect_entry(
        self, scenario_id, query_id, condition_mapped_name, interaction_type, query
    ):
        """
        Property 4: Legacy field values do not affect the created entry.

        **Feature: benchmark-seed-refactor, Property 4: Backward Compatibility for Legacy Fields**
        **Validates: Requirements 2.5, 3.10, 4.6**
        """
        data = {
            "scenario_id": scenario_id,
            "patient_story": "Test patient story",
            "patient_profile": {},
            # Legacy fields
            "query_id": query_id,
            "condition_mapped_name": condition_mapped_name,
            "interaction_type": interaction_type,
            "query": query,
        }

        # from_dict() should not raise an exception
        entry = BenchmarkEntry.from_dict(data)

        # Entry should be created successfully
        assert entry is not None

        # scenario_id should be preferred over query_id
        assert entry.scenario_id == scenario_id

        # Legacy fields should not be accessible
        assert not hasattr(entry, "condition_mapped_name")
        assert not hasattr(entry, "interaction_type")
        assert not hasattr(entry, "query")

    @given(
        query_id=st.text(min_size=1, max_size=50),
        condition_mapped_name=st.text(max_size=100),
        interaction_type=st.text(max_size=50),
        query=st.text(max_size=500),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_query_id_fallback_with_legacy_fields(
        self, query_id, condition_mapped_name, interaction_type, query
    ):
        """
        Property 4: query_id is used as fallback when scenario_id is absent.

        **Feature: benchmark-seed-refactor, Property 4: Backward Compatibility for Legacy Fields**
        **Validates: Requirements 2.5, 3.10, 4.6**
        """
        data = {
            # No scenario_id - should fall back to query_id
            "query_id": query_id,
            "patient_story": "Test patient story",
            "patient_profile": {},
            # Other legacy fields
            "condition_mapped_name": condition_mapped_name,
            "interaction_type": interaction_type,
            "query": query,
        }

        # from_dict() should not raise an exception
        entry = BenchmarkEntry.from_dict(data)

        # Entry should be created successfully
        assert entry is not None

        # query_id should be used as scenario_id when scenario_id is absent
        assert entry.scenario_id == query_id

        # Legacy fields should not be accessible
        assert not hasattr(entry, "condition_mapped_name")
        assert not hasattr(entry, "interaction_type")
        assert not hasattr(entry, "query")


# =============================================================================
# Property 5: Scenario ID Preference
# =============================================================================


@st.composite
def benchmark_entry_dict_with_both_ids(draw):
    """
    Generate a dictionary that always contains BOTH query_id and scenario_id.

    This strategy ensures we test the scenario_id preference behavior.
    """
    # Generate distinct values for query_id and scenario_id to verify preference
    scenario_id = draw(st.text(min_size=1, max_size=50))
    query_id = draw(st.text(min_size=1, max_size=50))

    return {
        "scenario_id": scenario_id,
        "query_id": query_id,
        "patient_story": draw(st.text(min_size=1, max_size=500)),
        "patient_profile": draw(patient_profile_dict_strategy()),
        "condition_name": draw(st.text(max_size=50)),
        "preferred_care_option": draw(st.text(max_size=50)),
        "severity_level": draw(st.text(max_size=20)),
        "task_type": draw(st.text(max_size=50)),
        "has_image": draw(st.sampled_from(["True", "False"])),
        "scenario_complexity": draw(st.sampled_from(["", "regular", "chronic", "complicated"])),
    }


class TestScenarioIdPreferenceProperty:
    """
    **Feature: benchmark-seed-refactor, Property 5: Scenario ID Preference**

    *For any* dictionary containing both "query_id" and "scenario_id" keys,
    BenchmarkEntry.from_dict() SHALL use the scenario_id value for the entry's
    scenario_id field, ignoring query_id.

    **Validates: Requirements 4.6**
    """

    @given(data=benchmark_entry_dict_with_both_ids())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_5_scenario_id_preferred_over_query_id(self, data):
        """
        Property 5: scenario_id is always preferred when both are present.

        **Feature: benchmark-seed-refactor, Property 5: Scenario ID Preference**
        **Validates: Requirements 4.6**
        """
        # Capture the input values
        input_scenario_id = data["scenario_id"]
        input_query_id = data["query_id"]

        # from_dict() should not raise an exception
        entry = BenchmarkEntry.from_dict(data)

        # Entry should be created successfully
        assert entry is not None
        assert isinstance(entry, BenchmarkEntry)

        # scenario_id should be used, NOT query_id
        assert entry.scenario_id == input_scenario_id

        # Verify the id property also returns scenario_id
        assert entry.id == input_scenario_id

    @given(
        scenario_id=st.text(min_size=1, max_size=50),
        query_id=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_5_scenario_id_used_regardless_of_query_id_value(
        self, scenario_id, query_id
    ):
        """
        Property 5: scenario_id is used regardless of what query_id contains.

        **Feature: benchmark-seed-refactor, Property 5: Scenario ID Preference**
        **Validates: Requirements 4.6**
        """
        data = {
            "scenario_id": scenario_id,
            "query_id": query_id,
            "patient_story": "Test patient story",
            "patient_profile": {},
        }

        # from_dict() should not raise an exception
        entry = BenchmarkEntry.from_dict(data)

        # Entry should be created successfully
        assert entry is not None

        # scenario_id should always be used when present
        assert entry.scenario_id == scenario_id

        # Even if query_id is different, scenario_id takes precedence
        if scenario_id != query_id:
            assert entry.scenario_id != query_id

    @given(
        scenario_id=st.text(min_size=1, max_size=50),
        query_id=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_5_id_property_returns_scenario_id(self, scenario_id, query_id):
        """
        Property 5: The id property returns scenario_id, not query_id.

        **Feature: benchmark-seed-refactor, Property 5: Scenario ID Preference**
        **Validates: Requirements 4.6**
        """
        data = {
            "scenario_id": scenario_id,
            "query_id": query_id,
            "patient_story": "Test patient story",
            "patient_profile": {},
        }

        entry = BenchmarkEntry.from_dict(data)

        # The id property should return scenario_id
        assert entry.id == scenario_id
        assert entry.id == entry.scenario_id


# =============================================================================
# Property 6: ID Property Returns Scenario ID
# =============================================================================


@st.composite
def benchmark_entry_dict_with_scenario_id(draw):
    """
    Generate a dictionary with a non-empty scenario_id for testing the id property.
    """
    return {
        "scenario_id": draw(st.text(min_size=1, max_size=100)),
        "patient_story": draw(st.text(min_size=1, max_size=500)),
        "patient_profile": draw(patient_profile_dict_strategy()),
        "condition_name": draw(st.text(max_size=50)),
        "preferred_care_option": draw(st.text(max_size=50)),
        "severity_level": draw(st.text(max_size=20)),
        "task_type": draw(st.text(max_size=50)),
        "has_image": draw(st.sampled_from(["True", "False"])),
        "scenario_complexity": draw(st.sampled_from(["", "regular", "chronic", "complicated"])),
    }


class TestIdPropertyReturnsScenarioId:
    """
    **Feature: benchmark-seed-refactor, Property 6: ID Property Returns Scenario ID**

    *For any* BenchmarkEntry with a non-empty scenario_id, the id property SHALL
    return the exact scenario_id value.

    **Validates: Requirements 4.3**
    """

    @given(data=benchmark_entry_dict_with_scenario_id())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_id_returns_exact_scenario_id(self, data):
        """
        Property 6: id property returns the exact scenario_id value.

        **Feature: benchmark-seed-refactor, Property 6: ID Property Returns Scenario ID**
        **Validates: Requirements 4.3**
        """
        input_scenario_id = data["scenario_id"]

        entry = BenchmarkEntry.from_dict(data)

        # The id property must return the exact scenario_id value
        assert entry.id == input_scenario_id
        assert entry.id == entry.scenario_id

    @given(scenario_id=st.text(min_size=1, max_size=100))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_id_equals_scenario_id_for_any_value(self, scenario_id):
        """
        Property 6: id property equals scenario_id for any non-empty string value.

        **Feature: benchmark-seed-refactor, Property 6: ID Property Returns Scenario ID**
        **Validates: Requirements 4.3**
        """
        data = {
            "scenario_id": scenario_id,
            "patient_story": "Test patient story",
            "patient_profile": {},
        }

        entry = BenchmarkEntry.from_dict(data)

        # The id property must be identical to scenario_id
        assert entry.id == scenario_id
        assert entry.id == entry.scenario_id

    @given(
        scenario_id=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_id_preserves_special_characters(self, scenario_id):
        """
        Property 6: id property preserves special characters in scenario_id.

        **Feature: benchmark-seed-refactor, Property 6: ID Property Returns Scenario ID**
        **Validates: Requirements 4.3**
        """
        data = {
            "scenario_id": scenario_id,
            "patient_story": "Test patient story",
            "patient_profile": {},
        }

        entry = BenchmarkEntry.from_dict(data)

        # The id property must preserve the exact value including special chars
        assert entry.id == scenario_id
        assert len(entry.id) == len(scenario_id)
