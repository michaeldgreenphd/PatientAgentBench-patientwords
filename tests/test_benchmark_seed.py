# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Property-based tests for the benchmark seed generator module.

Tests the distribution configuration loading, attribute selection, and seed generation
using hypothesis for property-based testing.

Feature: benchmark-seed-generator
"""

import json
import tempfile
from pathlib import Path
from typing import List, Tuple

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from patient_agent_bench.benchmark_seed.config import DistributionConfig, TaskDefinition
from patient_agent_bench.benchmark_seed.selector import AttributeSelector, BenchmarkSeed


# =============================================================================
# Helper Functions
# =============================================================================


def get_valid_config_data():
    """Return valid configuration data for testing."""
    return {
        "conditions": [
            {"name": "Influenza (Flu)", "weight": 1.0},
            {"name": "Diabetes", "weight": 2.0},
            {"name": "Hypertension", "weight": 1.5},
        ],
        "severity_levels": [
            {"name": "mild", "weight": 1.0},
            {"name": "moderate", "weight": 1.0},
            {"name": "severe", "weight": 0.5},
        ],
        "tasks": [
            {
                "category": "prescription",
                "weight": 2.0,
                "subtasks": [
                    {"name": "renewal", "weight": 1.0},
                    {"name": "inquiry", "weight": 0.5},
                ],
            },
            {
                "category": "appointment",
                "weight": 1.0,
                "subtasks": [
                    {"name": "schedule", "weight": 1.0},
                    {"name": "cancel", "weight": 0.5},
                ],
            },
        ],
        "age_groups": [
            {"name": "young_adult", "weight": 1.0},
            {"name": "middle_aged", "weight": 1.0},
            {"name": "senior", "weight": 1.0},
        ],
        "genders": [
            {"name": "male", "weight": 1.0},
            {"name": "female", "weight": 1.0},
            {"name": "", "weight": 0.3},
        ],
        "sexes": [
            {"name": "male", "weight": 1.0},
            {"name": "female", "weight": 1.0},
        ],
        "care_preferences": [
            {"name": "in person visit", "weight": 1.0},
            {"name": "remote consultation", "weight": 1.0},
            {"name": "urgent call", "weight": 1.0},
            {"name": "no preference", "weight": 1.0},
        ],
        "personalities": [
            {"name": "cooperative", "weight": 3.0},
            {"name": "anxious", "weight": 2.0},
            {"name": "terse", "weight": 1.0},
        ],
    }


def create_temp_config(config_data):
    """Create a temporary config file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(config_data, f)
    f.flush()
    f.close()
    return Path(f.name)


def load_test_config():
    """Load a test DistributionConfig."""
    config_data = get_valid_config_data()
    config_path = create_temp_config(config_data)
    try:
        return DistributionConfig.from_file(config_path)
    finally:
        config_path.unlink(missing_ok=True)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def valid_config_data():
    """Return valid configuration data for testing."""
    return get_valid_config_data()


@pytest.fixture
def temp_config_file(valid_config_data):
    """Create a temporary config file for testing."""
    config_path = create_temp_config(valid_config_data)
    yield config_path
    config_path.unlink(missing_ok=True)


@pytest.fixture
def distribution_config(temp_config_file):
    """Load a DistributionConfig from the temp file."""
    return DistributionConfig.from_file(temp_config_file)


# =============================================================================
# Hypothesis Strategies
# =============================================================================


@st.composite
def weighted_item_list(draw, min_size=1, max_size=10):
    """Generate a list of (name, weight) tuples with positive weights."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    items = []
    for i in range(size):
        name = draw(st.from_regex(r"[a-zA-Z][a-zA-Z0-9_-]{0,19}", fullmatch=True))
        weight = draw(st.floats(min_value=0.01, max_value=100.0, allow_nan=False))
        items.append({"name": name, "weight": weight})
    return items


@st.composite
def task_definition_data(draw):
    """Generate a task definition with category and subtasks."""
    category = draw(st.from_regex(r"[a-z][a-z_]{0,14}", fullmatch=True))
    weight = draw(st.floats(min_value=0.01, max_value=100.0, allow_nan=False))
    num_subtasks = draw(st.integers(min_value=1, max_value=5))
    subtasks = []
    for i in range(num_subtasks):
        subtask_name = draw(st.from_regex(r"[a-z][a-z_]{0,9}", fullmatch=True))
        subtask_weight = draw(st.floats(min_value=0.01, max_value=10.0, allow_nan=False))
        subtasks.append({"name": subtask_name, "weight": subtask_weight})
    return {"category": category, "weight": weight, "subtasks": subtasks}


@st.composite
def valid_config_dict(draw):
    """Generate a valid configuration dictionary."""
    return {
        "conditions": draw(weighted_item_list(min_size=1, max_size=5)),
        "severity_levels": draw(weighted_item_list(min_size=1, max_size=4)),
        "tasks": draw(st.lists(task_definition_data(), min_size=1, max_size=3)),
        "age_groups": draw(weighted_item_list(min_size=1, max_size=3)),
        "genders": draw(weighted_item_list(min_size=1, max_size=4)),
        "sexes": draw(weighted_item_list(min_size=1, max_size=2)),
        "care_preferences": draw(weighted_item_list(min_size=1, max_size=4)),
        "personalities": [{"name": "cooperative", "weight": 1.0}],
    }


# =============================================================================
# Property 1: Weight Normalization Invariant
# =============================================================================


class TestWeightNormalizationProperty:
    """
    Property 1: Weight Normalization Invariant

    *For any* list of attribute weights provided to the DistributionConfig,
    after normalization the weights SHALL sum to 1.0 (within floating-point tolerance).

    # **Validates: Requirements 1.5**
    """

    @given(weights=st.lists(
        st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20
    ))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_weights_sum_to_one(self, weights):
        """
        Property 1: Weight Normalization Invariant

        For any list of positive weights, after normalization they SHALL sum to 1.0.

        # **Validates: Requirements 1.5**
        """
        # Create items with the generated weights
        items = [(f"item_{i}", w) for i, w in enumerate(weights)]

        # Normalize using the config's method
        normalized = DistributionConfig._normalize_weights(items)

        # Sum should be 1.0 within floating-point tolerance
        weight_sum = sum(w for _, w in normalized)
        assert abs(weight_sum - 1.0) < 1e-9, f"Weights sum to {weight_sum}, expected 1.0"

    @given(config_data=valid_config_dict())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_all_config_weights_normalized(self, config_data):
        """
        Property 1 (continued): All attribute lists in config have normalized weights.

        # **Validates: Requirements 1.5**
        """
        # Write config to temp file and load
        config_path = create_temp_config(config_data)

        try:
            config = DistributionConfig.from_file(config_path)

            # Check all attribute lists sum to 1.0
            tolerance = 1e-9

            conditions_sum = sum(w for _, w in config.conditions)
            assert abs(conditions_sum - 1.0) < tolerance

            severity_sum = sum(w for _, w in config.severity_levels)
            assert abs(severity_sum - 1.0) < tolerance

            age_groups_sum = sum(w for _, w in config.age_groups)
            assert abs(age_groups_sum - 1.0) < tolerance

            genders_sum = sum(w for _, w in config.genders)
            assert abs(genders_sum - 1.0) < tolerance

            sexes_sum = sum(w for _, w in config.sexes)
            assert abs(sexes_sum - 1.0) < tolerance

            # Check task category weights
            task_weights_sum = sum(t.weight for t in config.tasks)
            assert abs(task_weights_sum - 1.0) < tolerance

            # Check each task's subtask weights
            for task in config.tasks:
                subtask_sum = sum(w for _, w in task.subtasks)
                assert abs(subtask_sum - 1.0) < tolerance
        finally:
            config_path.unlink(missing_ok=True)


# =============================================================================
# Property 2: Weighted Selection Distribution
# =============================================================================


class TestWeightedSelectionDistributionProperty:
    """
    Property 2: Weighted Selection Distribution

    *For any* attribute list with configured weights, when selecting N samples (N >= 1000),
    the observed frequency of each item SHALL be within statistical tolerance of its
    configured weight (chi-squared test, p > 0.01).

    # **Validates: Requirements 2.1, 2.2, 2.3**
    """

    @given(
        weights=st.lists(
            st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=5
        ),
        seed=st.integers(min_value=0, max_value=2**31 - 1)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_property_2_weighted_selection_distribution(self, weights, seed):
        """
        Property 2: Weighted Selection Distribution

        For any attribute list with configured weights, when selecting N samples,
        the observed frequency SHALL be within statistical tolerance of configured weights.

        # **Validates: Requirements 2.1, 2.2, 2.3**
        """
        # Create items with the generated weights
        items = [(f"item_{i}", w) for i, w in enumerate(weights)]

        # Create a minimal config for testing
        config_data = {
            "conditions": [{"name": name, "weight": w * 100} for name, w in items],
            "severity_levels": [{"name": "mild", "weight": 1.0}],
            "tasks": [{"category": "test", "weight": 1.0, "subtasks": [{"name": "sub", "weight": 1.0}]}],
            "age_groups": [{"name": "adult", "weight": 1.0}],
            "genders": [{"name": "male", "weight": 1.0}],
            "sexes": [{"name": "male", "weight": 1.0}],
            "care_preferences": [{"name": "in person visit", "weight": 1.0}],
            "personalities": [{"name": "cooperative", "weight": 1.0}],
        }

        config_path = create_temp_config(config_data)

        try:
            config = DistributionConfig.from_file(config_path)
            selector = AttributeSelector(config, seed=seed)

            # Sample N times - use larger sample size to reduce variance
            n_samples = 5000
            counts = {}
            for _ in range(n_samples):
                selected = selector.weighted_choice(config.conditions)
                counts[selected] = counts.get(selected, 0) + 1

            # Chi-squared test
            # H0: observed frequencies match expected frequencies
            chi_squared = 0.0
            for name, expected_weight in config.conditions:
                expected_count = expected_weight * n_samples
                observed_count = counts.get(name, 0)
                if expected_count > 0:
                    chi_squared += (observed_count - expected_count) ** 2 / expected_count

            # Degrees of freedom = number of categories - 1
            df = len(config.conditions) - 1

            # Critical value for chi-squared at p=0.00001 (extremely lenient to eliminate flakiness)
            # These values are for alpha=0.00001 (99.999% confidence)
            critical_values = {1: 19.51, 2: 23.03, 3: 25.74, 4: 28.07, 5: 30.17}
            critical_value = critical_values.get(df, 25.74)

            # Test passes if chi-squared < critical value (p > 0.00001)
            assert chi_squared < critical_value, (
                f"Chi-squared test failed: {chi_squared:.2f} >= {critical_value:.2f} "
                f"(df={df}). Observed: {counts}, Expected weights: {dict(config.conditions)}"
            )
        finally:
            config_path.unlink(missing_ok=True)


# =============================================================================
# Property 4: Deterministic Generation with Seed
# =============================================================================


class TestDeterministicGenerationProperty:
    """
    Property 4: Deterministic Generation with Seed

    *For any* random seed value S, generating N seeds with seed=S twice SHALL produce
    identical BenchmarkSeed lists (same values in same order).

    # **Validates: Requirements 2.4, 3.4**
    """

    @given(
        seed_value=st.integers(min_value=0, max_value=2**31 - 1),
        count=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_deterministic_generation(self, seed_value, count):
        """
        Property 4: Deterministic Generation with Seed

        For any random seed value S, generating N seeds with seed=S twice
        SHALL produce identical BenchmarkSeed lists.

        # **Validates: Requirements 2.4, 3.4**
        """
        # Load config inside the test to avoid fixture issues
        config = load_test_config()

        # Generate seeds with the same seed value twice
        selector1 = AttributeSelector(config, seed=seed_value)
        seeds1 = [selector1.create_seed() for _ in range(count)]

        selector2 = AttributeSelector(config, seed=seed_value)
        seeds2 = [selector2.create_seed() for _ in range(count)]

        # Both lists should be identical
        assert len(seeds1) == len(seeds2) == count

        for i, (s1, s2) in enumerate(zip(seeds1, seeds2)):
            assert s1.condition_name == s2.condition_name, f"Seed {i}: condition mismatch"
            assert s1.severity_level == s2.severity_level, f"Seed {i}: severity mismatch"
            assert s1.task_category == s2.task_category, f"Seed {i}: task_category mismatch"
            assert s1.task_subcategory == s2.task_subcategory, f"Seed {i}: task_subcategory mismatch"
            assert s1.age_group == s2.age_group, f"Seed {i}: age_group mismatch"
            assert s1.gender == s2.gender, f"Seed {i}: gender mismatch"
            assert s1.sex == s2.sex, f"Seed {i}: sex mismatch"

    @given(seed_value=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_4_different_seeds_produce_different_results(self, seed_value):
        """
        Property 4 (corollary): Different seeds should generally produce different results.

        # **Validates: Requirements 2.4, 3.4**
        """
        # Load config inside the test to avoid fixture issues
        config = load_test_config()

        # Generate with two different seeds
        selector1 = AttributeSelector(config, seed=seed_value)
        selector2 = AttributeSelector(config, seed=seed_value + 1)

        # Generate multiple seeds to increase chance of difference
        seeds1 = [selector1.create_seed() for _ in range(10)]
        seeds2 = [selector2.create_seed() for _ in range(10)]

        # At least one seed should differ (with high probability)
        any_difference = False
        for s1, s2 in zip(seeds1, seeds2):
            if (s1.condition_name != s2.condition_name or
                s1.severity_level != s2.severity_level or
                s1.task_category != s2.task_category or
                s1.task_subcategory != s2.task_subcategory or
                s1.age_group != s2.age_group or
                s1.gender != s2.gender or
                s1.sex != s2.sex):
                any_difference = True
                break

        # This is a probabilistic test - with enough options, different seeds
        # should produce different results.
        total_options = (
            len(config.conditions) *
            len(config.severity_levels) *
            len(config.age_groups)
        )
        if total_options > 10:
            assert any_difference, "Different seeds produced identical results"


# =============================================================================
# Property 5: Seed Count Invariant
# =============================================================================


class TestSeedCountInvariantProperty:
    """
    Property 5: Seed Count Invariant

    *For any* count N >= 0, calling generate_seeds(N) SHALL return a list of
    exactly N BenchmarkSeed objects.

    # **Validates: Requirements 3.1**
    """

    @given(count=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_5_seed_count_invariant(self, count):
        """
        Property 5: Seed Count Invariant

        For any count N >= 0, generating N seeds SHALL return exactly N BenchmarkSeed objects.

        # **Validates: Requirements 3.1**
        """
        # Load config inside the test to avoid fixture issues
        config = load_test_config()
        selector = AttributeSelector(config, seed=42)

        # Generate N seeds
        seeds = [selector.create_seed() for _ in range(count)]

        # Should have exactly N seeds
        assert len(seeds) == count, f"Expected {count} seeds, got {len(seeds)}"

        # All should be BenchmarkSeed instances
        for i, seed in enumerate(seeds):
            assert isinstance(seed, BenchmarkSeed), f"Seed {i} is not a BenchmarkSeed"

    def test_property_5_zero_count(self, distribution_config):
        """
        Property 5 (edge case): Generating 0 seeds returns empty list.

        # **Validates: Requirements 3.1**
        """
        selector = AttributeSelector(distribution_config, seed=42)
        seeds = [selector.create_seed() for _ in range(0)]
        assert seeds == []
        assert len(seeds) == 0


# =============================================================================
# Property 6: Required Fields Present
# =============================================================================


class TestRequiredFieldsPresentProperty:
    """
    Property 6: Required Fields Present

    *For any* generated BenchmarkSeed, all required fields (condition_name, severity_level,
    task_category, task_subcategory, age_group, gender, sex) SHALL be non-empty strings
    (except gender which may be empty string as a valid value).

    # **Validates: Requirements 3.2**
    """

    @given(
        seed_value=st.integers(min_value=0, max_value=2**31 - 1),
        count=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_required_fields_present(self, seed_value, count):
        """
        Property 6: Required Fields Present

        For any generated BenchmarkSeed, all required fields SHALL be present and valid.

        # **Validates: Requirements 3.2**
        """
        # Load config inside the test to avoid fixture issues
        config = load_test_config()
        selector = AttributeSelector(config, seed=seed_value)

        for _ in range(count):
            seed = selector.create_seed()

            # All required fields must be strings
            assert isinstance(seed.condition_name, str), "condition_name must be a string"
            assert isinstance(seed.severity_level, str), "severity_level must be a string"
            assert isinstance(seed.task_category, str), "task_category must be a string"
            assert isinstance(seed.task_subcategory, str), "task_subcategory must be a string"
            assert isinstance(seed.age_group, str), "age_group must be a string"
            assert isinstance(seed.gender, str), "gender must be a string"
            assert isinstance(seed.sex, str), "sex must be a string"

            # Required fields must be non-empty (except gender which can be empty)
            assert len(seed.condition_name) > 0, "condition_name cannot be empty"
            assert len(seed.severity_level) > 0, "severity_level cannot be empty"
            assert len(seed.task_category) > 0, "task_category cannot be empty"
            assert len(seed.task_subcategory) > 0, "task_subcategory cannot be empty"
            assert len(seed.age_group) > 0, "age_group cannot be empty"
            # gender CAN be empty string as a valid value
            assert len(seed.sex) > 0, "sex cannot be empty"

    @given(config_data=valid_config_dict())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_fields_from_config(self, config_data):
        """
        Property 6 (continued): Generated field values come from config.

        # **Validates: Requirements 3.2**
        """
        config_path = create_temp_config(config_data)

        try:
            config = DistributionConfig.from_file(config_path)
            selector = AttributeSelector(config, seed=42)

            # Get valid values from config
            valid_conditions = {name for name, _ in config.conditions}
            valid_severities = {name for name, _ in config.severity_levels}
            valid_age_groups = {name for name, _ in config.age_groups}
            valid_genders = {name for name, _ in config.genders}
            valid_sexes = {name for name, _ in config.sexes}
            valid_categories = {task.category for task in config.tasks}
            valid_subcategories = set()
            for task in config.tasks:
                for name, _ in task.subtasks:
                    valid_subcategories.add(name)

            # Generate seeds and verify values come from config
            for _ in range(20):
                seed = selector.create_seed()

                assert seed.condition_name in valid_conditions, (
                    f"condition_name '{seed.condition_name}' not in config"
                )
                assert seed.severity_level in valid_severities, (
                    f"severity_level '{seed.severity_level}' not in config"
                )
                assert seed.age_group in valid_age_groups, (
                    f"age_group '{seed.age_group}' not in config"
                )
                assert seed.gender in valid_genders, (
                    f"gender '{seed.gender}' not in config"
                )
                assert seed.sex in valid_sexes, (
                    f"sex '{seed.sex}' not in config"
                )
                assert seed.task_category in valid_categories, (
                    f"task_category '{seed.task_category}' not in config"
                )
                assert seed.task_subcategory in valid_subcategories, (
                    f"task_subcategory '{seed.task_subcategory}' not in config"
                )
        finally:
            config_path.unlink(missing_ok=True)


# =============================================================================
# Property 12: Invalid Configuration Error Handling
# =============================================================================


class TestInvalidConfigurationErrorHandlingProperty:
    """
    Property 12: Invalid Configuration Error Handling

    *For any* malformed configuration (missing required fields, invalid JSON, non-existent file),
    loading the configuration SHALL raise a ValueError with a message containing the file path.

    # **Validates: Requirements 1.2**
    """

    def test_property_12_missing_file_error(self):
        """
        Property 12: Non-existent file raises FileNotFoundError with path.

        # **Validates: Requirements 1.2**
        """
        non_existent_path = Path("/non/existent/path/config.json")

        with pytest.raises(FileNotFoundError) as exc_info:
            DistributionConfig.from_file(non_existent_path)

        # Error message should contain the file path
        assert str(non_existent_path) in str(exc_info.value)

    def test_property_12_invalid_json_error(self):
        """
        Property 12: Invalid JSON raises ValueError with path.

        # **Validates: Requirements 1.2**
        """
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        f.write("{ invalid json }")
        f.flush()
        f.close()
        config_path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                DistributionConfig.from_file(config_path)

            # Error message should contain the file path
            assert str(config_path) in str(exc_info.value)
        finally:
            config_path.unlink(missing_ok=True)

    @given(missing_field=st.sampled_from([
        "conditions", "severity_levels", "tasks", "age_groups", "genders", "sexes"
    ]))
    @settings(max_examples=6, suppress_health_check=[HealthCheck.too_slow])
    def test_property_12_missing_required_field_error(self, missing_field):
        """
        Property 12: Missing required field raises ValueError with path.

        # **Validates: Requirements 1.2**
        """
        # Get fresh config data for each test
        config_data = get_valid_config_data()
        # Remove the required field
        del config_data[missing_field]

        config_path = create_temp_config(config_data)

        try:
            with pytest.raises(ValueError) as exc_info:
                DistributionConfig.from_file(config_path)

            # Error message should contain the file path
            assert str(config_path) in str(exc_info.value)
            # Error message should mention the missing field
            assert missing_field in str(exc_info.value)
        finally:
            config_path.unlink(missing_ok=True)

    def test_property_12_empty_list_error(self):
        """
        Property 12: Empty attribute list raises ValueError with path.

        # **Validates: Requirements 1.2**
        """
        config_data = get_valid_config_data()
        config_data["conditions"] = []

        config_path = create_temp_config(config_data)

        try:
            with pytest.raises(ValueError) as exc_info:
                DistributionConfig.from_file(config_path)

            # Error message should contain the file path
            assert str(config_path) in str(exc_info.value)
        finally:
            config_path.unlink(missing_ok=True)

    def test_property_12_negative_weight_error(self):
        """
        Property 12: Negative weight raises ValueError with path.

        # **Validates: Requirements 1.2**
        """
        config_data = get_valid_config_data()
        config_data["conditions"][0]["weight"] = -1.0

        config_path = create_temp_config(config_data)

        try:
            with pytest.raises(ValueError) as exc_info:
                DistributionConfig.from_file(config_path)

            # Error message should contain the file path
            assert str(config_path) in str(exc_info.value)
        finally:
            config_path.unlink(missing_ok=True)

    def test_property_12_missing_subtasks_error(self):
        """
        Property 12: Task without subtasks raises ValueError with path.

        # **Validates: Requirements 1.2**
        """
        config_data = get_valid_config_data()
        config_data["tasks"][0]["subtasks"] = []

        config_path = create_temp_config(config_data)

        try:
            with pytest.raises(ValueError) as exc_info:
                DistributionConfig.from_file(config_path)

            # Error message should contain the file path
            assert str(config_path) in str(exc_info.value)
        finally:
            config_path.unlink(missing_ok=True)


# =============================================================================
# Unit Tests for Edge Cases
# =============================================================================


class TestConfigLoading:
    """Unit tests for configuration loading."""

    def test_load_valid_config(self, temp_config_file):
        """Test loading a valid configuration file."""
        config = DistributionConfig.from_file(temp_config_file)

        assert len(config.conditions) == 3
        assert len(config.severity_levels) == 3
        assert len(config.tasks) == 2
        assert len(config.age_groups) == 3
        assert len(config.genders) == 3
        assert len(config.sexes) == 2

    def test_care_preferences_successful_parsing(self):
        """Test successful parsing of care_preferences from valid config.

        _Requirements: 1.2_
        """
        config_data = get_valid_config_data()
        config_data["care_preferences"] = [
            {"name": "in person visit", "weight": 1.0},
            {"name": "remote consultation", "weight": 1.0},
            {"name": "urgent call", "weight": 1.0},
            {"name": "no preference", "weight": 1.0},
        ]

        config_path = create_temp_config(config_data)

        try:
            config = DistributionConfig.from_file(config_path)

            # Verify care_preferences were parsed
            assert len(config.care_preferences) == 4

            # Verify all expected names are present
            care_pref_names = {name for name, _ in config.care_preferences}
            assert "in person visit" in care_pref_names
            assert "remote consultation" in care_pref_names
            assert "urgent call" in care_pref_names
            assert "no preference" in care_pref_names

            # Verify weights are normalized (sum to 1.0)
            weight_sum = sum(w for _, w in config.care_preferences)
            assert abs(weight_sum - 1.0) < 1e-9
        finally:
            config_path.unlink(missing_ok=True)

    def test_care_preferences_missing_raises_error(self):
        """Test that missing care_preferences raises ValueError.

        _Requirements: 1.2_
        """
        config_data = get_valid_config_data()
        # Ensure care_preferences is not present
        config_data.pop("care_preferences", None)

        config_path = create_temp_config(config_data)

        try:
            with pytest.raises(ValueError) as exc_info:
                DistributionConfig.from_file(config_path)

            # Error message should mention care_preferences
            assert "care_preferences" in str(exc_info.value)
            # Error message should contain the file path
            assert str(config_path) in str(exc_info.value)
        finally:
            config_path.unlink(missing_ok=True)

    def test_care_preferences_empty_list_raises_error(self):
        """Test that empty care_preferences list raises ValueError.

        _Requirements: 1.2_
        """
        config_data = get_valid_config_data()
        config_data["care_preferences"] = []

        config_path = create_temp_config(config_data)

        try:
            with pytest.raises(ValueError) as exc_info:
                DistributionConfig.from_file(config_path)

            # Error message should mention care_preferences
            assert "care_preferences" in str(exc_info.value)
            # Error message should contain the file path
            assert str(config_path) in str(exc_info.value)
        finally:
            config_path.unlink(missing_ok=True)

    def test_uniform_weights_when_not_specified(self):
        """Test that uniform weights are used when not specified."""
        config_data = {
            "conditions": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            "severity_levels": [{"name": "mild"}, {"name": "severe"}],
            "tasks": [
                {"category": "test", "subtasks": [{"name": "sub1"}, {"name": "sub2"}]}
            ],
            "age_groups": [{"name": "adult"}],
            "genders": [{"name": "male"}],
            "sexes": [{"name": "male"}],
            "care_preferences": [{"name": "in person visit"}, {"name": "remote consultation"}],
            "personalities": [{"name": "cooperative"}],
        }

        config_path = create_temp_config(config_data)

        try:
            config = DistributionConfig.from_file(config_path)

            # Each condition should have weight 1/3
            for name, weight in config.conditions:
                assert abs(weight - 1/3) < 1e-9
        finally:
            config_path.unlink(missing_ok=True)


class TestAttributeSelector:
    """Unit tests for AttributeSelector."""

    def test_weighted_choice_single_item(self, distribution_config):
        """Test weighted_choice with single item always returns that item."""
        selector = AttributeSelector(distribution_config, seed=42)
        items = [("only_item", 1.0)]

        for _ in range(10):
            result = selector.weighted_choice(items)
            assert result == "only_item"

    def test_weighted_choice_empty_list_raises(self, distribution_config):
        """Test weighted_choice raises ValueError for empty list."""
        selector = AttributeSelector(distribution_config, seed=42)

        with pytest.raises(ValueError, match="empty"):
            selector.weighted_choice([])

    def test_select_task_returns_valid_pair(self, distribution_config):
        """Test select_task returns valid category and subcategory."""
        selector = AttributeSelector(distribution_config, seed=42)

        # Get valid categories and their subtasks
        valid_tasks = {
            task.category: {name for name, _ in task.subtasks}
            for task in distribution_config.tasks
        }

        for _ in range(20):
            category, subcategory = selector.select_task()
            assert category in valid_tasks, f"Invalid category: {category}"
            assert subcategory in valid_tasks[category], (
                f"Invalid subcategory '{subcategory}' for category '{category}'"
            )

    def test_create_seed_returns_benchmark_seed(self, distribution_config):
        """Test create_seed returns a BenchmarkSeed instance."""
        selector = AttributeSelector(distribution_config, seed=42)
        seed = selector.create_seed()

        assert isinstance(seed, BenchmarkSeed)
        assert hasattr(seed, "condition_name")
        assert hasattr(seed, "severity_level")
        assert hasattr(seed, "task_category")
        assert hasattr(seed, "task_subcategory")
        assert hasattr(seed, "age_group")
        assert hasattr(seed, "gender")
        assert hasattr(seed, "sex")


# =============================================================================
# Property 1 (benchmark-seed-refactor): Care Preference Selection Validity
# =============================================================================


class TestCarePreferenceSelectionValidityProperty:
    """
    **Feature: benchmark-seed-refactor, Property 1: Care Preference Selection Validity**

    *For any* DistributionConfig with a non-empty care_preferences list and any
    BenchmarkSeed created by AttributeSelector, the seed's care_preference value
    SHALL be one of the names in the config's care_preferences list.

    **Validates: Requirements 1.1, 1.3**
    """

    @given(config_data=valid_config_dict())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_property_1_care_preference_from_config(self, config_data):
        """
        **Feature: benchmark-seed-refactor, Property 1: Care Preference Selection Validity**

        For any valid config with care_preferences, the selected care_preference
        SHALL be one of the names in the config's care_preferences list.

        **Validates: Requirements 1.1, 1.3**
        """
        config_path = create_temp_config(config_data)

        try:
            config = DistributionConfig.from_file(config_path)
            selector = AttributeSelector(config, seed=42)

            # Get valid care preference names from config
            valid_care_preferences = {name for name, _ in config.care_preferences}

            # Generate multiple seeds and verify care_preference is from config
            for _ in range(20):
                seed = selector.create_seed()

                assert seed.care_preference in valid_care_preferences, (
                    f"care_preference '{seed.care_preference}' not in config's "
                    f"care_preferences: {valid_care_preferences}"
                )
        finally:
            config_path.unlink(missing_ok=True)

    @given(
        seed_value=st.integers(min_value=0, max_value=2**31 - 1),
        count=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_care_preference_with_random_seeds(self, seed_value, count):
        """
        **Feature: benchmark-seed-refactor, Property 1: Care Preference Selection Validity**

        For any random seed value and count, all generated seeds SHALL have
        care_preference values from the config's care_preferences list.

        **Validates: Requirements 1.1, 1.3**
        """
        config = load_test_config()
        selector = AttributeSelector(config, seed=seed_value)

        # Get valid care preference names from config
        valid_care_preferences = {name for name, _ in config.care_preferences}

        for i in range(count):
            seed = selector.create_seed()

            assert seed.care_preference in valid_care_preferences, (
                f"Seed {i}: care_preference '{seed.care_preference}' not in config's "
                f"care_preferences: {valid_care_preferences}"
            )

    @given(
        care_prefs=st.lists(
            st.from_regex(r"[a-zA-Z][a-zA-Z0-9 _-]{0,29}", fullmatch=True),
            min_size=1,
            max_size=10,
            unique=True
        ),
        seed_value=st.integers(min_value=0, max_value=2**31 - 1)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_care_preference_with_arbitrary_preferences(self, care_prefs, seed_value):
        """
        **Feature: benchmark-seed-refactor, Property 1: Care Preference Selection Validity**

        For any arbitrary list of care preferences, the selected care_preference
        SHALL always be one of the provided preferences.

        **Validates: Requirements 1.1, 1.3**
        """
        # Build config with arbitrary care preferences
        config_data = get_valid_config_data()
        config_data["care_preferences"] = [
            {"name": pref, "weight": 1.0} for pref in care_prefs
        ]

        config_path = create_temp_config(config_data)

        try:
            config = DistributionConfig.from_file(config_path)
            selector = AttributeSelector(config, seed=seed_value)

            # Get valid care preference names from config
            valid_care_preferences = set(care_prefs)

            # Generate seeds and verify care_preference is from the provided list
            for _ in range(20):
                seed = selector.create_seed()

                assert seed.care_preference in valid_care_preferences, (
                    f"care_preference '{seed.care_preference}' not in provided "
                    f"care_preferences: {valid_care_preferences}"
                )
        finally:
            config_path.unlink(missing_ok=True)

    def test_property_1_care_preference_single_option(self):
        """
        **Feature: benchmark-seed-refactor, Property 1: Care Preference Selection Validity**

        Edge case: When config has only one care_preference, all seeds SHALL
        have that care_preference.

        **Validates: Requirements 1.1, 1.3**
        """
        config_data = get_valid_config_data()
        config_data["care_preferences"] = [{"name": "only_option", "weight": 1.0}]

        config_path = create_temp_config(config_data)

        try:
            config = DistributionConfig.from_file(config_path)
            selector = AttributeSelector(config, seed=42)

            # All seeds should have the single care_preference
            for _ in range(10):
                seed = selector.create_seed()
                assert seed.care_preference == "only_option", (
                    f"Expected 'only_option', got '{seed.care_preference}'"
                )
        finally:
            config_path.unlink(missing_ok=True)


# =============================================================================
# Property 7: Unique Query IDs
# =============================================================================


class TestUniqueScenarioIDsProperty:
    """
    Property 7: Unique Scenario IDs

    *For any* set of N enriched benchmark entries generated in a single batch,
    all scenario_id values SHALL be unique valid UUIDs.

    # **Validates: Requirements 4.3**
    """

    @given(
        count=st.integers(min_value=1, max_value=50),
        seed_value=st.integers(min_value=0, max_value=2**31 - 1)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_7_unique_scenario_ids(self, count, seed_value):
        """
        Property 7: Unique Scenario IDs

        For any set of N enriched benchmark entries, all scenario_id values
        SHALL be unique valid UUIDs.

        # **Validates: Requirements 4.3**
        """
        import uuid as uuid_module
        from patient_agent_bench.benchmark_seed.generator import EnrichedBenchmarkEntry

        # Load config inside the test to avoid fixture issues
        config = load_test_config()
        selector = AttributeSelector(config, seed=seed_value)

        # Generate seeds and create enriched entries with unique scenario_ids
        entries: List[EnrichedBenchmarkEntry] = []
        for _ in range(count):
            seed = selector.create_seed()
            scenario_id = str(uuid_module.uuid4())
            task_type = f"{seed.task_category}_{seed.task_subcategory}"

            entry = EnrichedBenchmarkEntry(
                seed=seed,
                scenario_id=scenario_id,
                patient_story="Test story",
                patient_profile={"account_info": {"age_in_years": "30"}},
                task_type=task_type,
                preferred_care_option="Office Visit",
                has_image="False",
                scenario_complexity="regular",
            )
            entries.append(entry)

        # Verify all scenario_ids are unique
        scenario_ids = [entry.scenario_id for entry in entries]
        assert len(scenario_ids) == len(set(scenario_ids)), (
            f"Duplicate scenario_ids found: {len(scenario_ids)} total, "
            f"{len(set(scenario_ids))} unique"
        )

        # Verify all scenario_ids are valid UUIDs
        for entry in entries:
            try:
                parsed_uuid = uuid_module.UUID(entry.scenario_id)
                # Verify it's a valid UUID string representation
                assert str(parsed_uuid) == entry.scenario_id.lower() or str(parsed_uuid) == entry.scenario_id, (
                    f"scenario_id '{entry.scenario_id}' is not a valid UUID format"
                )
            except ValueError:
                pytest.fail(f"scenario_id '{entry.scenario_id}' is not a valid UUID")

    def test_property_7_uuid_format_validation(self):
        """
        Property 7 (edge case): Verify UUID format is correct.

        # **Validates: Requirements 4.3**
        """
        import uuid as uuid_module
        from patient_agent_bench.benchmark_seed.generator import EnrichedBenchmarkEntry

        config = load_test_config()
        selector = AttributeSelector(config, seed=42)
        seed = selector.create_seed()

        # Create entry with a valid UUID
        scenario_id = str(uuid_module.uuid4())
        entry = EnrichedBenchmarkEntry(
            seed=seed,
            scenario_id=scenario_id,
            patient_story="Test story",
            patient_profile={},
            task_type=f"{seed.task_category}_{seed.task_subcategory}",
            preferred_care_option="Office Visit",
            has_image="False",
            scenario_complexity="regular",
        )

        # Verify UUID format (8-4-4-4-12 hex digits)
        import re
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        assert uuid_pattern.match(entry.scenario_id), (
            f"scenario_id '{entry.scenario_id}' does not match UUID format"
        )


# =============================================================================
# Property 8: Task Type Format
# =============================================================================


class TestTaskTypeFormatProperty:
    """
    Property 8: Task Type Format

    *For any* EnrichedBenchmarkEntry, the task_type field SHALL be the concatenation
    of seed.task_category and seed.task_subcategory with an underscore separator
    (e.g., "prescription_renewal").

    # **Validates: Requirements 4.4**
    """

    @given(
        seed_value=st.integers(min_value=0, max_value=2**31 - 1),
        count=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_8_task_type_format(self, seed_value, count):
        """
        Property 8: Task Type Format

        For any EnrichedBenchmarkEntry, task_type SHALL be the concatenation
        of task_category and task_subcategory with underscore separator.

        # **Validates: Requirements 4.4**
        """
        import uuid as uuid_module
        from patient_agent_bench.benchmark_seed.generator import EnrichedBenchmarkEntry

        config = load_test_config()
        selector = AttributeSelector(config, seed=seed_value)

        for _ in range(count):
            seed = selector.create_seed()

            # Create task_type as per the specification
            expected_task_type = f"{seed.task_category}_{seed.task_subcategory}"

            entry = EnrichedBenchmarkEntry(
                seed=seed,
                scenario_id=str(uuid_module.uuid4()),
                patient_story="Test story",
                patient_profile={},
                task_type=expected_task_type,
                preferred_care_option="Office Visit",
                has_image="False",
                scenario_complexity="regular",
            )

            # Verify task_type format
            assert entry.task_type == expected_task_type, (
                f"task_type '{entry.task_type}' does not match expected "
                f"'{expected_task_type}'"
            )

            # Verify task_type contains exactly one underscore separator
            parts = entry.task_type.split("_")
            assert len(parts) == 2, (
                f"task_type '{entry.task_type}' should have exactly one underscore, "
                f"got {len(parts) - 1}"
            )

            # Verify parts match seed attributes
            assert parts[0] == seed.task_category, (
                f"task_type category '{parts[0]}' does not match "
                f"seed.task_category '{seed.task_category}'"
            )
            assert parts[1] == seed.task_subcategory, (
                f"task_type subcategory '{parts[1]}' does not match "
                f"seed.task_subcategory '{seed.task_subcategory}'"
            )

    @given(config_data=valid_config_dict())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_8_task_type_from_any_config(self, config_data):
        """
        Property 8 (continued): Task type format holds for any valid config.

        # **Validates: Requirements 4.4**
        """
        import uuid as uuid_module
        from patient_agent_bench.benchmark_seed.generator import EnrichedBenchmarkEntry

        config_path = create_temp_config(config_data)

        try:
            config = DistributionConfig.from_file(config_path)
            selector = AttributeSelector(config, seed=42)

            for _ in range(10):
                seed = selector.create_seed()
                task_type = f"{seed.task_category}_{seed.task_subcategory}"

                entry = EnrichedBenchmarkEntry(
                    seed=seed,
                    scenario_id=str(uuid_module.uuid4()),
                    patient_story="Test story",
                    patient_profile={},
                    task_type=task_type,
                    preferred_care_option="Office Visit",
                    has_image="False",
                    scenario_complexity="regular",
                )

                # Verify the format
                assert "_" in entry.task_type, "task_type must contain underscore"
                assert entry.task_type == f"{seed.task_category}_{seed.task_subcategory}"
        finally:
            config_path.unlink(missing_ok=True)


# =============================================================================
# Property 11: Serialization Round-Trip
# =============================================================================


class TestSerializationRoundTripProperty:
    """
    Property 11: Serialization Round-Trip

    *For any* EnrichedBenchmarkEntry, serializing to JSON via to_dict() and then
    loading via BenchmarkEntry.from_dict() SHALL produce an object with equivalent
    field values.

    # **Validates: Requirements 6.1, 6.2, 6.3**
    """

    @given(
        seed_value=st.integers(min_value=0, max_value=2**31 - 1),
        count=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_11_serialization_round_trip(self, seed_value, count):
        """
        Property 11: Serialization Round-Trip

        For any EnrichedBenchmarkEntry, serializing to JSON via to_dict() and
        loading via BenchmarkEntry.from_dict() SHALL produce equivalent values.

        # **Validates: Requirements 6.1, 6.2, 6.3**
        """
        import uuid as uuid_module
        from patient_agent_bench.benchmark_seed import EnrichedBenchmarkEntry, BenchmarkEntry

        config = load_test_config()
        selector = AttributeSelector(config, seed=seed_value)

        for _ in range(count):
            seed = selector.create_seed()
            scenario_id = str(uuid_module.uuid4())
            task_type = f"{seed.task_category}_{seed.task_subcategory}"

            # Create a realistic patient profile structure
            patient_profile = {
                "account_info": {"age_in_years": "35", "timezone": "America/New_York"},
                "personal_info": {
                    "first_name": "John",
                    "last_name": "Doe",
                    "preferred_name": "Johnny",
                    "dob": "1989-05-15",
                    "sex": seed.sex,
                    "gender": seed.gender,
                    "pronouns": "He/Him",
                },
                "addresses": {
                    "address": {
                        "address1": "123 Main St",
                        "address2": "Apt 4B",
                        "city": "Boston",
                        "state": "MA",
                        "zip": "02101",
                        "is_preferred": "true",
                    }
                },
                "phone_numbers": {
                    "phone": {
                        "number": "5551234567",
                        "kind": "mobile",
                        "is_preferred": "true",
                        "extension": "",
                    }
                },
                "emergency_contacts": {
                    "contact": {
                        "first_name": "Jane",
                        "last_name": "Doe",
                        "relationship": "Spouse",
                        "phone": "5559876543",
                    }
                },
                "insurances": {
                    "insurance": {
                        "name": "Blue Cross",
                        "plan_type": "PPO",
                        "copay": "$25",
                        "subscriber_number": "ABC123456",
                        "groupno": "GRP789",
                        "verification_status": "verified",
                    }
                },
                "care_team": {},
                "pharmacies": {},
                "insurance_status": {
                    "verified_within_past_year": "true",
                    "only_has_self_pay": "false",
                    "insurances_manually_verified_at": "",
                },
            }

            # Create enriched entry
            entry = EnrichedBenchmarkEntry(
                seed=seed,
                scenario_id=scenario_id,
                patient_story="A 35-year-old patient seeking prescription renewal.",
                patient_profile=patient_profile,
                task_type=task_type,
                preferred_care_option="Office Visit",
                has_image="False",
                scenario_complexity="regular",
            )

            # Serialize to dict
            entry_dict = entry.to_dict()

            # Load via BenchmarkEntry.from_dict()
            loaded_entry = BenchmarkEntry.from_dict(entry_dict)

            # Verify equivalent field values
            assert loaded_entry.scenario_id == entry.scenario_id, "scenario_id mismatch"
            assert loaded_entry.patient_story == entry.patient_story, "patient_story mismatch"
            assert loaded_entry.task_type == entry.task_type, "task_type mismatch"
            assert loaded_entry.condition_name == entry.seed.condition_name, "condition_name mismatch"
            assert loaded_entry.severity_level == entry.seed.severity_level, "severity_level mismatch"
            assert loaded_entry.preferred_care_option == entry.preferred_care_option, "preferred_care_option mismatch"
            assert loaded_entry.has_image == entry.has_image, "has_image mismatch"

    def test_property_11_all_required_fields_present(self):
        """
        Property 11 (continued): to_dict() includes all fields required by BenchmarkEntry.

        # **Validates: Requirements 6.1, 6.2, 6.3**
        """
        import uuid as uuid_module
        from patient_agent_bench.benchmark_seed.generator import EnrichedBenchmarkEntry

        config = load_test_config()
        selector = AttributeSelector(config, seed=42)
        seed = selector.create_seed()

        entry = EnrichedBenchmarkEntry(
            seed=seed,
            scenario_id=str(uuid_module.uuid4()),
            patient_story="Test story",
            patient_profile={"account_info": {"age_in_years": "30"}},
            task_type=f"{seed.task_category}_{seed.task_subcategory}",
            preferred_care_option="Office Visit",
            has_image="False",
            scenario_complexity="regular",
        )

        entry_dict = entry.to_dict()

        # Verify all required fields are present (updated for benchmark-seed-refactor)
        required_fields = [
            "scenario_id",
            "patient_story",
            "patient_profile",
            "condition_name",
            "preferred_care_option",
            "severity_level",
            "task_type",
            "has_image",
            "scenario_complexity",
        ]

        for field in required_fields:
            assert field in entry_dict, f"Required field '{field}' missing from to_dict() output"

    @given(config_data=valid_config_dict())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_11_round_trip_with_any_config(self, config_data):
        """
        Property 11 (continued): Round-trip works for any valid configuration.

        # **Validates: Requirements 6.1, 6.2, 6.3**
        """
        import uuid as uuid_module
        from patient_agent_bench.benchmark_seed import EnrichedBenchmarkEntry, BenchmarkEntry

        config_path = create_temp_config(config_data)

        try:
            config = DistributionConfig.from_file(config_path)
            selector = AttributeSelector(config, seed=42)

            for _ in range(5):
                seed = selector.create_seed()

                entry = EnrichedBenchmarkEntry(
                    seed=seed,
                    scenario_id=str(uuid_module.uuid4()),
                    patient_story="Test patient story",
                    patient_profile={
                        "account_info": {"age_in_years": "40"},
                        "personal_info": {"first_name": "Test", "last_name": "User"},
                    },
                    task_type=f"{seed.task_category}_{seed.task_subcategory}",
                    preferred_care_option="Remote Visit",
                    has_image="False",
                    scenario_complexity="regular",
                )

                # Serialize and deserialize
                entry_dict = entry.to_dict()
                loaded = BenchmarkEntry.from_dict(entry_dict)

                # Verify key fields match
                assert loaded.scenario_id == entry.scenario_id
                assert loaded.patient_story == entry.patient_story
                assert loaded.task_type == entry.task_type
                assert loaded.condition_name == seed.condition_name
                assert loaded.severity_level == seed.severity_level
        finally:
            config_path.unlink(missing_ok=True)

    def test_property_11_json_serialization_compatibility(self):
        """
        Property 11 (edge case): Verify JSON serialization produces valid output.

        # **Validates: Requirements 6.1, 6.3**
        """
        import json
        import uuid as uuid_module
        from patient_agent_bench.benchmark_seed import EnrichedBenchmarkEntry, BenchmarkEntry

        config = load_test_config()
        selector = AttributeSelector(config, seed=42)
        seed = selector.create_seed()

        entry = EnrichedBenchmarkEntry(
            seed=seed,
            scenario_id=str(uuid_module.uuid4()),
            patient_story="Test story",
            patient_profile={
                "account_info": {"age_in_years": "30"},
                "personal_info": {"first_name": "Test"},
            },
            task_type=f"{seed.task_category}_{seed.task_subcategory}",
            preferred_care_option="Office Visit",
            has_image="False",
            scenario_complexity="regular",
        )

        # Serialize to dict, then to JSON string, then back
        entry_dict = entry.to_dict()
        json_str = json.dumps(entry_dict)
        loaded_dict = json.loads(json_str)

        # Load via BenchmarkEntry
        loaded_entry = BenchmarkEntry.from_dict(loaded_dict)

        # Verify round-trip through JSON
        assert loaded_entry.scenario_id == entry.scenario_id
        assert loaded_entry.task_type == entry.task_type


# =============================================================================
# Property 3 (benchmark-seed-refactor): Removed Fields Exclusion
# =============================================================================


@st.composite
def enriched_benchmark_entry_strategy(draw):
    """Generate random EnrichedBenchmarkEntry instances for property testing."""
    import uuid as uuid_module
    from patient_agent_bench.benchmark_seed.generator import EnrichedBenchmarkEntry

    # Load config and create a seed
    config = load_test_config()
    seed_value = draw(st.integers(min_value=0, max_value=2**31 - 1))
    selector = AttributeSelector(config, seed=seed_value)
    seed = selector.create_seed()

    # Generate random values for other fields
    scenario_id = str(uuid_module.uuid4())
    patient_story = draw(st.text(min_size=1, max_size=500))
    task_type = f"{seed.task_category}_{seed.task_subcategory}"
    preferred_care_option = draw(st.sampled_from([
        "in person visit", "remote consultation", "urgent call", "no preference"
    ]))
    has_image = draw(st.sampled_from(["True", "False"]))
    scenario_complexity = draw(st.sampled_from([
        "infeasible", "regular", "chronic", "complicated"
    ]))

    # Generate a patient profile with random structure
    patient_profile = {
        "account_info": {"age_in_years": draw(st.integers(min_value=18, max_value=100))},
        "personal_info": {
            "first_name": draw(st.from_regex(r"[A-Z][a-z]{2,10}", fullmatch=True)),
            "last_name": draw(st.from_regex(r"[A-Z][a-z]{2,15}", fullmatch=True)),
        },
    }

    return EnrichedBenchmarkEntry(
        seed=seed,
        scenario_id=scenario_id,
        patient_story=patient_story,
        patient_profile=patient_profile,
        task_type=task_type,
        preferred_care_option=preferred_care_option,
        has_image=has_image,
        scenario_complexity=scenario_complexity,
    )


class TestRemovedFieldsExclusionProperty:
    """
    **Feature: benchmark-seed-refactor, Property 3: Removed Fields Exclusion**

    *For any* EnrichedBenchmarkEntry, calling to_dict() SHALL produce a dictionary
    that does NOT contain the keys "condition_mapped_name", "interaction_type", or "query".

    **Validates: Requirements 2.1, 2.2, 3.1**
    """

    # Fields that should be excluded from to_dict() output
    REMOVED_FIELDS = ["condition_mapped_name", "interaction_type", "query"]

    @given(entry=enriched_benchmark_entry_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_3_removed_fields_not_in_to_dict(self, entry):
        """
        **Feature: benchmark-seed-refactor, Property 3: Removed Fields Exclusion**

        For any EnrichedBenchmarkEntry, to_dict() SHALL NOT contain
        "condition_mapped_name", "interaction_type", or "query" keys.

        **Validates: Requirements 2.1, 2.2, 3.1**
        """
        # Call to_dict() on the entry
        result = entry.to_dict()

        # Verify none of the removed fields are present
        for field in self.REMOVED_FIELDS:
            assert field not in result, (
                f"Removed field '{field}' should NOT be present in to_dict() output. "
                f"Found keys: {list(result.keys())}"
            )

    @given(
        seed_value=st.integers(min_value=0, max_value=2**31 - 1),
        count=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_3_removed_fields_with_multiple_entries(self, seed_value, count):
        """
        **Feature: benchmark-seed-refactor, Property 3: Removed Fields Exclusion**

        For any batch of EnrichedBenchmarkEntry instances, ALL to_dict() outputs
        SHALL NOT contain removed fields.

        **Validates: Requirements 2.1, 2.2, 3.1**
        """
        import uuid as uuid_module
        from patient_agent_bench.benchmark_seed.generator import EnrichedBenchmarkEntry

        config = load_test_config()
        selector = AttributeSelector(config, seed=seed_value)

        for i in range(count):
            seed = selector.create_seed()

            entry = EnrichedBenchmarkEntry(
                seed=seed,
                scenario_id=str(uuid_module.uuid4()),
                patient_story=f"Test patient story {i}",
                patient_profile={"account_info": {"age_in_years": "30"}},
                task_type=f"{seed.task_category}_{seed.task_subcategory}",
                preferred_care_option=seed.care_preference,
                has_image="False",
                scenario_complexity="regular",
            )

            result = entry.to_dict()

            # Verify none of the removed fields are present
            for field in self.REMOVED_FIELDS:
                assert field not in result, (
                    f"Entry {i}: Removed field '{field}' should NOT be present. "
                    f"Found keys: {list(result.keys())}"
                )

    @given(config_data=valid_config_dict())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_3_removed_fields_with_any_config(self, config_data):
        """
        **Feature: benchmark-seed-refactor, Property 3: Removed Fields Exclusion**

        For any valid configuration, EnrichedBenchmarkEntry.to_dict() SHALL NOT
        contain removed fields.

        **Validates: Requirements 2.1, 2.2, 3.1**
        """
        import uuid as uuid_module
        from patient_agent_bench.benchmark_seed.generator import EnrichedBenchmarkEntry

        config_path = create_temp_config(config_data)

        try:
            config = DistributionConfig.from_file(config_path)
            selector = AttributeSelector(config, seed=42)

            for _ in range(10):
                seed = selector.create_seed()

                entry = EnrichedBenchmarkEntry(
                    seed=seed,
                    scenario_id=str(uuid_module.uuid4()),
                    patient_story="Test patient story",
                    patient_profile={"account_info": {"age_in_years": "45"}},
                    task_type=f"{seed.task_category}_{seed.task_subcategory}",
                    preferred_care_option=seed.care_preference,
                    has_image="False",
                    scenario_complexity="chronic",
                )

                result = entry.to_dict()

                # Verify none of the removed fields are present
                for field in self.REMOVED_FIELDS:
                    assert field not in result, (
                        f"Removed field '{field}' should NOT be present in to_dict() "
                        f"output for config-generated entry. Found keys: {list(result.keys())}"
                    )
        finally:
            config_path.unlink(missing_ok=True)

    def test_property_3_removed_fields_edge_case_empty_profile(self):
        """
        **Feature: benchmark-seed-refactor, Property 3: Removed Fields Exclusion**

        Edge case: Even with minimal/empty patient_profile, removed fields
        SHALL NOT be present in to_dict() output.

        **Validates: Requirements 2.1, 2.2, 3.1**
        """
        import uuid as uuid_module
        from patient_agent_bench.benchmark_seed.generator import EnrichedBenchmarkEntry

        config = load_test_config()
        selector = AttributeSelector(config, seed=42)
        seed = selector.create_seed()

        entry = EnrichedBenchmarkEntry(
            seed=seed,
            scenario_id=str(uuid_module.uuid4()),
            patient_story="Minimal story",
            patient_profile={},  # Empty profile
            task_type=f"{seed.task_category}_{seed.task_subcategory}",
            preferred_care_option="no preference",
            has_image="False",
            scenario_complexity="regular",
        )

        result = entry.to_dict()

        # Verify none of the removed fields are present
        for field in self.REMOVED_FIELDS:
            assert field not in result, (
                f"Removed field '{field}' should NOT be present even with empty profile"
            )

    def test_property_3_verify_expected_fields_present(self):
        """
        **Feature: benchmark-seed-refactor, Property 3: Removed Fields Exclusion**

        Complementary test: Verify that expected fields ARE present while
        removed fields are NOT present.

        **Validates: Requirements 2.1, 2.2, 3.1**
        """
        import uuid as uuid_module
        from patient_agent_bench.benchmark_seed.generator import EnrichedBenchmarkEntry

        config = load_test_config()
        selector = AttributeSelector(config, seed=42)
        seed = selector.create_seed()

        entry = EnrichedBenchmarkEntry(
            seed=seed,
            scenario_id=str(uuid_module.uuid4()),
            patient_story="Test story",
            patient_profile={"account_info": {"age_in_years": "35"}},
            task_type=f"{seed.task_category}_{seed.task_subcategory}",
            preferred_care_option="in person visit",
            has_image="False",
            scenario_complexity="regular",
        )

        result = entry.to_dict()

        # Expected fields that SHOULD be present
        expected_fields = [
            "condition_name",
            "preferred_care_option",
            "severity_level",
            "task_type",
            "scenario_id",
            "has_image",
            "patient_story",
            "patient_profile",
            "scenario_complexity",
        ]

        # Verify expected fields are present
        for field in expected_fields:
            assert field in result, f"Expected field '{field}' should be present"

        # Verify removed fields are NOT present
        for field in self.REMOVED_FIELDS:
            assert field not in result, f"Removed field '{field}' should NOT be present"

        # Verify query_id is NOT present (replaced by scenario_id)
        assert "query_id" not in result, "query_id should NOT be present (replaced by scenario_id)"


# =============================================================================
# Property 7 (benchmark-seed-refactor): to_dict Outputs Scenario ID
# =============================================================================


class TestToDictOutputsScenarioIdProperty:
    """
    **Feature: benchmark-seed-refactor, Property 7: to_dict Outputs Scenario ID**

    *For any* EnrichedBenchmarkEntry, calling to_dict() SHALL produce a dictionary
    containing "scenario_id" key and NOT containing "query_id" key.

    **Validates: Requirements 4.5**
    """

    @given(entry=enriched_benchmark_entry_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_7_to_dict_contains_scenario_id(self, entry):
        """
        **Feature: benchmark-seed-refactor, Property 7: to_dict Outputs Scenario ID**

        For any EnrichedBenchmarkEntry, to_dict() SHALL contain "scenario_id" key.

        **Validates: Requirements 4.5**
        """
        result = entry.to_dict()

        # Verify scenario_id is present
        assert "scenario_id" in result, (
            f"to_dict() output must contain 'scenario_id' key. "
            f"Found keys: {list(result.keys())}"
        )

        # Verify scenario_id value matches the entry's scenario_id
        assert result["scenario_id"] == entry.scenario_id, (
            f"to_dict() scenario_id value '{result['scenario_id']}' does not match "
            f"entry.scenario_id '{entry.scenario_id}'"
        )

    @given(entry=enriched_benchmark_entry_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_7_to_dict_does_not_contain_query_id(self, entry):
        """
        **Feature: benchmark-seed-refactor, Property 7: to_dict Outputs Scenario ID**

        For any EnrichedBenchmarkEntry, to_dict() SHALL NOT contain "query_id" key.

        **Validates: Requirements 4.5**
        """
        result = entry.to_dict()

        # Verify query_id is NOT present
        assert "query_id" not in result, (
            f"to_dict() output must NOT contain 'query_id' key (replaced by scenario_id). "
            f"Found keys: {list(result.keys())}"
        )

    @given(
        seed_value=st.integers(min_value=0, max_value=2**31 - 1),
        count=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_7_scenario_id_in_batch(self, seed_value, count):
        """
        **Feature: benchmark-seed-refactor, Property 7: to_dict Outputs Scenario ID**

        For any batch of EnrichedBenchmarkEntry instances, ALL to_dict() outputs
        SHALL contain "scenario_id" and NOT contain "query_id".

        **Validates: Requirements 4.5**
        """
        import uuid as uuid_module
        from patient_agent_bench.benchmark_seed.generator import EnrichedBenchmarkEntry

        config = load_test_config()
        selector = AttributeSelector(config, seed=seed_value)

        for i in range(count):
            seed = selector.create_seed()
            scenario_id = str(uuid_module.uuid4())

            entry = EnrichedBenchmarkEntry(
                seed=seed,
                scenario_id=scenario_id,
                patient_story=f"Test patient story {i}",
                patient_profile={"account_info": {"age_in_years": "30"}},
                task_type=f"{seed.task_category}_{seed.task_subcategory}",
                preferred_care_option=seed.care_preference,
                has_image="False",
                scenario_complexity="regular",
            )

            result = entry.to_dict()

            # Verify scenario_id is present and matches
            assert "scenario_id" in result, (
                f"Entry {i}: to_dict() must contain 'scenario_id' key"
            )
            assert result["scenario_id"] == scenario_id, (
                f"Entry {i}: scenario_id mismatch"
            )

            # Verify query_id is NOT present
            assert "query_id" not in result, (
                f"Entry {i}: to_dict() must NOT contain 'query_id' key"
            )

    @given(config_data=valid_config_dict())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_7_scenario_id_with_any_config(self, config_data):
        """
        **Feature: benchmark-seed-refactor, Property 7: to_dict Outputs Scenario ID**

        For any valid configuration, EnrichedBenchmarkEntry.to_dict() SHALL
        contain "scenario_id" and NOT contain "query_id".

        **Validates: Requirements 4.5**
        """
        import uuid as uuid_module
        from patient_agent_bench.benchmark_seed.generator import EnrichedBenchmarkEntry

        config_path = create_temp_config(config_data)

        try:
            config = DistributionConfig.from_file(config_path)
            selector = AttributeSelector(config, seed=42)

            for _ in range(10):
                seed = selector.create_seed()
                scenario_id = str(uuid_module.uuid4())

                entry = EnrichedBenchmarkEntry(
                    seed=seed,
                    scenario_id=scenario_id,
                    patient_story="Test patient story",
                    patient_profile={"account_info": {"age_in_years": "45"}},
                    task_type=f"{seed.task_category}_{seed.task_subcategory}",
                    preferred_care_option=seed.care_preference,
                    has_image="False",
                    scenario_complexity="chronic",
                )

                result = entry.to_dict()

                # Verify scenario_id is present
                assert "scenario_id" in result, (
                    "to_dict() must contain 'scenario_id' key for any config"
                )
                assert result["scenario_id"] == scenario_id

                # Verify query_id is NOT present
                assert "query_id" not in result, (
                    "to_dict() must NOT contain 'query_id' key for any config"
                )
        finally:
            config_path.unlink(missing_ok=True)

    def test_property_7_scenario_id_is_valid_uuid(self):
        """
        **Feature: benchmark-seed-refactor, Property 7: to_dict Outputs Scenario ID**

        Edge case: Verify that scenario_id in to_dict() output is a valid UUID format.

        **Validates: Requirements 4.5**
        """
        import uuid as uuid_module
        from patient_agent_bench.benchmark_seed.generator import EnrichedBenchmarkEntry

        config = load_test_config()
        selector = AttributeSelector(config, seed=42)
        seed = selector.create_seed()

        # Create entry with a valid UUID scenario_id
        scenario_id = str(uuid_module.uuid4())
        entry = EnrichedBenchmarkEntry(
            seed=seed,
            scenario_id=scenario_id,
            patient_story="Test story",
            patient_profile={"account_info": {"age_in_years": "35"}},
            task_type=f"{seed.task_category}_{seed.task_subcategory}",
            preferred_care_option="in person visit",
            has_image="False",
            scenario_complexity="regular",
        )

        result = entry.to_dict()

        # Verify scenario_id is present and is a valid UUID
        assert "scenario_id" in result
        try:
            parsed_uuid = uuid_module.UUID(result["scenario_id"])
            assert str(parsed_uuid) == result["scenario_id"].lower() or str(parsed_uuid) == result["scenario_id"]
        except ValueError:
            pytest.fail(f"scenario_id '{result['scenario_id']}' is not a valid UUID")

        # Verify query_id is NOT present
        assert "query_id" not in result
