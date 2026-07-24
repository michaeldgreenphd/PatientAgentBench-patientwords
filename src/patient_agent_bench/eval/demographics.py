# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Shared utilities for deriving demographic attributes from patient data."""

from typing import Optional


def derive_age_group(age: Optional[int]) -> str:
    """Derive age group from age in years."""
    if age is None:
        return "unknown"
    if age <= 17:
        return "pediatric"
    elif age <= 35:
        return "young_adult"
    elif age <= 64:
        return "middle_aged"
    return "senior"


def derive_gender_identity(sex: str, gender: str) -> str:
    """Derive gender identity category from sex and gender fields."""
    sex = sex.lower().strip()
    gender = gender.lower().strip()
    if gender in ("woman", "girl"):
        gender = "female"
    if gender in ("man", "boy"):
        gender = "male"
    if gender == "non-binary":
        return "non_binary"
    if "transgender" in gender:
        # Check female markers first: "man"/"male" are substrings of
        # "woman"/"female", so matching male first misclassifies trans women.
        if "woman" in gender or "female" in gender:
            return "transgender_female"
        return "transgender_male"
    if not gender:
        return "unknown"
    if sex == gender:
        return f"cisgender_{gender}"
    if sex and gender:
        return f"transgender_{gender}"
    return "unknown"
