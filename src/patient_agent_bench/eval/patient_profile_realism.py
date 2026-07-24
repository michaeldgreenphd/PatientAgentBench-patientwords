# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Patient Profile Realism dimension for human annotation.

This is NOT an LLM-as-judge rubric — it has no EVALUATION_PROMPT and is not
registered in rubrics.py. It exists solely to co-locate the SCORING_GUIDE
with the other evaluation dimensions.
"""


class PatientProfileRealismDimension:
    """Evaluates how realistic the generated patient profile is."""

    RUBRIC_NAME = "patient_profile_realism"
    SCORING_GUIDE = [
        "1: Implausible or internally contradictory profile;"
        " clinically impossible combinations",
        "2: Major realism gaps \u2014 unlikely demographics,"
        " inconsistent medical history, unrealistic medication regimen",
        "3: Adequate \u2014 mostly believable patient;"
        " minor inconsistencies that don't undermine overall realism",
        "4: Realistic clinical details; coherent medical history;"
        " plausible demographics, medications, and care team",
        "5: Fully realistic patient \u2014"
        " indistinguishable from a real EHR profile;"
        " nuanced clinical details",
    ]
