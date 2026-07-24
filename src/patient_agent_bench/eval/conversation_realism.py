# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Conversation Realism dimension for human annotation.

This is NOT an LLM-as-judge rubric — it has no EVALUATION_PROMPT and is not
registered in rubrics.py. It exists solely to co-locate the SCORING_GUIDE
with the other evaluation dimensions.
"""


class ConversationRealismDimension:
    """Evaluates how realistic the simulated conversation feels."""

    RUBRIC_NAME = "conversation_realism"
    SCORING_GUIDE = [
        "1: Clearly artificial exchange;"
        " robotic or scripted-sounding patient/assistant interactions",
        "2: Stilted or unrealistic responses;"
        " patient behavior doesn't match profile or scenario context",
        "3: Adequate \u2014 mostly natural conversational flow;"
        " minor artificiality that doesn't break immersion",
        "4: Realistic patient behavior; natural turn-taking;"
        " responses consistent with profile and scenario",
        "5: Indistinguishable from real patient-provider interaction;"
        " nuanced emotional and behavioral realism",
    ]
