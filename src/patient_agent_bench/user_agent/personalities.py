# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Patient personality type system for simulated patient diversity.

Defines named personality profiles, each with preset trait levels across
7 behavioral dimensions. The benchmark seed selects a personality type
from a weighted distribution, and the user agent prompt adapts accordingly.
"""

from typing import Dict, List


# Trait definitions: trait_name -> {level: behavioral description}
TRAIT_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "cooperation": {
        "low": "Resistant and skeptical. Pushes back on recommendations, questions AI's competence, wants quick fixes.",
        "medium": "Somewhat cooperative. Occasional pushback but generally willing to engage and verify advice.",
        "high": "Cooperative and trusting. Accepts recommendations readily, follows guidance.",
    },
    "anxiety": {
        "low": "Calm and matter-of-fact. Practical approach, minimal worry.",
        "medium": "Moderate concern. Some reassurance-seeking, but generally composed.",
        "high": "Very worried. Worst-case questions. Seeks repeated reassurance.",
    },
    "health_literacy": {
        "low": "Simple language. Misunderstands medical terms. May offer incorrect self-diagnoses.",
        "medium": "Mostly simple language with occasional medical terms.",
        "high": "Uses medical terminology correctly. Asks informed, specific questions.",
    },
    "patience": {
        "low": "Very impatient. Wants the bottom line immediately. Gets frustrated with lengthy explanations.",
        "medium": "Moderate patience. Occasionally wants quicker answers.",
        "high": "Patient and willing to discuss details at length.",
    },
    "clarity": {
        "low": "Vague about symptoms. Forgets details, may contradict self. Gets descriptions wrong.",
        "medium": "Moderate detail. Occasionally forgets minor things. Approximate descriptions.",
        "high": "Clear, specific, and accurate details about symptoms, timeline, and history.",
    },
    "urgency": {
        "low": "Downplays symptoms. Delays seeking care. Minimizes severity.",
        "medium": "Moderate concern. Willing to seek care but not alarmed.",
        "high": "High urgency. Wants immediate answers. Presses for fastest resolution.",
    },
    "communication": {
        "low": "Formal tone, proper grammar. Brief responses, 1-2 short sentences. No mention of practical barriers.",
        "medium": "Mix of formal and informal. 2-3 sentences. May mention scheduling or access concerns.",
        "high": "Casual language, abbreviations, minor typos. 3-5 sentences with context. Mentions cost, insurance, or access worries.",
    },
}

# Named personality profiles: type_name -> {trait: level}
PERSONALITY_TYPES: Dict[str, Dict[str, str]] = {
    "cooperative": {
        "cooperation": "high",
        "anxiety": "low",
        "health_literacy": "high",
        "patience": "high",
        "clarity": "high",
        "urgency": "medium",
        "communication": "medium",
    },
    "anxious": {
        "cooperation": "high",
        "anxiety": "high",
        "health_literacy": "medium",
        "patience": "medium",
        "clarity": "medium",
        "urgency": "high",
        "communication": "high",
    },
    "terse": {
        "cooperation": "low",
        "anxiety": "low",
        "health_literacy": "medium",
        "patience": "low",
        "clarity": "medium",
        "urgency": "medium",
        "communication": "low",
    },
    "confused": {
        "cooperation": "high",
        "anxiety": "medium",
        "health_literacy": "low",
        "patience": "high",
        "clarity": "low",
        "urgency": "low",
        "communication": "medium",
    },
    "skeptical": {
        "cooperation": "low",
        "anxiety": "medium",
        "health_literacy": "high",
        "patience": "medium",
        "clarity": "high",
        "urgency": "medium",
        "communication": "low",
    },
    "stoic": {
        "cooperation": "medium",
        "anxiety": "low",
        "health_literacy": "medium",
        "patience": "high",
        "clarity": "high",
        "urgency": "low",
        "communication": "low",
    },
}

PERSONALITY_TYPE_NAMES: List[str] = list(PERSONALITY_TYPES.keys())


def get_personality_prompt(personality_type: str) -> str:
    """Format a personality type's traits as a prompt section for the user agent.

    Args:
        personality_type: Name of the personality type (e.g., "cooperative")

    Returns:
        XML-formatted personality traits section for the prompt

    Raises:
        KeyError: If personality_type is not recognized
    """
    if personality_type not in PERSONALITY_TYPES:
        available = ", ".join(sorted(PERSONALITY_TYPES.keys()))
        raise KeyError(
            f"Unknown personality type '{personality_type}'. Available: {available}"
        )

    traits = PERSONALITY_TYPES[personality_type]
    lines = [f'<personality-traits type="{personality_type}">']
    for trait, level in traits.items():
        description = TRAIT_DEFINITIONS.get(trait, {}).get(level, "")
        if description:
            lines.append(f'  <{trait} level="{level}">{description}</{trait}>')
    lines.append("</personality-traits>")
    return "\n".join(lines)
