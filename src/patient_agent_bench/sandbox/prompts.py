# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Prompts for LLM-based sandbox data generation.
"""

GENERATION_PROMPT = """Generate realistic healthcare data for a patient simulation sandbox.

Location: {city}, {state}

Generate the following as a single JSON object:

1. offices: Generate 4-6 healthcare office locations in or near {city}, {state}
   - Include a mix of facility types (primary care clinic, specialist center, hospital outpatient)
   - Use realistic addresses and phone numbers for the area

2. doctors: Generate 10-15 doctors across all offices
   - Include various specialties: Primary Care, Cardiology, Dermatology, Endocrinology, Orthopedics, Psychiatry, OB/GYN, Neurology, Gastroenterology
   - Each doctor must reference a valid office_id from the offices list
   - Use realistic names with proper titles (Dr. First Last)
   - Include a mix of credentials (MD, DO, NP, PA-C)

JSON Schema:
{schema}

IMPORTANT:
- Return ONLY valid JSON, no explanation or markdown
- All office_id references in doctors must match actual office ids
- Ensure good coverage of specialties for diverse patient scenarios
- Use today's date ({today}) as reference for any dates
"""
