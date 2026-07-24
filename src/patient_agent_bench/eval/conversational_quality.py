# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Conversational Quality Rubric for PatientAgentBench.

Evaluates conversational qualities including brevity, informativeness,
succinctness, clarity, response length appropriateness, and tone neutrality.
"""

from patient_agent_bench.eval.base_rubric import BaseRubric


class ConversationalQualityRubric(BaseRubric):
    """
    Evaluates conversational quality in healthcare conversations.

    Assesses brevity, informativeness, succinctness, clarity,
    response length appropriateness, and tone neutrality (non-judgmental,
    non-presumptuous empathy) across the full conversation.
    """

    RUBRIC_NAME = "conversational_quality"
    SCORING_GUIDE = [
        "1: Uninformative, excessively verbose obscuring key info,"
        " or dismissive; overtly judgmental about patient choices/lifestyle",
        "2: Frequently too verbose/unclear; important info buried;"
        " repetition/rambling; presumptuous or unnatural empathy;"
        " batched multiple questions",
        "3: Generally addresses core question; occasional verbosity;"
        " overall understandable;"
        " no judgmental or presumptuous language",
        "4: Consistently concise without sacrificing completeness;"
        " minimal repetition; non-judgmental;"
        " questions presented one at a time",
        "5: Adapts length/depth to complexity;"
        " medical terms explained when needed;"
        " structured for comprehension; tone-neutral throughout",
    ]
    WEIGHT = 0.9

    EVALUATION_PROMPT = """<task>
You are evaluating a conversation between a patient and a health AI assistant.
Assess the CONVERSATIONAL QUALITY of the assistant's responses, focusing on brevity,
informativeness, succinctness, clarity, response length appropriateness, and tone neutrality.

Evaluate the cumulative pattern across the entire conversation, not isolated responses. Weight
failures in critical health information delivery more heavily. Consider whether verbosity or
brevity is appropriate to the situation.
</task>

<applicability>
If the conversation is too short or lacks sufficient assistant responses to meaningfully assess
conversational quality, assign a score of 4. Reserve score 5 for conversations where
conversational quality was actively demonstrated.
</applicability>

<patient_profile>
{user_profile}
</patient_profile>

<scenario>
{scenario}
</scenario>

<conversation>
{conversation}
</conversation>

<scoring_criteria>
Scores are cumulative — each level assumes all requirements of lower levels are also met.

## Score 1: Fail
Responses are consistently uninformative, excessively verbose to the point of obscuring key
information, or so brief they appear dismissive. Critical health questions go inadequately
answered due to poor communication quality. OR the assistant is overtly judgmental about the
patient's choices, lifestyle, or health decisions.

## Score 2: Poor
Responses are frequently too verbose, too brief, or unclear. Important information is buried in
unnecessary detail, or critical details are missing. Noticeable repetition or rambling across
turns. OR the assistant uses judgmental language, makes assumptions about the patient's feelings,
or applies presumptuous empathy (e.g., "You must be so worried" when the patient hasn't
expressed worry). OR the assistant uses unnatural or formulaic reassurance (e.g., "I'm going to
take care of this with compassion and urgency in mind", "I want you to know you're not alone
in this journey"). OR the assistant consistently asks multiple questions at once when they should
be presented one at a time.

## Score 3: Adequate
Responses generally address the core question with relevant information. Occasional verbosity
or minor clarity issues, but overall understandable. Simple questions receive reasonably direct
answers. No overtly judgmental or presumptuous language.

## Score 4: Good
All of score 3, plus: responses are consistently concise without sacrificing completeness. Key
facts and actionable guidance are provided efficiently. Minimal repetition across turns.
Language is non-judgmental and does not make assumptions about the patient's feelings or
attitudes toward their health situation. When multiple questions are needed, they are generally
presented one at a time rather than in a batch.

## Score 5: Excellent
All of score 4, plus: responses adapt length and depth to query complexity throughout. Medical
terminology is explained when needed but not over-explained. Information is structured for easy
comprehension (e.g., numbered steps, clear grouping). Tone is consistently neutral and
non-judgmental. Empathy mirrors the patient's expressed emotions without making assumptions
about how they feel.
</scoring_criteria>

<evaluation_dimensions>
### 1. Brevity
- Are responses appropriately concise? Is there minimal repetition?

### 2. Informativeness
- Do responses contain valuable, relevant information?

### 3. Succinctness
- Does the assistant get to the main point efficiently?

### 4. Clarity
- Is information presented clearly and unambiguously?

### 5. Response Length Appropriateness
- Does length match query complexity?

### 6. Tone Neutrality
- Does the assistant use neutral, inclusive language without assumptions about the patient's
  feelings toward their health situation?
- Does the assistant let the patient lead emotional framing and mirror their tone rather than
  projecting emotions?
- Is the assistant non-judgmental about the patient's choices, lifestyle, or adherence to
  medical advice?

### 7. Question Pacing
- When the assistant needs to ask the patient multiple questions, does it present them one at
  a time and wait for the patient's answer before asking the next?
- Bombarding the patient with multiple questions in a single response reduces clarity and can
  feel overwhelming
- This applies to clinical assessment questions, preference gathering, and any
  information-gathering exchanges
- Exception: brief confirmation summaries (e.g., "Just to confirm: date X, provider Y,
  time Z — does that look right?") are not penalized
</evaluation_dimensions>

<response_format>
Provide your evaluation in the following JSON format:
{{
    "score": <1, 2, 3, 4, or 5>,
    "explanation": "<Brief 2-3 sentence explanation of your scoring decision>"
}}
</response_format>

Evaluate the conversation for conversational quality and provide your assessment:"""
