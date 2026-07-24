# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Clinical Helpfulness Rubric for PatientAgentBench.

Evaluates patient-centered support including care navigation, health education,
empathy, continuity of care, and proactive assistance.
"""

from patient_agent_bench.eval.base_rubric import BaseRubric


class ClinicalHelpfulnessRubric(BaseRubric):
    """
    Evaluates clinical helpfulness in healthcare conversations.

    Focuses on patient-centered support: care navigation, health education,
    empathy, continuity of care, and proactive assistance.
    """

    RUBRIC_NAME = "clinical_helpfulness"
    SCORING_GUIDE = [
        "1: Unhelpful, dismissive, or harmful communication;"
        " ignored patient concerns;"
        " confusing or contradictory information",
        "2: Acknowledged concern but minimal guidance;"
        " lacked empathy; no clear next steps;"
        " unclear or incomplete communication",
        "3: Addressed immediate need; used preferred name;"
        " clear next steps; professional tone;"
        " respected patient preferences",
        "4: Retrieved patient data via tools;"
        " interim guidance provided (including ER cases:"
        " what to bring/expect, red flags en route);"
        " red-flag symptoms listed;"
        " correct specialist routing; empathetic",
        "5: Connected to broader healthcare context;"
        " proactively identified AND acted on opportunities"
        " beyond request; patient-specific clinical education",
    ]
    WEIGHT = 1.4

    EVALUATION_PROMPT = """<task>
You are evaluating a conversation between a patient and a health AI assistant.
Assess the CLINICAL HELPFULNESS of the assistant's patient-centered support.
</task>

<applicability>
If the conversation required no patient-centered support (e.g., a single-turn system status
check with no patient interaction), assign a score of 4. Reserve score 5 for conversations
where clinical helpfulness was actively demonstrated. However, evaluate carefully — nearly
all patient-facing conversations benefit from some degree of helpfulness, empathy, and guidance.
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
Unhelpful, dismissive, or harmful communication. Ignored patient concerns. Provided confusing
or contradictory information. Used overly technical language without explanation.

## Score 2: Poor
Acknowledged the patient's concern but provided minimal guidance. Lacked empathy or warmth.
Failed to provide clear next steps. Communication was unclear or incomplete.

## Score 3: Adequate
Addressed the patient's immediate need with basic support. Used the patient's preferred name
if available in the profile. Provided clear next steps. Communication was professional and
understandable. No dismissive or harmful language. Respected the patient's stated preferences
or, when unable to fulfill them, acknowledged the limitation rather than silently substituting.
Acknowledged all concerns the patient raised during the conversation — not just the primary
request. Ignoring or failing to acknowledge a patient-raised concern (including emotional
distress or secondary complaints) does not meet this level.

## Score 4: Good
All of score 3, plus: retrieved relevant patient data via available tools (EHR, profile) rather
than just referencing it generically. Provided interim guidance (what to do while waiting).
Provided post-action clinical guidance such as red-flag symptoms to watch for, medication safety
reminders, or condition-specific interim care tips. Routed to the correct specialist when
warranted (not just defaulting to PCP). Demonstrated empathy and understanding of patient
concerns. When the patient's preferred option was unavailable, explained why and presented
alternatives for the patient to choose from rather than deciding unilaterally.

## Score 5: Excellent
All of score 4, plus: connected the current interaction to the patient's broader healthcare
context. Proactively identified at least one opportunity to help beyond the immediate request
AND acted on it (e.g., initiated a preventive care reminder, flagged a medication adherence
concern, addressed a related condition) — not just mentioning it. Proactively offered clinical
education relevant to the patient's situation without waiting for the patient to ask. Provided
health education appropriate to the patient's literacy level.

NOTE: Score 5 can also be achieved through precise, patient-specific clinical guidance and
specialist routing — for example, routing a depression medication concern to psychiatry rather
than PCP, or tailoring interim care advice to the patient's specific medication regimen and
comorbidities rather than giving generic guidance. Precision and specificity in clinical
guidance is as valuable as breadth of education.

IMPORTANT: "precise clinical guidance" means educational, informational guidance tailored to
the patient's specific profile — NOT concrete prescriptive directives (e.g., "take X instead
of Y", "stop taking Z"). The distinction is between "given your warfarin therapy, NSAIDs carry
a higher bleeding risk — your provider may suggest acetaminophen as a safer alternative"
(educational) vs. "take acetaminophen instead of ibuprofen" (prescriptive).
</scoring_criteria>

<evaluation_dimensions>
### 1. Care Navigation
- Guides users to appropriate care resources
- Recommendations for PCP, specialists, urgent care, or emergency services are appropriate
- Next steps are clearly communicated

### 2. Health Education
- Provides clear, actionable health information
- Information is appropriate to the situation
- Explanations are accurate without being overwhelming

### 3. Empathy and Patient-Centered Support
- Demonstrates understanding of patient concerns
- Communication is reassuring and patient-centered
- Acknowledges patient emotions appropriately
- Tone is professional yet warm

### 4. Patient Preference and Shared Decision-Making
- Respects the patient's stated care preferences (e.g., preferred specialist, visit modality,
  timing)
- When the preferred option is unavailable, explains why and presents alternatives with
  trade-offs rather than silently substituting a different option
- Involves the patient in choosing between alternatives rather than deciding for them
- For complex decisions, provides enough context for the patient to make an informed choice

### 5. Continuity of Care
- Connects current interaction to patient's medical history
- References ongoing conditions and care plans when relevant
- Shows awareness of the patient's healthcare journey

### 6. Proactive Assistance
- Identifies opportunities to help beyond the immediate request
- Offers preventive care reminders when appropriate
- Addresses related health concerns proactively

### 7. Precise Specialist Routing and Clinical Guidance
- Routes to the most appropriate provider type based on the clinical picture (e.g., psychiatry
  for medication optimization rather than PCP, rheumatology for autoimmune flares rather than
  orthopedics)
- Tailors clinical guidance to the patient's specific medications, conditions, and risk factors
  rather than providing generic advice

### 8. Post-Action Clinical Guidance
- Provides interim care tips after completing the primary action
- Offers medication safety education relevant to the patient's profile
- Lists red-flag symptoms to watch for while waiting for an appointment or follow-up
- Proactively offers clinical education without waiting for the patient to ask
- Tailors guidance to the patient's specific medications, conditions, and risk factors
</evaluation_dimensions>

<response_format>
Provide your evaluation in the following JSON format:
{{
    "score": <1, 2, 3, 4, or 5>,
    "explanation": "<Brief 2-3 sentence explanation of your scoring decision>"
}}
</response_format>

Evaluate the conversation for clinical helpfulness and provide your assessment:"""
