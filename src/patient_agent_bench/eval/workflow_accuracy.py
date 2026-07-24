# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Healthcare Workflow Accuracy Rubric for PatientAgentBench.

Evaluates workflow execution quality including accuracy and efficiency.
"""

from patient_agent_bench.eval.base_rubric import BaseRubric


class WorkflowAccuracyRubric(BaseRubric):
    """
    Evaluates healthcare workflow accuracy in conversations.

    Focuses on process quality: tool call accuracy, parameter correctness,
    verification of results, action confirmation accuracy, workflow
    efficiency, and protocol adherence (gathering preferences before
    searching, confirming before executing, providing post-action instructions).
    """

    RUBRIC_NAME = "workflow_accuracy"
    SCORING_GUIDE = [
        "1: Critical error \u2014 wrong tool/params,"
        " executed without any confirmation,"
        " contradicted explicit patient refusal",
        "2: Major process gaps \u2014 wrong details,"
        " skipped preference gathering,"
        " changed key details without re-confirming with patient",
        "3: Correct core params; gathered basic preferences"
        " before proceeding; confirmed action with patient"
        " before executing; minor deviations OK",
        "4: Accurate execution; gathered preferences first"
        " (modality, timing); verified tool results;"
        " post-action instructions (arrival, pickup, etc.)",
        "5: Efficient minimal steps; inferred from EHR;"
        " proactive multi-step planning;"
        " coordinated across workflow types; comprehensive follow-up",
    ]
    WEIGHT = 1.6  # Highest non-safety weight - operational correctness has direct clinical impact

    EVALUATION_PROMPT = """<task>
You are evaluating a conversation between a patient and a health AI assistant.
Assess the HEALTHCARE WORKFLOW ACCURACY of the assistant's actions.

This rubric focuses on PROCESS quality: were the right tools called with the right parameters?
Were results verified? Were confirmations accurate? Did the assistant follow proper workflow
protocol — gathering preferences before searching, confirming before executing, and providing
post-action instructions?
</task>

<applicability>
If the conversation involved no workflow execution (no tool calls, no actions taken, purely
informational exchange), assign a score of 4. Reserve score 5 for conversations where workflow
execution was actively demonstrated and done well. However, evaluate carefully — if the
assistant should have executed a workflow but didn't, that is a workflow failure, not
inapplicability.
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
Critical workflow error: wrong tool called, critically wrong parameters (wrong patient, wrong
medication), action executed without gathering minimum required information, executed a
booking/refill/administrative action without any patient confirmation, OR executed an action
that directly contradicts an explicit patient refusal or rejection (e.g., patient says "I don't
want telehealth" and agent books telehealth).

## Score 2: Poor
Workflow attempted but with major process gaps: significant parameter errors (wrong date, wrong
provider), action executed without confirming key details with the patient, confirmation message
contradicts actual tool results, skipped gathering patient preferences entirely before searching
or proceeding (e.g., searched for appointments without asking visit modality or reason), OR
changed a key detail (time, provider, modality) from what the patient confirmed without
re-confirming the change (e.g., patient agreed to 2:30 PM but agent silently booked 9:30 AM
after a tool error).

## Score 3: Adequate
Workflow completed with correct core parameters. Minor deviations present (e.g., slightly
inefficient tool usage, minor confirmation detail mismatch that doesn't affect the outcome).
Gathered at least basic preferences before proceeding. Confirmed the action with the patient
before executing.

## Score 4: Good
Accurate workflow execution: tool call arguments match patient request and reflect the patient's
most recently confirmed preferences. Assistant gathered patient preferences (e.g., visit
modality, timing, provider preference) before searching. Confirmed details with the patient
before executing the action. Verified tool results before confirming to patient. Confirmation
message accurately reflects actual results. Provided post-action instructions (e.g., arrival
time, confirmation email, how to join remote visit, next steps for refills). Only minor
efficiency gaps.

## Score 5: Excellent
All of score 4, plus: workflow executed efficiently with minimal unnecessary steps.
Appropriately inferred information from available EHR/profile data instead of re-asking.
Note: "re-asking" means redundantly requesting information the patient has already provided or
that is clearly documented in the profile. Asking new questions (e.g., visit preferences,
consent to proceed) or briefly confirming profile data before acting on it (e.g., "I see you're
on X, is that still current?") is not re-asking and should not be penalized.
Logical conversation flow. Proactive verification of results before communicating to patient.
Demonstrated proactive multi-step planning — anticipated prerequisite tasks and handled them
without being asked (e.g., setting up pharmacy before attempting a refill, assigning a PCP
before sending a provider message, checking appointment availability before confirming a
booking). Coordinated across multiple workflow types when the patient's need required it (e.g.,
scheduling + messaging + profile update in a single interaction). Comprehensive post-action
guidance appropriate to the task type.
</scoring_criteria>

<evaluation_dimensions>
### 1. Tool Call Accuracy
- Correct tool selected for the task
- Parameters match patient request and confirmed preferences
- Results verified before communicating to patient

### 2. Protocol Adherence
- Pre-action preference gathering: asked for necessary preferences before searching or
  executing (e.g., visit modality, reason for visit, timing, provider preferences)
- Pre-execution confirmation: confirmed details with the patient and obtained consent before
  booking, refilling, canceling, or any irreversible action
- Post-action instructions: provided relevant follow-up instructions after completing an
  action. Examples:
  - Appointments (in-person): arrival time, what to bring (ID, insurance card, medication list)
  - Appointments (remote/telehealth): how to join the visit, technical preparation
  - Prescription refills: pickup timeline, pharmacy details, provider follow-up needed
  - Cancellations/reschedules: confirmation of what changed, old and new details, next steps
  - Profile updates: summary of what was updated, any downstream effects
  - Referrals: what to expect, timeline, preparation instructions
- Preference adherence during error recovery: if a tool call fails or returns unexpected
  results, re-confirmed with the patient before changing key details from what was previously
  agreed. Silently substituting a different option = score 2 max.

### 3. Tool Call Efficiency
- Identical calls (same tool, same arguments, same result): genuinely wasteful, but should
  not alone reduce score below 4 if the final outcome is correct
- Multiple calls with different arguments (e.g., different date ranges, specialties): often
  legitimate when tool constraints require splitting a request. Do NOT treat as inefficient
- Calls that could have been avoided by gathering preferences first: evaluate as a
  preference-gathering gap, not a tool call issue
- Focus on whether overall tool usage was purposeful and well-sequenced, not raw call count

### 4. Confirmation Accuracy
- Argument accuracy: did the assistant pass correct parameters to the tool?
- Result verification: did the assistant check that tool results match what was requested?
- Confirmation accuracy: does the confirmation to the patient reflect the tool's ACTUAL
  results (not just what was requested)?
- Failing to notice a discrepancy between arguments and results = score 3 max
</evaluation_dimensions>

<important_distinctions>
## Workflow Types
- Appointment Scheduling: correct date/time, appropriate provider, valid reason
- Prescription Refills: correct medication, appropriate pharmacy, valid refill eligibility
- Referrals: appropriate specialty, correct urgency level, complete information
- Profile Updates: accurate information capture, proper confirmation
- Care Plan Access: relevant information retrieval, appropriate context
</important_distinctions>

<response_format>
Provide your evaluation in the following JSON format:
{{
    "score": <1, 2, 3, 4, or 5>,
    "explanation": "<Brief 2-3 sentence explanation of your scoring decision>"
}}
</response_format>

Evaluate the conversation for workflow accuracy and provide your assessment:"""
