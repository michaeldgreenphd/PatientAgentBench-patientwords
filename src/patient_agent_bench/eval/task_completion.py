# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Task Completion Rubric for PatientAgentBench.

Evaluates whether the user's main intent was resolved or addressed.
"""

from patient_agent_bench.eval.base_rubric import BaseRubric


class TaskCompletionRubric(BaseRubric):
    """
    Evaluates task completion in healthcare conversations.

    Focuses on outcome vs. expectation: was the patient's intent resolved?
    """

    RUBRIC_NAME = "task_completion"
    SCORING_GUIDE = [
        "1: Primary intent not addressed; patient abandoned;"
        " no meaningful progress toward resolution",
        "2: Acknowledged intent but did not execute any concrete action;"
        " task largely unresolved or irrelevant solution provided",
        "3: Partially addressed with concrete action (not just deferral);"
        " core need acknowledged with meaningful step toward resolution."
        " Non-tool resolutions (e.g., clinical guidance, emergency direction) count as completion",
        "4: Task completed through correct workflow"
        " (actually executed via tools, not just acknowledged);"
        " minor gaps in confirmation/follow-up",
        "5: Completed with confirmation matching actual tool results;"
        " all details communicated;"
        " secondary/implicit needs addressed; follow-up offered",
    ]
    WEIGHT = 1.0  # Demoted - most models ceiling here; software functionality metric

    EVALUATION_PROMPT = """<task>
You are evaluating a conversation between a patient and a health AI assistant.
Assess whether the patient's main intent was resolved or addressed.

This rubric focuses on the OUTCOME: did the patient get what they needed?
</task>

<applicability>
If the conversation had no identifiable patient task or intent to resolve (e.g., the patient
only greeted the assistant with no follow-up), assign a score of 4. Reserve score 5 for
conversations where the task was actively completed with full confirmation. However, evaluate
carefully — most conversations have at least an implicit intent.
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
Patient's primary intent was not addressed. Patient abandoned the conversation. Or the agent
failed to make meaningful progress toward resolution.

## Score 2: Poor
Agent acknowledged the intent but did not execute any concrete action toward resolution. Task
remains largely unresolved. Or agent provided an irrelevant solution. The key distinction from
score 1 is that the agent at least recognized what the patient wanted, but from score 3 is that
no tangible action was taken — only verbal acknowledgment or vague deferral (e.g., "I'll let
your provider know" without actually doing it).

## Score 3: Adequate
Task was partially addressed but the outcome does not fully match the original request. The
patient received some value through an alternative approach, but the alternative must involve a
concrete action (not just "I'll send a message to your provider" or "I'll let someone know").
The patient's core need was acknowledged with a meaningful step toward resolution.

NOTE on non-tool task resolution: Not all clinically appropriate resolutions require tool calls.
When the agent determines that the patient's situation is best addressed through direct guidance
(e.g., directing to emergency services, advising a specific self-care action, providing
clinical education that fully resolves the patient's question), this constitutes meaningful task
completion and should be scored at 4 or higher if the guidance is accurate, complete, and
addresses the patient's core need.

## Score 4: Good
Primary task completed through the correct workflow (e.g., appointment actually scheduled via
tool, prescription actually refilled, profile actually updated). Minor gaps in confirmation or
follow-up details. The task was not merely acknowledged or deferred — it was executed.

## Score 5: Excellent
All of score 4, plus: confirmation provided to the patient accurately reflects actual tool
results (not just what was requested). All relevant details communicated (time, provider,
location, medication, dosage). Secondary or implicit patient needs were identified and addressed
(e.g., related profile updates, care coordination steps, follow-up scheduling, prerequisite
tasks). Appropriate follow-up or next steps offered that go beyond the immediate request.
</scoring_criteria>

<evaluation_dimensions>
### 1. Primary Intent Resolution
- Identify the primary intent from the initial message and any clarifications
- Consider the scenario context
- Does the end state match what the patient originally requested?
- Was the patient's need ultimately met, even if through recovery or alternatives?

### 2. Confirmation Quality (scores 4-5)
- Does the agent's confirmation accurately reflect the actual outcome?
- Are all relevant details (time, provider, location, medication) communicated?
- Are appropriate next steps or follow-up expectations provided?

### 3. Secondary Task Handling (differentiates score 4 from 5)
- Did the agent identify and address needs beyond the primary request?
- Examples: updating a profile after booking with a new provider, setting up pharmacy before
  processing a prescription, coordinating across care team members, offering to handle related
  administrative tasks
- A model that completes only the stated task gets 4; a model that anticipates and addresses
  the broader patient need gets 5
</evaluation_dimensions>

<important_distinctions>
## Conversation Drop-Off
If the conversation ends abruptly (patient drops off or conversation is truncated) after the
patient made a request but BEFORE the agent had an opportunity to respond or act on it, do NOT
penalize the agent for that unresolved request. Evaluate task completion based only on what the
agent had the opportunity to complete. A patient dropping off mid-conversation is not an agent
failure.

## System and Tool Limitations (Infeasibility)
When a task cannot be fully completed due to system or tool limitations (e.g., no tool exists
to perform a specific action, the sandbox lacks the required data, or the system returns an
error for a valid request), evaluate the agent's response to the limitation rather than
penalizing for the incomplete task itself. Specifically:
- If the agent recognizes the limitation, explains it to the patient, and provides a concrete,
  clinically appropriate alternative or workaround (e.g., "I can't notify your dialysis center
  directly, but please ask the ER team to contact them"), this should be scored the same as if
  the task were completed — the agent did everything within its capability.
- If the agent encounters a system limitation and simply gives up or provides only a vague
  deferral without a concrete alternative, that is a genuine gap and should be scored lower.
- Multiple infeasible sub-tasks handled with appropriate alternatives should not compound into
  a lower score — judge each workaround on its own merit.
</important_distinctions>

<response_format>
Provide your evaluation in the following JSON format:
{{
    "score": <1, 2, 3, 4, or 5>,
    "explanation": "<Brief 2-3 sentence explanation of your scoring decision>"
}}
</response_format>

Evaluate the conversation and provide your assessment:"""
