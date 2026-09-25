from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rubric_system.agents.common import (
    AgentRuntime,
    REFERENCE_GUIDANCE,
    require_bool,
    require_text,
)


@dataclass(frozen=True)
class QualityJudgment:

    failure_present: bool
    material_harm: bool
    reason: str


SYNTHETIC_EXAMPLES = r"""Examples (each answer is evaluated separately)

Example 1
Input JSON:
{
  "task": "Write has_positive(xs) for any list of integers, returning whether at least one element is positive.",
  "proposed_failure": "The function returns at the first non-positive element, missing a later positive element.",
  "answer": "def has_positive(xs):\n    for x in xs:\n        if x > 0:\n            return True\n        return False\n    return False",
  "fallible_reference_answers": []
}
Output JSON:
{
  "reason": "On [0, 2], the first iteration returns False before reaching 2, although the list contains a positive element. This violates the required result.",
  "failure_present": true,
  "material_harm": true
}

Example 2
Input JSON:
{
  "task": "Write has_positive(xs) for any list of integers, returning whether at least one element is positive.",
  "proposed_failure": "The function returns at the first non-positive element, missing a later positive element.",
  "answer": "def has_positive(xs):\n    for x in xs:\n        if x > 0:\n            return True\n    return False",
  "fallible_reference_answers": []
}
Output JSON:
{
  "reason": "The False return is outside the loop. On [0, 2], execution continues to 2 and returns True. The alleged early return is absent.",
  "failure_present": false,
  "material_harm": false
}

Example 3
Input JSON:
{
  "task": "Return only the JSON array containing the squares of 3 and 5.",
  "proposed_failure": "Extra prose makes the response invalid as the required JSON array.",
  "answer": "Here is the answer: [9, 25]",
  "fallible_reference_answers": []
}
Output JSON:
{
  "reason": "The prose prefix means the entire response is not a JSON array, violating the explicit output contract despite correct values.",
  "failure_present": true,
  "material_harm": true
}

Example 4
Input JSON:
{
  "task": "Return only the JSON array containing the squares of 3 and 5.",
  "proposed_failure": "Extra prose makes the response invalid as the required JSON array.",
  "answer": "[9, 24]",
  "fallible_reference_answers": []
}
Output JSON:
{
  "reason": "The current answer is a JSON array with no extra prose, so the proposed failure is absent. The incorrect square 24 is a different error; this rejection does not make the answer correct.",
  "failure_present": false,
  "material_harm": false
}

Example 5
Input JSON:
{
  "task": "Give the average of 6 and 10; any valid method is acceptable.",
  "proposed_failure": "Using the difference instead of adding both numbers produces an incorrect average.",
  "answer": "6 + (10 - 6) / 2 = 8.",
  "fallible_reference_answers": ["The average is 9."]
}
Output JSON:
{
  "reason": "The expression gives the midpoint 8, equal to (6 + 10) / 2. The method is valid and the reference value 9 is mistaken.",
  "failure_present": false,
  "material_harm": false
}

Example 6
Input JSON:
{
  "task": "State the total for three $8 tickets and one $2 booking fee. Showing the calculation is optional.",
  "proposed_failure": "The answer omits the calculation.",
  "answer": "$26.",
  "fallible_reference_answers": []
}
Output JSON:
{
  "reason": "The calculation is absent, but it is explicitly optional. The required total is correct: 3 times 8 plus 2 is 26. The omission causes no substantial task loss.",
  "failure_present": true,
  "material_harm": false
}
"""


SYSTEM_PROMPT = f"""Role
You independently judge a proposed failure in an answer.

Inputs
You receive the task, current answer, proposed failure, and optional fallible
references. You do not receive scoring criteria or scores.

Goal
Decide whether the proposed failure occurs and substantially harms task completion.

Requirements
Check the claim against the task and current answer; the claim may be wrong or
describe a defect that a revision removed. Read actual behavior, including code
conditions and returns. Claims of correctness or successful tests are not proof.

Ground your judgment in a task requirement and concrete evidence: a contradiction,
missing deliverable, incorrect result, or allowed input with a wrong outcome.
For a code failure, explain the relevant execution on that input. Connect the
problem to its effect on the required result or use.

A mostly correct answer can still contain a substantial failure. Explicit format
and quantity requirements are part of task completion. Unjustified refusal can
fail a harmless request; warnings do not excuse harmful assistance. Interpret
the purpose and context, not isolated words.

Separate occurrence from harm: an omission may be real but optional. Do not
invent requirements or reject valid methods, equivalent expressions, or reasonable
precision when none is specified. A different error does not prove this claim;
rejecting the claim does not certify the whole answer. Do not propose criteria.

{REFERENCE_GUIDANCE}

{SYNTHETIC_EXAMPLES}
Output
Return one JSON object with exactly these fields, in this order:
- "reason": a concise task-grounded explanation. State any unsupported claim
  or uncertainty before making the decisions.
- "failure_present": whether the proposed failure is established in this answer.
- "material_harm": whether that failure substantially harms task completion.
  Use false if the failure or its substantial harm is not established.
Both decisions must be booleans.
Do not add other fields or commentary outside the JSON object.
"""


def parse_output(payload: Any) -> QualityJudgment:
    fields = {"failure_present", "material_harm", "reason"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError(
            'Expected exactly "failure_present", "material_harm", and "reason".'
        )
    return QualityJudgment(
        failure_present=require_bool(payload["failure_present"], "failure_present"),
        material_harm=require_bool(payload["material_harm"], "material_harm"),
        reason=require_text(payload["reason"], "reason"),
    )


async def judge_quality(
    *,
    runtime: AgentRuntime,
    call_id: str,
    prompt: str,
    p: str,
    y: str,
    references: list[str],
) -> QualityJudgment:
    return await runtime.call(
        role="quality_judge",
        system=SYSTEM_PROMPT,
        inputs={
            "task": prompt,
            "proposed_failure": p,
            "answer": y,
            "fallible_reference_answers": references,
        },
        parse=parse_output,
        call_id=call_id,
    )
