from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rubric_system.agents.common import (
    AgentRuntime,
    REFERENCE_GUIDANCE,
    RUBRIC_PRINCIPLES,
    require_rubrics,
    require_text,
)


@dataclass(frozen=True)
class RepairProposal:

    change_reason: str
    rubrics: list[str]


SYNTHETIC_EXAMPLES = r"""
Synthetic examples
Each example shows one complete input and its output.

Example 1

Input JSON:
{
  "task": "A parcel weighs 8 kg. Shipping costs $3 per kg plus a $5 fixed fee. State the total and the delivery estimate from the notice: 2–4 days.",
  "current_criteria": [
    "The shipping total is $19.",
    "The delivery estimate is 2–4 days."
  ],
  "alleged_failure": "The shipping total is $19 instead of the required $29.",
  "submitted_answer": "Shipping is $19; delivery takes 2–4 days.",
  "quality_judgment_reason": "The task specifies 8*3+5=$29. The submitted $19 total is incorrect.",
  "fallible_reference_answers": [],
  "past_committed_repairs": [],
  "previous_draft": null,
  "revision_feedback": null
}

Output JSON:
{
  "change_reason": "Correct the shipping total to $29 and retain the supplied 2–4 day delivery estimate.",
  "rubrics": [
    "The shipping total is $29.",
    "The delivery estimate is 2–4 days."
  ]
}

Example 2

Input JSON:
{
  "task": "Using this support policy, give one permitted contact route for urgent requests and one for ordinary requests. Urgent requests may use the hotline or web chat; either route is acceptable. Ordinary requests use email. The hotline number is +1-202-555-0148; including it is optional. Weekend opening hours are not provided.",
  "current_criteria": [
    "The response gives a contact route for urgent requests.",
    "The response directs ordinary requests to email.",
    "If the response gives the hotline number, it is +1-202-555-0148."
  ],
  "alleged_failure": "Email is the wrong contact route for urgent requests.",
  "submitted_answer": "For urgent requests, send an email. For ordinary requests, also use email.",
  "quality_judgment_reason": "The policy permits the hotline or web chat for urgent requests. Sending those requests to email violates the supplied routing rule.",
  "fallible_reference_answers": [],
  "past_committed_repairs": [],
  "previous_draft": null,
  "revision_feedback": null
}

Output JSON:
{
  "change_reason": "Specify the permitted urgent routes, preserving the ordinary route and the conditional phone-number check.",
  "rubrics": [
    "The response directs urgent requests to the hotline or web chat; either route is acceptable.",
    "The response directs ordinary requests to email.",
    "If the response gives the hotline number, it is +1-202-555-0148."
  ]
}
"""


SYSTEM_PROMPT = f"""Role
You repair scoring criteria after an answer passed them and a quality judge
confirmed a task failure.

Inputs
You receive the task, current criteria, answer, failure description, quality
judgment, past repairs, and optional fallible references. Retries include the
previous draft and feedback.

Goal
Make the smallest task-grounded change that detects this failure and similar
failures. Return the complete set.

Requirements
Correct wrong requirements, add missing checks, or clarify vague ones. Merely
mentioning correctness does not resolve a demonstrated gap. Keep unaffected
criterion text unchanged; edit, split, merge, or remove only as needed.

Preserve the valid intent of past repairs without treating them as proof. Use
feedback to revise the previous draft for the same answer and current criteria.
An unchanged draft needs a substantive repair. Changed drafts are scored first;
those that make the answer fail are then reviewed for justified changes.

Rubric properties
Apply these properties to this repair, preserving valid coverage elsewhere.
{RUBRIC_PRINCIPLES}

Reference answers
{REFERENCE_GUIDANCE}

{SYNTHETIC_EXAMPLES}

Output
Return one JSON object with exactly these fields, in this order:
- "change_reason": a short explanation of the failure, the task basis for the
  changes, and how the new criteria detect it.
- "rubrics": the full nonempty array of revised and retained criterion strings.
Do not return edit operations, IDs, weights, scores, or other commentary.
"""


def parse_output(payload: Any) -> RepairProposal:
    if not isinstance(payload, dict) or set(payload) != {"rubrics", "change_reason"}:
        raise ValueError('Expected exactly "rubrics" and "change_reason".')
    return RepairProposal(
        rubrics=require_rubrics(payload["rubrics"]),
        change_reason=require_text(payload["change_reason"], "change_reason"),
    )


async def repair(
    *,
    runtime: AgentRuntime,
    call_id: str,
    prompt: str,
    rubrics: list[str],
    p: str,
    y: str,
    judge_reason: str,
    references: list[str],
    memory: list[dict],
    previous: RepairProposal | None = None,
    feedback: dict | None = None,
) -> RepairProposal:
    return await runtime.call(
        role="repair",
        system=SYSTEM_PROMPT,
        inputs={
            "task": prompt,
            "current_criteria": rubrics,
            "alleged_failure": p,
            "submitted_answer": y,
            "quality_judgment_reason": judge_reason,
            "fallible_reference_answers": references,
            "past_committed_repairs": memory,
            "previous_draft": asdict(previous) if previous is not None else None,
            "revision_feedback": feedback,
        },
        parse=parse_output,
        call_id=call_id,
    )
