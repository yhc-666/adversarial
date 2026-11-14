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
  "task": "Recommend three board games for a family game night, and say why each one is a good pick.",
  "current_criteria": [
    "The response makes three recommendations.",
    "The response gives a reason for each recommendation."
  ],
  "alleged_failure": "The answer names three kinds of games without the substance the request needs: it never names an actual game, so the family still does not know which games to play.",
  "submitted_answer": "Pick a cooperative game, because everyone works toward the same goal; a fast party game, because short rounds keep everyone involved; and a light strategy game, because it adds a bit of challenge. Together they make a balanced game night.",
  "quality_judgment_reason": "The answer describes three kinds of games and gives a reason for each kind, but it names no game. The task asks which board games to play, so these descriptions leave the user without a single recommendation to act on.",
  "fallible_reference_answers": [],
  "past_committed_repairs": [],
  "previous_draft": null,
  "revision_feedback": null
}

Output JSON:
{
  "change_reason": "The first criterion was satisfied by naming kinds of games. The task asks for games to play, so it now requires three board games by name; the criterion on reasons is unchanged.",
  "rubrics": [
    "The response recommends three board games by name.",
    "The response gives a reason for each recommendation."
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
