from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rubric_system.agents.common import (
    AgentRuntime,
    REFERENCE_GUIDANCE,
    RUBRIC_PRINCIPLES,
    require_bool,
    require_text,
)


@dataclass(frozen=True)
class ReviewResult:

    reason: str
    approved: bool
    issues: list[str]


SYNTHETIC_EXAMPLES = r"""
Synthetic examples
Each example shows one complete input and its output.

Example 1

Input JSON:
{
  "task": "A parcel weighs 8 kg. Shipping costs $3 per kg plus a $5 fixed fee. State the total and the delivery estimate from the notice: 2–4 days.",
  "old_criteria": [
    "The shipping total is $19.",
    "The delivery estimate is 2–4 days."
  ],
  "proposed_criteria": [
    "The shipping total is $29.",
    "The delivery estimate is 2–4 days."
  ],
  "alleged_failure": "The shipping total is $19 instead of the required $29.",
  "submitted_answer": "Shipping is $19; delivery takes 2–4 days.",
  "quality_judgment_reason": "The task specifies 8*3+5=$29. The submitted $19 total is incorrect.",
  "repair_explanation": "Correct the shipping total to $29 and retain the supplied 2–4 day delivery estimate.",
  "fallible_reference_answers": [],
  "old_verification": [
    {
      "criterion": "The shipping total is $19.",
      "reason": "The answer gives the required $19.",
      "verdict": "PASS"
    },
    {
      "criterion": "The delivery estimate is 2–4 days.",
      "reason": "The answer states 2–4 days.",
      "verdict": "PASS"
    }
  ],
  "new_verification": [
    {
      "criterion": "The shipping total is $29.",
      "reason": "The answer gives $19 rather than the required $29.",
      "verdict": "FAIL"
    },
    {
      "criterion": "The delivery estimate is 2–4 days.",
      "reason": "The answer states 2–4 days.",
      "verdict": "PASS"
    }
  ]
}

Output JSON:
{
  "reason": "The new shipping FAIL identifies the submitted $19 error. Replacing the wrong target with 8*3+5=$29 is task-grounded and preserves the delivery requirement.",
  "approved": true,
  "issues": []
}

Example 2

Input JSON:
{
  "task": "Write a Python function is_multiple_of_three(n) for integer n.",
  "old_criteria": [
    "The response supplies a Python function named is_multiple_of_three with parameter n.",
    "For every integer n, including zero and negatives, the function returns True exactly when n is divisible by 3."
  ],
  "proposed_criteria": [
    "The response supplies a Python function named is_multiple_of_three with parameter n.",
    "For every integer n, including zero and negatives, the function returns True exactly when n is divisible by 3.",
    "The function takes abs(n) before testing divisibility."
  ],
  "alleged_failure": "Negative multiples are rejected.",
  "submitted_answer": "def is_multiple_of_three(n): return n % 3 == 0",
  "quality_judgment_reason": "The function needs an absolute-value step to handle negative multiples.",
  "repair_explanation": "Require abs(n) before testing divisibility to address the reported negative-input failure.",
  "fallible_reference_answers": [],
  "old_verification": [
    {
      "criterion": "The response supplies a Python function named is_multiple_of_three with parameter n.",
      "reason": "The named Python function takes n.",
      "verdict": "PASS"
    },
    {
      "criterion": "For every integer n, including zero and negatives, the function returns True exactly when n is divisible by 3.",
      "reason": "The modulo test handles positive, zero, and negative multiples.",
      "verdict": "PASS"
    }
  ],
  "new_verification": [
    {
      "criterion": "The response supplies a Python function named is_multiple_of_three with parameter n.",
      "reason": "The named Python function takes n.",
      "verdict": "PASS"
    },
    {
      "criterion": "For every integer n, including zero and negatives, the function returns True exactly when n is divisible by 3.",
      "reason": "The modulo test handles positive, zero, and negative multiples.",
      "verdict": "PASS"
    },
    {
      "criterion": "The function takes abs(n) before testing divisibility.",
      "reason": "The function does not call abs(n).",
      "verdict": "FAIL"
    }
  ]
}

Output JSON:
{
  "reason": "The actual FAIL is for omitting abs(n), not rejecting negative multiples. At n=-6 the submitted code returns True. The added rule excludes a valid implementation without a task basis.",
  "approved": false,
  "issues": [
    "Remove the mandatory abs(n) step; the observed FAIL does not substantiate the alleged negative-input failure."
  ]
}

Example 3

Input JSON:
{
  "task": "A parcel weighs 8 kg. Shipping costs $3 per kg plus a $5 fixed fee. State the total and the delivery estimate from the notice: 2–4 days.",
  "old_criteria": [
    "The shipping total is $19.",
    "The delivery estimate is 2–4 days."
  ],
  "proposed_criteria": [
    "The shipping total is $29.",
    "The delivery estimate is 2–4 days."
  ],
  "alleged_failure": "The shipping total is $19 instead of the required $29.",
  "submitted_answer": "Shipping is $19; delivery takes 2–4 days.",
  "quality_judgment_reason": "The task specifies 8*3+5=$29. The submitted $19 total is incorrect.",
  "repair_explanation": "Correct the shipping total to $29 and retain the supplied 2–4 day delivery estimate.",
  "fallible_reference_answers": [],
  "old_verification": [
    {
      "criterion": "The shipping total is $19.",
      "reason": "The answer gives the required $19.",
      "verdict": "PASS"
    },
    {
      "criterion": "The delivery estimate is 2–4 days.",
      "reason": "The answer states 2–4 days.",
      "verdict": "PASS"
    }
  ],
  "new_verification": [
    {
      "criterion": "The shipping total is $29.",
      "reason": "$19 is a stated shipping total.",
      "verdict": "PASS"
    },
    {
      "criterion": "The delivery estimate is 2–4 days.",
      "reason": "The answer does not explain why 2–4 days applies to this particular parcel.",
      "verdict": "FAIL"
    }
  ]
}

Output JSON:
{
  "reason": "The only FAIL is on the untouched delivery criterion and adds an unsupported explanation requirement. The shipping error still receives PASS, so this verdict change does not validate the shipping repair.",
  "approved": false,
  "issues": [
    "The repair needs an actual task-grounded FAIL for the shipping-total error; the unrelated delivery verdict change is not evidence that it worked."
  ]
}
"""


SYSTEM_PROMPT = f"""Role
You review revised scoring criteria after the same answer has been graded
against the old and new sets.

Inputs
You receive the task, old and proposed criteria, submitted answer, failure
description, quality judgment, repair explanation, and optional fallible
references. The old and new verification arrays contain each criterion, its
verdict, and its grading reason, in the corresponding set's order.

Goal
Approve a justified repair that catches the confirmed task failure.
Use the actual grading evidence and preserve other valid requirements.

Requirements
Identify a new FAIL that concerns the described failure in the submitted answer.
Explain how a necessary change to the criteria makes that failure detectable.
A changed verdict on an untouched or unrelated criterion does not establish
that the repair worked.

Check the changes and the claimed FAIL against the task and answer. The quality
judgment, repair explanation, and grading reasons may be wrong. Reject unsupported
requirements, restrictions that exclude valid answers, unchanged sets, and drafts
without a substantive repair. Do not treat a grading label alone as proof of repair.

A clearer check of an existing requirement can be a valid repair. If an added
criterion duplicates an existing one, identify the check that should be revised.
Apply the rubric properties below. Focus feedback on the repair and any coverage
it loses; do not request unrelated expansion or cosmetic rewrites.

Rubric properties
{RUBRIC_PRINCIPLES}

Reference answers
{REFERENCE_GUIDANCE}

{SYNTHETIC_EXAMPLES}

Output
Return one JSON object with exactly these fields, in this order:
- "reason": a short explanation connecting the actual FAIL, the described
  failure, the justified change, and retained coverage and valid alternatives.
- "approved": a boolean approving or rejecting this proposed set.
- "issues": concrete problems with the affected criteria when rejecting;
  an empty array when approving.
Do not rewrite the criteria or add commentary outside the JSON object.
"""


def parse_output(payload: Any) -> ReviewResult:
    if not isinstance(payload, dict) or set(payload) != {"reason", "approved", "issues"}:
        raise ValueError('Expected exactly "reason", "approved", and "issues".')
    if not isinstance(payload["issues"], list):
        raise ValueError('"issues" must be an array of strings.')
    approved = require_bool(payload["approved"], "approved")
    issues = [require_text(issue, "issues item") for issue in payload["issues"]]
    if approved and issues:
        raise ValueError('"issues" must be empty when "approved" is true.')
    return ReviewResult(
        reason=require_text(payload["reason"], "reason"),
        approved=approved,
        issues=issues,
    )


async def review(
    *,
    runtime: AgentRuntime,
    call_id: str,
    prompt: str,
    rubrics: list[str],
    candidate_rubrics: list[str],
    p: str,
    y: str,
    judge_reason: str,
    change_reason: str,
    references: list[str],
    old_verification: list[dict],
    new_verification: list[dict],
) -> ReviewResult:
    return await runtime.call(
        role="review",
        system=SYSTEM_PROMPT,
        inputs={
            "task": prompt,
            "old_criteria": rubrics,
            "proposed_criteria": candidate_rubrics,
            "alleged_failure": p,
            "submitted_answer": y,
            "quality_judgment_reason": judge_reason,
            "repair_explanation": change_reason,
            "fallible_reference_answers": references,
            "old_verification": old_verification,
            "new_verification": new_verification,
        },
        parse=parse_output,
        call_id=call_id,
    )
