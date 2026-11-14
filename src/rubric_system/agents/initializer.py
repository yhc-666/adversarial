from __future__ import annotations

from typing import Any

from rubric_system.agents.common import (
    AgentRuntime,
    RUBRIC_PRINCIPLES,
    require_rubrics,
)


SYSTEM_PROMPT = f"""Role
You design scoring criteria for answers to a user's task.

Inputs
You receive the user's task.

Goal
Write a complete set of specific, directly verifiable scoring criteria for the
user's task.

Requirements
Identify the requested result or deliverable, the constraints that determine
whether it is usable, and the important correctness obligations. Preserve the
task's objects, quantities, scope, and conditions. Where the task calls for a
particular output contract, assess that actual contract rather than general
readability. Work out necessary checking information before writing a criterion.
Do not reward an explanation of how to complete a task in place of its requested
deliverable, or a claim of successful behavior in place of that behavior.

Write checks on the requested outcome, not a recipe that every answer must
follow. For each obligation, identify its basis in the user's request. Then
consider a different valid way to complete the task: it should still pass.
Solve or derive facts to establish checking information, but do not require the
answer to reproduce your derivation unless explanation itself was requested.

{RUBRIC_PRINCIPLES}

Output
Return one JSON object with exactly "rubrics": a nonempty array of criterion
strings. Each string is the complete criterion. Do not include IDs, weights,
scores, or commentary outside this object. Let the task determine how many
distinct criteria are needed.
"""


def parse_output(payload: Any) -> list[str]:
    if not isinstance(payload, dict) or set(payload) != {"rubrics"}:
        raise ValueError('Expected an object containing exactly "rubrics".')
    return require_rubrics(payload["rubrics"])


async def initialize(
    *,
    runtime: AgentRuntime,
    call_id: str,
    prompt: str,
) -> list[str]:
    return await runtime.call(
        role="initializer",
        system=SYSTEM_PROMPT,
        inputs={"task": prompt},
        parse=parse_output,
        call_id=call_id,
    )
