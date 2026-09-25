from __future__ import annotations

from typing import Any

from rubric_system.agents.common import AgentRuntime, require_text


SYSTEM_PROMPT = """Role
You are a careful assistant answering a user's task.

Inputs
You receive the user's task and any material included in it.

Goal
Write the complete answer the user should receive. This will be one fallible
reference answer, not a ground-truth solution or a required template.

Requirements
Follow the requested deliverable, constraints, and format. Work out important
facts, calculations, and behavior carefully. Preserve the meaning of supplied
material. Do not invent missing facts or claim external actions you did not
perform. Address a benign request in its actual context; handle harmful requests
with appropriate boundaries and useful safe assistance.

Output
Return one JSON object with exactly one field, "answer", containing the complete
answer as a string. Put the user's requested answer format inside this string.
Do not add commentary outside the JSON object.
"""


def parse_output(payload: Any) -> str:
    if not isinstance(payload, dict) or set(payload) != {"answer"}:
        raise ValueError('Expected an object containing exactly "answer".')
    return require_text(payload["answer"], "answer")


async def generate_reference(
    *, runtime: AgentRuntime, call_id: str, prompt: str
) -> str:
    return await runtime.call(
        role="reference",
        system=SYSTEM_PROMPT,
        inputs={"task": prompt},
        parse=parse_output,
        call_id=call_id,
    )
