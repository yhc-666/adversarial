from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from infra.llm.client import LLMClient
from rubric_system.orchestration.config import EvolutionConfig


RUBRIC_PRINCIPLES = """Write criteria that collectively cover the important task
requirements, especially the actual deliverable, correct results, and explicit
constraints. Each criterion checks one distinct requirement with a clear pass
boundary. Avoid double-counting the same defect across overlapping criteria;
do not split a single meaningful check merely because it contains 'and'.
Make every check specific and self-contained: include justified values,
relations, permitted inputs, or source facts needed to check it directly.
The scorer may read, compare, and count, but should not need external research
or solve the original problem again. Establish correct grounds while writing.
Only require what the task supports. Preserve alternative correct solutions;
a particular implementation, explanation style, or incidental reference detail
is not mandatory unless the task makes it so. A required deliverable or input
case must remain required. Conditional checks on genuinely optional content
must not replace the requirement to complete the task. Keep unrelated valid
requirements intact. There is no fixed number of criteria or target pass rate.
"""

REFERENCE_GUIDANCE = """Each reference answer, if supplied, is an independent
attempt and may contain incorrect facts, reasoning, or omissions. None is a
gold answer, required template, or evidence by itself. Check their relevant
claims against the task. A different valid answer is acceptable; disagreement
with the references alone is not a failure. No reference answer is guaranteed
to pass the criteria. If none are supplied, work from the task directly.
"""


class AgentFormatError(Exception):
    pass


T = TypeVar("T")


def require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def require_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def require_rubrics(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("rubrics must be a nonempty list of strings")
    rubrics = [require_text(item, f"rubrics[{index}]") for index, item in enumerate(value)]
    if len(set(rubrics)) != len(rubrics):
        raise ValueError("rubrics must not contain identical duplicate criteria")
    return rubrics


def decode_json(content: str) -> Any:
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        first_newline = text.find("\n")
        if first_newline < 0:
            raise ValueError("JSON fence must contain a body")
        text = text[first_newline + 1:-3].strip()
    return json.loads(text)


@dataclass
class AgentRuntime:
    client: LLMClient
    config: EvolutionConfig

    async def call(
        self, *, role: str, system: str, inputs: dict | str,
        parse: Callable[[Any], T], call_id: str,
    ) -> T:
        user_message = inputs if isinstance(inputs, str) else json.dumps(inputs, ensure_ascii=False)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user_message}]
        last_error = ""
        for _ in range(self.config.format_max_attempts):
            content = await self.client.complete({
                "model": self.config.model,
                "messages": list(messages),
                "temperature": 0.7 if role in {"reference", "attack"} else 0.0,
                "max_tokens": 30720,
                "response_format": {"type": "json_object"},
            })
            try:
                return parse(decode_json(content))
            except (ValueError, TypeError, KeyError) as error:
                last_error = str(error)
                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": (
                        "Your previous output could not be parsed: " + last_error +
                        "\nReturn the same intended answer in the required JSON format. "
                        "Correct the format only; do not change a judgment to seek approval."
                    )},
                ])
        raise AgentFormatError(f"{call_id}: exhausted {self.config.format_max_attempts} format attempts: {last_error}")

