from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from rubric_system.agents.common import require_text


@dataclass(frozen=True)
class Example:
    sample_id: str
    benchmark: str
    domain: str
    prompt: str
    response_a: str
    response_b: str
    preferred_response: str
    raw_id: str | None = None
    chosen: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()

    @property
    def column(self) -> str:
        return f"{self.benchmark}/{self.domain}"


def parse_example(row: dict[str, Any]) -> Example:
    fields = {name: require_text(row[name], name)
              for name in ("sample_id", "benchmark", "domain", "prompt")}
    for name in ("response_a", "response_b"):
        if not isinstance(row[name], str):
            raise ValueError(f"{name} must be a string")
        fields[name] = row[name]
    preferred = row["preferred_response"]
    if preferred not in ("A", "B"):
        raise ValueError("preferred_response must be A or B")
    if "label" in row and (type(row["label"]) is not int or row["label"] != int(preferred == "B")):
        raise ValueError("label must agree with preferred_response (0=A, 1=B)")
    fields["preferred_response"] = preferred
    if fields["benchmark"] == "rmbench":
        if fields["domain"] not in {"chat", "math", "code", "safety"}:
            raise ValueError("rmbench domain must be chat, math, code, or safety")
        raw = row["raw_record"]
        if not isinstance(raw, dict) or raw.get("prompt") != fields["prompt"]:
            raise ValueError("rmbench raw_record.prompt must match prompt")
        fields["raw_id"] = require_text(row["metadata"]["raw_id"], "metadata.raw_id")
        for side in ("chosen", "rejected"):
            responses = raw[side]
            if (not isinstance(responses, list) or len(responses) != 3
                    or any(not isinstance(text, str) for text in responses)):
                raise ValueError(f"rmbench raw_record.{side} must contain three response strings")
            fields[side] = tuple(responses)
    return Example(**fields)


def load_examples(path: Path) -> list[Example]:
    examples = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each line must be an object")
                example = parse_example(row)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if example.sample_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id {example.sample_id}")
            seen.add(example.sample_id)
            examples.append(example)
    if not examples:
        raise ValueError("dataset must contain at least one example")
    return examples


def evaluation_examples(examples: list[Example]) -> tuple[list[Example], dict[str, str]]:
    """Choose one fixed rubric source per RM-Bench question before generation."""
    groups = defaultdict(list)
    selected = []
    for example in examples:
        if example.benchmark == "rmbench":
            groups[example.raw_id].append(example)
        else:
            selected.append(example)
    generator = random.Random(20260914)
    representatives = {}
    for raw_id, siblings in sorted(groups.items()):
        first = siblings[0]
        for sibling in siblings:
            if (sibling.prompt, sibling.domain, sibling.chosen, sibling.rejected) != (
                first.prompt, first.domain, first.chosen, first.rejected
            ):
                raise ValueError(f"inconsistent RM-Bench question: {raw_id}")
        representative = generator.choice(sorted(siblings, key=lambda item: item.sample_id))
        representatives[raw_id] = representative.sample_id
        selected.append(representative)
    return selected, representatives
