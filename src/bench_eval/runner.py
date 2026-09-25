from __future__ import annotations

import asyncio
from collections import defaultdict
import hashlib
import json
from typing import Any

from rubric_system.agents.common import AgentFormatError, decode_json, require_rubrics
from rubric_system.agents.verifier import SYSTEM_PROMPT, build_user_message, parse_output
from bench_eval.datasets.pairwise_loader import Example
from infra.llm.client import LLMClient, LLMError


class Evaluator:
    """Reuse evaluation outcomes in memory for identical inputs."""

    def __init__(self, *, client: LLMClient, model: str) -> None:
        self.client = client
        self.model = model
        self._pending: dict[str, asyncio.Task] = {}
        self._outcomes: dict[str, str | LLMError] = {}

    async def criterion(self, *, prompt: str, criterion: str, response: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(
                    prompt=prompt, criterion=criterion, response=response,
                )},
            ],
            "temperature": 0.0, "top_p": 1.0, "max_tokens": 16384,
            "response_format": {"type": "json_object"},
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if key not in self._outcomes:
            task = self._pending.get(key)
            if task is None:
                task = asyncio.create_task(self._request(payload))
                self._pending[key] = task
            try:
                self._outcomes[key] = await asyncio.shield(task)
            finally:
                if task.done():
                    self._pending.pop(key, None)
        outcome = self._outcomes[key]
        if isinstance(outcome, LLMError):
            raise outcome
        return outcome

    async def _request(self, payload: dict) -> str | LLMError:
        try:
            for attempt in range(2):
                content = await self.client.complete(payload)
                try:
                    verdict = parse_output(decode_json(content))
                except (ValueError, TypeError, KeyError) as error:
                    if attempt == 1:
                        raise AgentFormatError(str(error)) from error
                    payload = {**payload, "messages": [*payload["messages"],
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": f"Your response could not be parsed: {error}. Return the required JSON shape. Reconsider only the formatting, without assuming any particular verdict."},
                    ]}
                else:
                    return verdict.verdict
        except (LLMError, AgentFormatError) as error:
            return LLMError(str(error))
        raise AssertionError("format budget must return an outcome")

    async def score(self, prompt: str, rubrics: list[str], response: str) -> dict:
        outcomes = await asyncio.gather(*(
            self.criterion(prompt=prompt, criterion=criterion, response=response)
            for criterion in require_rubrics(rubrics)
        ), return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException) and not isinstance(outcome, (LLMError, AgentFormatError)):
                raise outcome
        if any(isinstance(outcome, BaseException) for outcome in outcomes):
            return {"status": "failed", "score": None}
        applicable = [outcome for outcome in outcomes if outcome != "N/A"]
        return {"status": "completed" if applicable else "unscorable",
                "score": applicable.count("PASS") / len(applicable) if applicable else None}

    async def evaluate(self, example: Example, generation: dict) -> list[dict]:
        responses = ([*example.chosen, *example.rejected] if example.benchmark == "rmbench"
                     else [example.response_a, example.response_b])
        if generation["status"] == "failed":
            scores = [{"status": "failed", "score": None}
                      for _ in responses]
        else:
            scores = await asyncio.gather(*(
                self.score(example.prompt, generation["rubrics"], response)
                for response in responses
            ))
        base = {"sample_id": example.sample_id, "column": example.column}
        if example.benchmark == "rmbench":
            pairs = []
            for chosen in range(3):
                for rejected in range(3):
                    pairs.append({**base, "raw_id": example.raw_id,
                                  "chosen_style_index": chosen, "rejected_style_index": rejected,
                                  "difficulty": "normal" if chosen == rejected else "easy" if chosen > rejected else "hard",
                                  **compare(scores[chosen], scores[3 + rejected], "A")})
        else:
            pairs = [{**base, **compare(scores[0], scores[1], example.preferred_response)}]
        return pairs


def compare(a: dict, b: dict, preferred: str) -> dict:
    status = ("failed" if "failed" in (a["status"], b["status"]) else
              "unscorable" if "unscorable" in (a["status"], b["status"]) else "completed")
    prediction = None
    credit = None
    if status == "completed":
        prediction = "tie" if a["score"] == b["score"] else "A" if a["score"] > b["score"] else "B"
        credit = 0.5 if prediction == "tie" else float(prediction == preferred)
    return {"status": status, "prediction": prediction, "preferred_response": preferred,
            "credit": credit, "score_a": a["score"], "score_b": b["score"]}


def metrics(rows: list[dict]) -> dict:
    completed = [row for row in rows if row["status"] == "completed"]
    credit = sum(row["credit"] for row in completed)
    return {
        "requested": len(rows), "completed": len(completed),
        "failed": sum(row["status"] == "failed" for row in rows),
        "unscorable": sum(row["status"] == "unscorable" for row in rows),
        "ties": sum(row["prediction"] == "tie" for row in completed),
        "coverage": len(completed) / len(rows) if rows else None,
        "accuracy": credit / len(completed) if completed else None,
    }


def mean(values: list[float | None]) -> float | None:
    return sum(values) / len(values) if values and all(value is not None for value in values) else None


def summarize(rows: list[dict]) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["column"]].append(row)
    columns = {column: metrics(values) for column, values in sorted(groups.items())}
    summary = {"overall": metrics(rows), "per_column": columns,
               "macro_accuracy": mean([column["accuracy"] for column in columns.values()])}
    rm = [row for row in rows if row["column"].startswith("rmbench/")]
    if rm:
        domains = ("chat", "math", "code", "safety")
        summary["rmbench_overall"] = mean([
            columns.get(f"rmbench/{domain}", {}).get("accuracy") for domain in domains
        ])
        summary["rmbench_difficulty"] = {
            level: mean([
                metrics([row for row in rm if row["column"] == f"rmbench/{domain}"
                         and row["difficulty"] == level])["accuracy"]
                for domain in domains
            ]) for level in ("easy", "normal", "hard")
        }
    return summary
