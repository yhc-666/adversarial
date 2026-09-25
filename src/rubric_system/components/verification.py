from __future__ import annotations

import asyncio
from dataclasses import dataclass

from rubric_system.agents.common import AgentRuntime, require_rubrics
from rubric_system.agents.verifier import verify_criterion


@dataclass(frozen=True)
class VerificationResult:

    results: list[dict[str, str]]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("verification requires at least one criterion")
        for item in self.results:
            if item.get("verdict") not in {"PASS", "FAIL", "N/A"}:
                raise ValueError("verification result contains an invalid verdict")
            if not item.get("criterion") or not item.get("reason"):
                raise ValueError("verification requires each criterion and its reason")

    @property
    def pass_count(self) -> int:
        return sum(item["verdict"] == "PASS" for item in self.results)

    @property
    def fail_count(self) -> int:
        return sum(item["verdict"] == "FAIL" for item in self.results)

    @property
    def na_count(self) -> int:
        return sum(item["verdict"] == "N/A" for item in self.results)

    @property
    def score(self) -> float | None:
        denominator = self.pass_count + self.fail_count
        return self.pass_count / denominator if denominator else None

    @property
    def all_pass(self) -> bool:
        return self.pass_count > 0 and self.fail_count == 0

    @property
    def failed_criteria(self) -> list[dict[str, str]]:
        return [{"criterion": item["criterion"], "reason": item["reason"]}
                for item in self.results if item["verdict"] == "FAIL"]

    def to_json(self) -> dict:
        return {"results": self.results, "pass_count": self.pass_count,
                "fail_count": self.fail_count, "na_count": self.na_count,
                "score": self.score, "all_pass": self.all_pass}


async def verify_set(
    *, runtime: AgentRuntime, call_id: str, prompt: str,
    rubrics: list[str], response: str,
) -> VerificationResult:
    criteria = require_rubrics(rubrics)
    outcomes = await asyncio.gather(*(
        verify_criterion(runtime=runtime, call_id=f"{call_id}:criterion{index + 1}",
                         prompt=prompt, criterion=criterion, response=response)
        for index, criterion in enumerate(criteria)
    ), return_exceptions=True)


    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    return VerificationResult(results=[
        {"criterion": criterion, "verdict": outcome.verdict, "reason": outcome.reason}
        for criterion, outcome in zip(criteria, outcomes, strict=True)
    ])
