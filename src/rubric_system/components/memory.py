from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskMemory:

    all_pass_failed: list[dict[str, Any]] = field(default_factory=list)
    judge_rejected: list[dict[str, Any]] = field(default_factory=list)
    fixes: list[dict[str, Any]] = field(default_factory=list)

    def record_failed_attack(self, *, p: str, y: str, failed_criteria: list[dict]) -> None:
        if not failed_criteria:
            raise ValueError("a failed-attack memory needs at least one actual FAIL")
        self.all_pass_failed.append({"p": p, "y": y, "failed_criteria": deepcopy(failed_criteria)})

    def record_judge_rejection(self, *, p: str, y: str, judge_reason: str) -> None:
        self.judge_rejected.append({"p": p, "y": y, "judge_reason": judge_reason})

    def record_fix(self, *, p: str, y: str, rubric_before: list[str], rubric_after: list[str]) -> None:
        self.fixes.append({"p": p, "y": y, "rubric_before": list(rubric_before),
                           "rubric_after": list(rubric_after)})

    def attack_view(self) -> dict[str, list[dict]]:
        return {
            "all_pass_failed": deepcopy(self.all_pass_failed),
            "judge_rejected": deepcopy(self.judge_rejected),
            "fixed_hacks": [{"p": item["p"], "y": item["y"]} for item in self.fixes],
        }

    def repair_view(self) -> list[dict]:
        return deepcopy(self.fixes)

