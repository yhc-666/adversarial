from __future__ import annotations

from infra.llm.client import LLMClient, LLMError
from rubric_system.agents.attack import AttackCandidate, attack
from rubric_system.agents.common import AgentFormatError, AgentRuntime, require_text
from rubric_system.agents.initializer import initialize
from rubric_system.agents.quality_judge import judge_quality
from rubric_system.agents.reference import generate_reference
from rubric_system.agents.repair import repair
from rubric_system.agents.review import review
from rubric_system.components.memory import TaskMemory
from rubric_system.components.verification import VerificationResult, verify_set
from rubric_system.orchestration.config import EvolutionConfig


async def attempt_attack(
    *, runtime: AgentRuntime, call_id: str, prompt: str, rubrics: list[str],
    references: list[str], memory: TaskMemory,
) -> tuple[AttackCandidate | None, VerificationResult | None]:
    candidate = None
    feedback = None
    for attempt in range(runtime.config.attack_max_attempts):
        candidate = await attack(
            runtime=runtime, call_id=f"{call_id}:attack{attempt + 1}", prompt=prompt,
            rubrics=rubrics, references=references, memory=memory.attack_view(),
            previous=candidate, feedback=feedback,
        )
        if candidate is None:
            return None, None
        verification = await verify_set(
            runtime=runtime, call_id=f"{call_id}:attack{attempt + 1}:verify", prompt=prompt,
            rubrics=rubrics, response=candidate.y,
        )
        if verification.all_pass:
            return candidate, verification
        feedback = verification.results
        if attempt + 1 == runtime.config.attack_max_attempts:
            if verification.fail_count:
                memory.record_failed_attack(
                    p=candidate.p, y=candidate.y, failed_criteria=verification.failed_criteria,
                )
            return None, None
    raise AssertionError("positive attack budget must return a result")


async def attempt_repair(
    *, runtime: AgentRuntime, call_id: str, prompt: str, rubrics: list[str],
    candidate: AttackCandidate, judge_reason: str, references: list[str],
    old_verification: VerificationResult, memory: TaskMemory,
) -> list[str] | None:
    previous = None
    feedback = None
    for attempt in range(runtime.config.repair_max_attempts):
        proposal = await repair(
            runtime=runtime, call_id=f"{call_id}:repair{attempt + 1}", prompt=prompt,
            rubrics=rubrics, p=candidate.p, y=candidate.y, judge_reason=judge_reason,
            references=references, memory=memory.repair_view(),
            previous=previous, feedback=feedback,
        )
        previous = proposal
        if proposal.rubrics == rubrics:
            feedback = {
                "stage": "revision",
                "issue": "The proposed criteria are identical to the current set. "
                         "Return a revised set addressing the confirmed failure.",
            }
            continue
        verification = await verify_set(
            runtime=runtime, call_id=f"{call_id}:repair{attempt + 1}:verify", prompt=prompt,
            rubrics=proposal.rubrics, response=candidate.y,
        )
        if verification.fail_count == 0:
            feedback = {
                "stage": "verification", "results": verification.results,
                "issue": "The same attack still has no actual FAIL. N/A does not repair the failure.",
            }
            continue
        assessment = await review(
            runtime=runtime, call_id=f"{call_id}:repair{attempt + 1}:review", prompt=prompt,
            rubrics=rubrics, candidate_rubrics=proposal.rubrics,
            p=candidate.p, y=candidate.y, judge_reason=judge_reason,
            change_reason=proposal.change_reason, references=references,
            old_verification=old_verification.results,
            new_verification=verification.results,
        )
        if not assessment.approved:
            feedback = {"stage": "review", "reason": assessment.reason,
                        "issues": assessment.issues, "verification": verification.results}
            continue
        return proposal.rubrics
    return None


async def evolve(
    *, sample_id: str, prompt: str, config: EvolutionConfig, client: LLMClient,
) -> list[str]:
    """Generate and refine a rubric using task-local attack and repair memory."""
    require_text(sample_id, "sample_id")
    require_text(prompt, "prompt")
    runtime = AgentRuntime(client=client, config=config)
    references = []
    try:
        references.append(await generate_reference(
            runtime=runtime, call_id=f"{sample_id}:reference1", prompt=prompt,
        ))
    except (LLMError, AgentFormatError):
        pass

    rubrics = await initialize(
        runtime=runtime, call_id=f"{sample_id}:initializer", prompt=prompt,
    )
    memory = TaskMemory()
    no_update_count = 0

    for round_number in range(1, config.max_rounds + 1):
        call_id = f"{sample_id}:round{round_number}"
        updated = False
        try:
            candidate, old_verification = await attempt_attack(
                runtime=runtime, call_id=call_id, prompt=prompt, rubrics=rubrics,
                references=references, memory=memory,
            )
            if candidate is not None:
                judgment = await judge_quality(
                    runtime=runtime, call_id=f"{call_id}:quality_judge", prompt=prompt,
                    p=candidate.p, y=candidate.y, references=references,
                )
                if not (judgment.failure_present and judgment.material_harm):
                    memory.record_judge_rejection(
                        p=candidate.p, y=candidate.y, judge_reason=judgment.reason,
                    )
                else:
                    assert old_verification is not None
                    repaired = await attempt_repair(
                        runtime=runtime, call_id=call_id, prompt=prompt, rubrics=rubrics,
                        candidate=candidate, judge_reason=judgment.reason, references=references,
                        old_verification=old_verification, memory=memory,
                    )
                    if repaired is not None:
                        memory.record_fix(
                            p=candidate.p, y=candidate.y,
                            rubric_before=rubrics, rubric_after=repaired,
                        )
                        rubrics = list(repaired)
                        updated = True
        except (LLMError, AgentFormatError):
            pass
        no_update_count = 0 if updated else no_update_count + 1
        if no_update_count >= config.no_update_patience:
            break

    return rubrics
