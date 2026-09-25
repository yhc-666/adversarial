from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Awaitable, Callable, TypeVar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rubric_system.agents.common import AgentFormatError
from rubric_system.orchestration.config import EvolutionConfig
from bench_eval.datasets.pairwise_loader import Example, evaluation_examples, load_examples
from bench_eval.runner import Evaluator, summarize
from rubric_system.orchestration.evolve import evolve
from infra.llm.client import LLMClient, LLMError
from infra.persistence import append_jsonl, write_json, write_jsonl


T = TypeVar("T")
R = TypeVar("R")


def environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} before running the experiment")
    return value


async def map_concurrent(
    function: Callable[[T], Awaitable[R]], items: list[T], concurrency: int,
) -> list[R]:
    iterator = iter(enumerate(items))
    results = [None] * len(items)

    async def worker() -> None:
        for index, item in iterator:
            results[index] = await function(item)

    async with asyncio.TaskGroup() as group:
        for _ in range(min(concurrency, len(items))):
            group.create_task(worker())
    return results


async def experiment(args: argparse.Namespace) -> dict:
    generation_model = environment("GENERATION_MODEL")
    evaluation_model = environment("EVALUATION_MODEL")
    base_url = environment("API_BASE_URL")
    api_key = environment("API_KEY")
    examples = load_examples(args.input)
    selected, _ = evaluation_examples(examples)
    config = EvolutionConfig(model=generation_model)
    async with LLMClient(base_url=base_url, api_key=api_key, concurrency=args.concurrency) as client:
        try:
            args.output.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise ValueError("Output directory already exists; use a fresh --output directory") from None
        completed = 0

        async def generate(example: Example) -> dict:
            nonlocal completed
            try:
                rubrics = await evolve(
                    sample_id=example.sample_id, prompt=example.prompt, config=config, client=client,
                )
                record = {"sample_id": example.sample_id, "status": "completed", "rubrics": rubrics}
            except (LLMError, AgentFormatError):
                record = {"sample_id": example.sample_id, "status": "failed", "rubrics": None}
            append_jsonl(args.output / "rubrics.jsonl", record)
            completed += 1
            print(f"Generate {completed}/{len(examples)}: {record['status']}", flush=True)
            return record

        generations = await map_concurrent(generate, examples, args.concurrency)
        by_id = {record["sample_id"]: record for record in generations}
        write_jsonl(args.output / "rubrics.jsonl", generations)
        evaluator = Evaluator(client=client, model=evaluation_model)
        completed = 0

        async def evaluate(example: Example) -> list[dict]:
            nonlocal completed
            pairs = await evaluator.evaluate(example, by_id[example.sample_id])
            for pair in pairs:
                append_jsonl(args.output / "pair_results.jsonl", pair)
            completed += 1
            print(f"Evaluate {completed}/{len(selected)}", flush=True)
            return pairs

        evaluations = await map_concurrent(evaluate, selected, args.concurrency)
    pairs = [pair for example_pairs in evaluations for pair in example_pairs]
    write_jsonl(args.output / "pair_results.jsonl", pairs)
    summary = summarize(pairs)
    summary["generation"] = {
        "requested": len(generations),
        "completed": sum(record["status"] == "completed" for record in generations),
        "failed": sum(record["status"] == "failed" for record in generations),
    }
    write_json(args.output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ARME rubrics and evaluate response preferences.")
    parser.add_argument("--input", type=Path, default=Path("data/dataset.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/main"))
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    summary = asyncio.run(experiment(args))
    print(json.dumps(summary, indent=2))
    return int(summary["generation"]["failed"] > 0 or summary["overall"]["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
