## Data preparation

Place the prepared benchmark JSONL file at `data/dataset.jsonl`. Each line must contain `sample_id` (unique), `benchmark`, `domain`, `prompt`, `response_a`, `response_b`, and `preferred_response` (`A` or `B`).

For RM-Bench (`benchmark: "rmbench"`), also include `metadata.raw_id` and `raw_record` with `prompt`, `chosen`, and `rejected`. Each response list contains three strings in this order: concise, detailed plain text, detailed Markdown. Use domains `chat`, `math`, `code`, and `safety`.

## Run

Use Python 3.12 on Linux or macOS and an API supporting the chat-completions JSON response format. Set the API base URL, key, and model identifiers supplied by your provider:

```bash
pip install -r requirements.txt

export API_BASE_URL="<api-base-url>"
export API_KEY="<api-key>"
export GENERATION_MODEL="<generation-model>"
export EVALUATION_MODEL="<evaluation-model>"

python scripts/run_evolution_experiment.py --input data/dataset.jsonl --output outputs/main --concurrency 8
```

Only final rubrics (`rubrics.jsonl`), response-pair scores (`pair_results.jsonl`), and aggregate metrics (`summary.json`) are saved under `outputs/main/`. Initial rubrics, round counts, intermediate drafts, reference answers, and per-criterion judgments are not saved. Use a fresh output directory for each run; interrupted runs do not resume.
