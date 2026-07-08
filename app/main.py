"""
HopRouter entrypoint.

Contract (from the Participant Guide):
  - Read tasks from /input/tasks.json on startup
  - Write results to /output/results.json before exiting
  - Exit code 0 on success, non-zero on failure
  - Max runtime 10 minutes total, 30s per request
  - Must be ready within 60s of container start
  - All answers must be in English
"""

import json
import os
import sys
import time

from router import route_task, Route, Category, MODEL_PREFERENCE
from fireworks_client import FireworksClient, ConfigError
from local_model import generate as local_generate
from validators import passes_local_safety_check

INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"
PER_REQUEST_TIMEOUT_S = 28  # stay safely under the 30s hard limit


def load_tasks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_results(path: str, results: list):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def solve_task(task: dict, fw_client) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]

    category, route, model = route_task(prompt)

    try:
        if route == Route.LOCAL or fw_client is None:
            answer = local_generate(prompt)

            # Safety net: a locally-routed answer that looks broken risks
            # the all-or-nothing accuracy gate. If it fails a basic sanity
            # check and Fireworks is available, escalate instead of
            # risking the whole task's score to save a few tokens.
            if fw_client is not None and not passes_local_safety_check(category, answer):
                escalation_model = MODEL_PREFERENCE.get(category, MODEL_PREFERENCE[Category.UNKNOWN])
                answer = fw_client.chat_completion(model=escalation_model, prompt=prompt)
        else:
            answer = fw_client.chat_completion(model=model, prompt=prompt)
    except Exception as e:
        # Never let a single task crash the whole submission.
        # Fall back to local generation so we still emit a valid answer.
        try:
            answer = local_generate(prompt)
        except Exception:
            answer = f"Error processing task: {e}"

    return {"task_id": task_id, "answer": answer}


def main():
    start = time.time()

    try:
        tasks = load_tasks(INPUT_PATH)
    except Exception as e:
        print(f"FATAL: could not read {INPUT_PATH}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        fw_client = FireworksClient()
    except ConfigError as e:
        # Fireworks env vars missing: still try to complete everything
        # locally rather than failing outright.
        print(f"WARNING: Fireworks client unavailable: {e}", file=sys.stderr)
        fw_client = None

    results = []
    for task in tasks:
        results.append(solve_task(task, fw_client))

    try:
        write_results(OUTPUT_PATH, results)
    except Exception as e:
        print(f"FATAL: could not write {OUTPUT_PATH}: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start
    print(f"Done. Processed {len(results)} tasks in {elapsed:.1f}s", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
