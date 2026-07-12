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
from local_model import generate as local_generate, warmup as local_warmup
from validators import passes_local_safety_check

INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"


def load_tasks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_results(path: str, results: list):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _extract_explicit_category(task: dict):
    """
    Some published task schemas (including the harness's own public
    example set) attach a category/type label alongside the prompt. If
    present, this is strictly more reliable than guessing from the prompt
    text via regex - so we check a few likely key names and hand it to
    the router, which falls back to its regex classifier if none of these
    are present or recognized.
    """
    for key in ("category", "task_category", "type", "task_type"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def solve_task(task: dict, fw_client) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]

    category, route, model = route_task(prompt, explicit_category=_extract_explicit_category(task))
    # monotonic(): time.time() jumps under WSL2 clock skew (a task once
    # logged a negative elapsed) and must not be used for durations.
    t0 = time.monotonic()
    used = route.value

    try:
        if route == Route.LOCAL or fw_client is None:
            answer, complete = local_generate(prompt, category=category)

            # Safety net: a locally-routed answer that looks broken or was
            # cut off (deadline/max-token) risks the all-or-nothing
            # accuracy gate. If Fireworks is available, escalate instead
            # of risking the whole task's score to save a few tokens.
            if fw_client is not None and (
                    not complete
                    or not passes_local_safety_check(category, answer)):
                used = "local->remote (escalated)"
                escalation_model = MODEL_PREFERENCE.get(category, MODEL_PREFERENCE[Category.UNKNOWN])
                answer = fw_client.chat_completion(model=escalation_model, prompt=prompt, category=category)
        else:
            answer = fw_client.chat_completion(model=model, prompt=prompt, category=category)
    except Exception as e:
        # Never let a single task crash the whole submission.
        # Fall back to local generation so we still emit a valid answer.
        used += "->local fallback"
        try:
            answer, _ = local_generate(prompt, category=category)
        except Exception:
            answer = f"Error processing task: {e}"

    print(f"[{task_id}] category={category.value} route={used} "
          f"{time.monotonic() - t0:.1f}s", file=sys.stderr)
    return {"task_id": task_id, "answer": answer}


def main():
    start = time.monotonic()

    # Load the local model into memory now, during startup, so this cost
    # counts against the container's 60s readiness budget rather than
    # against the first task's 30s per-request budget.
    local_warmup()

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

    elapsed = time.monotonic() - start
    print(f"Done. Processed {len(results)} tasks in {elapsed:.1f}s", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
