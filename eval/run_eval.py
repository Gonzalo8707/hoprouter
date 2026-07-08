"""
Local evaluation harness.

Run this yourself (with your own Fireworks dev key + GPU access) BEFORE
submitting, to get a sense of:
  1. Whether the router's local/remote decisions are safe per category
     (using cheap automatic checks: sentiment label present, valid JSON
     for NER, expected substrings for math/logic/debug, word-count bounds
     for summaries).
  2. How many real tokens you're spending on Fireworks (from the API's own
     `usage` field), so you know your actual cost profile before the
     harness scores you on the real hidden tasks.

IMPORTANT: this is a heuristic sanity check, not the real LLM-Judge used
in evaluation. Passing everything here does not guarantee passing the
accuracy gate - but failing here is a strong signal something is broken.

Usage:
    export FIREWORKS_API_KEY=...
    export FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
    export ALLOWED_MODELS=minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4
    python eval/run_eval.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from router import route_task, Route, Category  # noqa: E402
from fireworks_client import FireworksClient, ConfigError  # noqa: E402
from local_model import generate as local_generate  # noqa: E402
from validators import passes_local_safety_check  # noqa: E402

EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_tasks.json")


def check_expectations(task: dict, answer: str) -> bool:
    if answer.startswith("[EVAL ERROR]") or answer.startswith("Error processing task"):
        return False

    lowered = answer.lower()

    if task.get("expect_json"):
        import re
        match = re.search(r"(\{.*\}|\[.*\])", answer, re.DOTALL)
        if not match:
            return False
        try:
            json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            return False

    if "expect_contains" in task:
        if not any(term.lower() in lowered for term in task["expect_contains"]):
            return False

    if "expect_min_words" in task:
        if len(answer.split()) < task["expect_min_words"]:
            return False

    if "expect_max_words" in task:
        if len(answer.split()) > task["expect_max_words"]:
            return False

    return True


def main():
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_tasks = json.load(f)

    try:
        fw_client = FireworksClient()
    except ConfigError as e:
        print(f"ERROR: {e}")
        print("Set FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS first.")
        sys.exit(1)

    total_remote_tokens = 0
    total_calls_remote = 0
    total_calls_local = 0
    escalations = 0
    passed = 0
    results_by_category = {}

    for task in eval_tasks:
        prompt = task["prompt"]
        category, route, model = route_task(prompt)
        expected_category = task.get("category")

        t0 = time.time()
        used_route = route.value

        try:
            if route == Route.LOCAL:
                answer = local_generate(prompt)
                total_calls_local += 1
                if not passes_local_safety_check(category, answer):
                    escalations += 1
                    used_route = "remote (escalated)"
                    answer, usage = fw_client.chat_completion_with_usage(model=model or "gemma-4-31b-it", prompt=prompt)
                    total_remote_tokens += usage.get("total_tokens", 0)
                    total_calls_remote += 1
            else:
                answer, usage = fw_client.chat_completion_with_usage(model=model, prompt=prompt)
                total_remote_tokens += usage.get("total_tokens", 0)
                total_calls_remote += 1
        except Exception as e:
            answer = f"[EVAL ERROR] {e}"

        elapsed = time.time() - t0
        ok = check_expectations(task, answer)
        passed += int(ok)

        cat_key = expected_category or category.value
        results_by_category.setdefault(cat_key, {"pass": 0, "total": 0})
        results_by_category[cat_key]["total"] += 1
        results_by_category[cat_key]["pass"] += int(ok)

        status = "PASS" if ok else "FAIL"
        category_match = "" if expected_category == category.value else f" (classified as {category.value}!)"
        print(f"[{status}] {task['task_id']:6s} route={used_route:18s} {elapsed:5.1f}s{category_match}")
        if not ok:
            print(f"         prompt: {prompt[:80]}...")
            print(f"         answer: {answer[:150]}")

    print("\n--- Summary by category ---")
    for cat, r in results_by_category.items():
        print(f"  {cat:28s} {r['pass']}/{r['total']}")

    print("\n--- Overall ---")
    print(f"  Passed: {passed}/{len(eval_tasks)}")
    print(f"  Local calls: {total_calls_local} (escalated to remote: {escalations})")
    print(f"  Remote calls: {total_calls_remote}")
    print(f"  Total remote tokens spent: {total_remote_tokens}")


if __name__ == "__main__":
    main()
