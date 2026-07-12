"""
Local model wrapper - Qwen2.5-3B-Instruct (Q4_K_M GGUF via llama.cpp).

Everything that runs through here counts as ZERO tokens toward the final
score. Model sizing is dictated by the scoring environment (Participant
Guide: 4GB RAM, 2 vCPU, no GPU):
  - 7B Q4 needs ~6GB resident -> OOM-killed on a 4GB host (total failure)
  - 3B Q4_K_M is 1.93GB on disk, ~2.4GB peak resident -> fits with margin,
    and 2-3B 4-bit is what the guide itself recommends
  - the old 0.5B fit easily but was too weak (caused the 73.7% gate fail)

Runs CPU-only through llama-cpp-python. llama.cpp has no max_time like
transformers' generate(), so the 30s-per-task harness limit is enforced
with streaming + a wall-clock deadline: if generation does not finish
naturally within the deadline, the answer is flagged incomplete and the
caller escalates to Fireworks (see main.py). Worst case we pay the same
remote tokens as the all-remote design - never less accuracy.

generate() returns (text, complete). `complete` is True only when the
model stopped on its own (finish_reason == "stop"); a deadline cut or a
max-token cut returns complete=False so the caller escalates.

If loading the model fails for any reason, we fall back to a marker
answer that the validators recognize, so the container never crashes.
"""

import os
import sys
import threading
import time
import traceback

from router import Category

_MODEL_PATH = os.environ.get(
    "LOCAL_MODEL_PATH", "/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf")

# 2048 ctx comfortably fits the longest prompt we route locally (a ~300
# word summary passage plus the system prompt is well under 1,000 tokens)
# while keeping the KV cache small on a 4GB host.
_N_CTX = 2048

# Wall-clock budget per generation, leaving margin under the harness's
# 30s-per-request hard limit for prompt tokenization + JSON overhead
# (prompt processing alone eats ~3-5s on the 2 vCPU host before the
# first token appears).
#
# NER gets extra headroom: its USER text (a full passage, which we don't
# control) dominates prompt eval on 2 vCPUs, and eval runs showed NER
# needing just past 26s. 28s still leaves ~2s for the escalation
# round-trip, which measured ~1.4s on the same runs - so even the
# worst case (deadline cut at 28s + escalate) stays under 30s.
_DEADLINE_S = 26.0
_DEADLINE_S_BY_CATEGORY = {
    Category.NER: 28.0,
}

# Local token caps are TIME caps in disguise: on the scoring host's
# 2 vCPUs a 3B Q4 generates roughly 5-10 tok/s, so caps are sized so
# generation finishes in <18s even at the slow end (~5 tok/s), leaving
# room for prompt processing inside the deadline. Anything that would
# need more tokens than this should be escalating to Fireworks anyway.
_MAX_NEW_TOKENS = {
    Category.FACTUAL: 40,
    Category.SENTIMENT: 50,
    Category.SUMMARY: 80,
    Category.NER: 60,
}
_DEFAULT_MAX_NEW_TOKENS = 40

# 3B-specific system prompts: shorter and more imperative than the remote
# ones (a 3B follows direct commands better than nuanced prose), but they
# encode the SAME harness rules that got us past the gate: the sentiment
# mixed/negative decision rules, NER's org-vs-location disambiguation and
# fence-free JSON, summary's exact-format compliance, factual's short
# direct answers. NER keeps a one-shot example - at 3B scale that is the
# single most effective way to lock the output schema - but the prompt is
# kept as tight as possible: on 2 vCPUs prompt PROCESSING (not just
# generation) costs real seconds, so local prompt length is not free.
_SYSTEM_PROMPTS = {
    Category.FACTUAL: (
        "Answer the question in 2 short sentences maximum. Be factual "
        "and specific. Start directly with the answer - no preamble."
    ),
    Category.SENTIMENT: (
        "Decide the sentiment of the text. Rules: answer 'negative' only "
        "if there is nothing positive at all; answer 'mixed' if the text "
        "has BOTH a positive and a negative side, even if one dominates; "
        "answer 'neutral' if it is a plain fact with no feeling; "
        "otherwise 'positive'. Write the first line exactly as "
        "'Sentiment: <label>'. Then write 1-2 sentences quoting the words "
        "that show the sentiment; if mixed, quote one positive and one "
        "negative phrase."
    ),
    Category.SUMMARY: (
        "Summarize the text. Obey the format the request asks for "
        "EXACTLY: if it asks for N sentences, write exactly N sentences; "
        "if it asks for N bullet points, write exactly N bullets; respect "
        "any word limits. If no format is given, write exactly one "
        "sentence. Write only the summary - nothing else."
    ),
    Category.NER: (
        "Extract named entities. Output ONLY one JSON object, no code "
        "fence, with keys \"person\", \"organization\", \"location\", "
        "\"date\"; each value is an array of exact strings from the text "
        "([] if none). Companies, universities, agencies and teams are "
        "\"organization\" even if named after a place; \"location\" is "
        "standalone places only. Example: {\"person\": [\"John Smith\"], "
        "\"organization\": [\"ETH Zurich\"], \"location\": [\"Paris\"], "
        "\"date\": [\"May 1, 2020\"]}"
    ),
}
_DEFAULT_SYSTEM_PROMPT = (
    "You are a precise assistant. Answer in English, follow the requested "
    "output format exactly, and be concise. No preamble."
)

_lock = threading.Lock()
_llm = None
_load_failed = False


def _get_llm():
    global _llm, _load_failed
    if _llm is not None or _load_failed:
        return _llm
    with _lock:
        if _llm is not None or _load_failed:
            return _llm
        try:
            from llama_cpp import Llama
            _llm = Llama(
                model_path=_MODEL_PATH,
                n_ctx=_N_CTX,
                n_threads=os.cpu_count() or 2,
                # Qwen2.5 uses the ChatML template; set it explicitly
                # rather than relying on GGUF-metadata autodetection.
                chat_format="chatml",
                verbose=False,
            )
        except Exception:
            # Loud failure: a silent load error makes every local task
            # escalate to Fireworks at full token cost with no visible
            # cause (this exact silence cost a debugging cycle once -
            # missing libgomp in the runtime image).
            print("ERROR: local model failed to load:", file=sys.stderr)
            traceback.print_exc()
            _load_failed = True
            _llm = None
    return _llm


def warmup():
    """Load the model at container startup (60s readiness window) and run
    a tiny generation so the first real task doesn't pay mmap page-in or
    graph-warmup costs against its own 30s budget."""
    llm = _get_llm()
    if llm is None:
        return
    try:
        llm.create_chat_completion(
            messages=[{"role": "user", "content": "Say OK."}],
            max_tokens=4,
            temperature=0.0,
        )
    except Exception:
        pass


_FALLBACK = "Unable to process with local model; please review manually."


def generate(prompt: str, category=None, deadline_s: float = None):
    """
    Returns (text, complete).

    complete=True only if the model finished on its own within the
    deadline (per-category, see _DEADLINE_S_BY_CATEGORY). Deadline cuts
    and max-token cuts return complete=False so main.py escalates to
    Fireworks instead of submitting a suspect answer (the accuracy gate
    is all-or-nothing; a truncated JSON or half summary is never worth
    the token saving).
    """
    if deadline_s is None:
        deadline_s = _DEADLINE_S_BY_CATEGORY.get(category, _DEADLINE_S)

    llm = _get_llm()
    if llm is None:
        return _FALLBACK, False

    messages = [
        {"role": "system",
         "content": _SYSTEM_PROMPTS.get(category, _DEFAULT_SYSTEM_PROMPT)},
        {"role": "user", "content": prompt},
    ]
    max_tokens = _MAX_NEW_TOKENS.get(category, _DEFAULT_MAX_NEW_TOKENS)

    # monotonic(): time.time() was observed jumping (even backwards)
    # under WSL2 clock skew, which would corrupt the deadline math.
    start = time.monotonic()
    parts = []
    finish_reason = None
    try:
        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
        )
        for chunk in stream:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            piece = delta.get("content")
            if piece:
                parts.append(piece)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            if time.monotonic() - start > deadline_s:
                break
    except Exception:
        if not parts:
            return _FALLBACK, False

    text = "".join(parts).strip()
    if not text:
        return _FALLBACK, False
    return text, finish_reason == "stop"
