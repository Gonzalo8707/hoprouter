"""
Local model wrapper.

Everything that runs through here counts as ZERO tokens toward the final
score. This is where cheap, "easy" categories (sentiment, simple NER,
short factual answers) should be resolved without ever touching Fireworks.

We use a small instruction-tuned model that is light enough to run on CPU
in the standardized scoring environment (no GPU guaranteed there), so the
router's local branch never depends on GPU access being available.

If loading the model fails for any reason (no internet in the sandboxed
scoring environment, missing weights, etc.), we fall back to a rule-based
answer so the container never crashes and always exits 0.
"""

import threading

_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # small, CPU-friendly, permissive license

_lock = threading.Lock()
_pipeline = None
_load_failed = False


def _get_pipeline():
    global _pipeline, _load_failed
    if _pipeline is not None or _load_failed:
        return _pipeline
    with _lock:
        if _pipeline is not None or _load_failed:
            return _pipeline
        try:
            from transformers import pipeline
            _pipeline = pipeline(
                "text-generation",
                model=_MODEL_NAME,
                device_map="auto",
            )
        except Exception:
            _load_failed = True
            _pipeline = None
    return _pipeline


def warmup():
    """Force the pipeline to load now (called once at container startup),
    so loading time is paid during the 60s readiness window instead of
    during the first task's 30s response window."""
    _get_pipeline()


def _fallback_answer(prompt: str) -> str:
    # Minimal, safe fallback so we never return empty/malformed output.
    return "Unable to process with local model; please review manually."


def generate(prompt: str, max_new_tokens: int = 120, max_time_s: float = 22.0) -> str:
    """
    Generate a local answer, bounded both by token count AND wall-clock time
    (max_time is enforced by transformers' generate() itself). The 22s cap
    leaves a safety margin under the harness's 30s-per-request hard limit,
    accounting for tokenization/pipeline overhead on top of raw generation.
    """
    pipe = _get_pipeline()
    if pipe is None:
        return _fallback_answer(prompt)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise, concise assistant. Always answer in "
                "English, regardless of the input language. Follow the "
                "requested output format exactly. Be concise: for open-ended "
                "questions, answer in at most 3 sentences. For entity "
                "extraction tasks, check carefully for ALL requested entity "
                "types (e.g. person, organization, location, AND date) "
                "before finishing - do not omit a category just because "
                "it's the last one you'd mention."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        out = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            max_time=max_time_s,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        generated = out[0]["generated_text"]
        # pipeline returns the full conversation; take the last assistant turn
        if isinstance(generated, list):
            return generated[-1]["content"].strip()
        return str(generated).strip()
    except Exception:
        return _fallback_answer(prompt)
