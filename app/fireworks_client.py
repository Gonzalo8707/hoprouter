"""
Thin wrapper around the Fireworks AI chat completions endpoint.

CRITICAL RULES (from the Participant Guide):
  - Every call MUST go through FIREWORKS_BASE_URL. Calls that bypass it are
    not recorded by the judging proxy and the submission scores zero.
  - Never hardcode model IDs. Only use models present in ALLOWED_MODELS.
  - Never hardcode or bundle your own API key. The harness injects
    FIREWORKS_API_KEY at evaluation time.
"""

import os
import time
import requests

from router import Category

_DEFAULT_SYSTEM_PROMPT = (
    "You are a precise task-solving assistant participating in an "
    "automated benchmark. Always answer in English, regardless of the "
    "language of the prompt. Follow any output-format instructions in the "
    "request EXACTLY (sentence counts, bullet counts, word limits, "
    "required labels, JSON keys, etc.) - never add more or less than what "
    "is asked. Never refuse, and never add meta-commentary, apologies, or "
    "preamble such as 'Sure, here is...'. Do not reveal internal "
    "step-by-step reasoning or <think> tags in the output - go straight to "
    "the requested answer. Make sure the answer is COMPLETE: never cut off "
    "mid-sentence, mid-code, or mid-structure, and never sacrifice "
    "correctness or completeness for brevity."
)

# Per-category system prompt + max_tokens. Tailoring the instructions to
# the exact expected output shape (a single label, strict JSON keys, a
# bare numeric answer, etc.) both improves correctness on structured
# categories and keeps completions on-target. max_tokens per category is
# set generously enough to leave headroom for models that emit internal
# reasoning tokens before the final answer (some of the allowed models are
# reasoning-tuned and were observed doing this - see
# FireworksClient._extract_content's reasoning_content fallback) without
# being so large it risks the per-request time budget.
_CATEGORY_CONFIG = {
    Category.FACTUAL: (
        "Answer the question directly and factually in 2-4 sentences. "
        "Cover the actual mechanism or definition being asked about, not "
        "just a one-line label. Be accurate and complete, but do not pad "
        "with unrelated examples or preamble.",
        380,
    ),
    Category.MATH: (
        "Solve the problem. Show the key calculation steps briefly, then "
        "give the final numeric answer clearly on its own line as "
        "'Answer: <value>' with nothing else after the value. Never omit "
        "the final numeric answer, and never show only the answer with no "
        "working.",
        350,
    ),
    Category.SENTIMENT: (
        "Classify the overall sentiment of the text as exactly one label: "
        "positive, negative, neutral, or mixed. Base the label strictly on "
        "the evidence in the text: use 'negative' only when the text has "
        "no positive or redeeming element at all; use 'mixed' whenever "
        "both positive and negative elements are present, even if one "
        "dominates; use 'neutral' for factual statements with no clear "
        "emotional charge. State the label first on its own line as "
        "'Sentiment: <label>', then in 1-2 sentences justify it by citing "
        "the specific words or phrases that drove the decision - if the "
        "text has both positive and negative elements, name both.",
        220,
    ),
    Category.SUMMARY: (
        "Summarize the given text. The request itself specifies the exact "
        "output format to use (for example: a number of sentences, a "
        "number of bullet points, a word limit per bullet, an overall word "
        "limit) - follow that format EXACTLY, producing precisely that "
        "many sentences/bullets and respecting any stated word limits, no "
        "more and no fewer. If the request does not specify a count, "
        "default to exactly one concise sentence. Output only the summary "
        "itself: no preamble like 'Here is a summary', no restating the "
        "instructions, no commentary before or after.",
        280,
    ),
    Category.NER: (
        "Extract named entities and return ONLY a valid JSON object with "
        "exactly these keys: \"person\", \"organization\", \"location\", "
        "\"date\" - each mapped to an array of the exact strings found in "
        "the text (use an empty array if none are found for that key, but "
        "never omit a key). Disambiguation rules: universities, companies, "
        "agencies, teams, and any other named institution are "
        "\"organization\" even when their name contains a place name (for "
        "example 'ETH Zurich', 'Bank of America', and 'University of "
        "Tokyo' are organizations, NOT locations). \"location\" is only "
        "for standalone places: cities, countries, regions, addresses, "
        "landmarks. Never list the same span under two keys. Return only "
        "the JSON object - no extra text, no markdown code fence.",
        420,
    ),
    Category.CODE_DEBUG: (
        "Identify the exact bug, then return the corrected function in a "
        "single complete code block (use a fenced code block with the "
        "language tag). State the bug in one short sentence before the "
        "code. The corrected code must be fully runnable, not a diff or "
        "partial snippet.",
        480,
    ),
    Category.LOGIC: (
        "Reason step by step internally, but keep any shown reasoning "
        "short - only include what's needed to justify the conclusion. "
        "Refer to entities using the exact names/terms used in the "
        "question. End with the final answer clearly labeled on its own "
        "line as 'Answer: <name/value>', using the exact wording from the "
        "question.",
        420,
    ),
    Category.CODE_GEN: (
        "Implement exactly what is requested as a single, complete, "
        "working function in a fenced code block, including the "
        "signature, all necessary imports, and the full body - no TODOs, "
        "no placeholders, no omitted logic. No extra explanation unless "
        "explicitly asked for one.",
        520,
    ),
}

_TEMPERATURE = 0.0

# Hard per-request ceiling is 30s (harness rule). We budget under that with
# margin for our own JSON parsing/serialization overhead. A single slow
# reasoning-model generation is more likely to succeed with one generous
# timeout than with two short ones, so the first attempt gets most of the
# budget; a retry (for genuine transient failures - timeouts, connection
# errors, 429s, 5xx) only gets whatever time is left.
_TOTAL_BUDGET_S = 27.0
_FIRST_ATTEMPT_TIMEOUT_S = 20.0
_MIN_RETRY_TIMEOUT_S = 5.0


class ConfigError(RuntimeError):
    pass


class FireworksClient:
    def __init__(self):
        self.api_key = os.environ.get("FIREWORKS_API_KEY")
        self.base_url = os.environ.get("FIREWORKS_BASE_URL")
        allowed = os.environ.get("ALLOWED_MODELS", "")
        self.allowed_models = [m.strip()
                               for m in allowed.split(",") if m.strip()]

        if not self.api_key:
            raise ConfigError(
                "FIREWORKS_API_KEY is not set in the environment")
        if not self.base_url:
            raise ConfigError(
                "FIREWORKS_BASE_URL is not set in the environment")
        if not self.allowed_models:
            raise ConfigError("ALLOWED_MODELS is not set in the environment")

    def _resolve_model(self, requested_model: str) -> str:
        model = requested_model
        if model not in self.allowed_models:
            # Fallback: if our preferred model isn't in the allowed list for
            # some reason (e.g. changed on launch day), use the first
            # allowed model rather than failing the whole task.
            model = self.allowed_models[0]

        # Fireworks model IDs are full paths like
        # "accounts/fireworks/models/<slug>". The hackathon announcement
        # published bare slugs (e.g. "minimax-m3"), which return 404 when
        # called directly. Auto-prefix if it looks like a bare slug.
        if "/" not in model:
            model = f"accounts/fireworks/models/{model}"
        return model

    def _post_chat(self, model: str, prompt: str, max_tokens: int,
                   temperature: float, system_prompt: str):
        model_id = self._resolve_model(model)
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Deadline-aware retry: the first attempt gets a generous timeout
        # (reasoning models can spend a while on hidden reasoning tokens
        # before emitting visible content), and only a genuine failure
        # (timeout, connection error, 429, 5xx) triggers a retry - with
        # whatever time remains under the overall per-request budget.
        # This stays well under the harness's 30s-per-request hard limit
        # while not wasting the whole budget on two short, likely-to-fail
        # attempts.
        start = time.time()
        last_exc = None
        attempt = 0
        while True:
            elapsed = time.time() - start
            remaining = _TOTAL_BUDGET_S - elapsed
            if remaining < _MIN_RETRY_TIMEOUT_S:
                break
            timeout = _FIRST_ATTEMPT_TIMEOUT_S if attempt == 0 else remaining
            timeout = min(timeout, remaining)
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                last_exc = e
                attempt += 1
                if attempt >= 2:
                    break
                time.sleep(0.3)
                continue
        raise last_exc

    @staticmethod
    def _extract_content(data: dict) -> str:
        """Defensively extract the answer text. Some models/response shapes
        put content in unexpected places (missing 'content', structured
        content blocks, or a 'reasoning_content' field instead) - we never
        want a KeyError to crash a task over this."""
        try:
            message = data["choices"][0].get("message", {})
        except (KeyError, IndexError, TypeError):
            return ""

        content = message.get("content")

        if isinstance(content, list):
            # Structured content blocks: join any text parts.
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            ]
            content = " ".join(p for p in parts if p)

        if not content:
            # Fall back to reasoning_content if the model used that field
            # instead (seen on some reasoning-tuned models).
            content = message.get("reasoning_content") or ""

        return str(content).strip()

    def _resolve_category_config(self, category, max_tokens, temperature):
        system_prompt, default_max_tokens = _CATEGORY_CONFIG.get(
            category, (_DEFAULT_SYSTEM_PROMPT, 420)
        )
        return (
            system_prompt,
            max_tokens if max_tokens is not None else default_max_tokens,
            temperature if temperature is not None else _TEMPERATURE,
        )

    def chat_completion(self, model: str, prompt: str, category=None,
                        max_tokens: int = None, temperature: float = None) -> str:
        system_prompt, max_tokens, temperature = self._resolve_category_config(
            category, max_tokens, temperature)
        data = self._post_chat(model, prompt, max_tokens, temperature, system_prompt)
        return self._extract_content(data)

    def chat_completion_with_usage(self, model: str, prompt: str, category=None,
                                   max_tokens: int = None, temperature: float = None):
        """Same as chat_completion, but also returns the token usage dict
        reported by the API (prompt_tokens, completion_tokens, total_tokens).
        Useful for local evaluation to track real cost per call; not needed
        by the harness itself (it measures tokens via its own proxy)."""
        system_prompt, max_tokens, temperature = self._resolve_category_config(
            category, max_tokens, temperature)
        data = self._post_chat(model, prompt, max_tokens, temperature, system_prompt)
        text = self._extract_content(data)
        usage = data.get("usage", {})
        return text, usage
