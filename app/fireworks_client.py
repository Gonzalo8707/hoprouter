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
    "You are a precise assistant. Answer in English. Be concise but make "
    "sure your answer is COMPLETE - do not cut off mid-sentence or "
    "mid-structure. Never sacrifice correctness or completeness for "
    "brevity."
)

# Per-category system prompt + max_tokens. Tailoring the instructions to the
# exact expected output shape (a single label, strict JSON keys, a bare
# numeric answer, etc.) both improves correctness on structured categories
# and keeps completions short - the two things we're optimizing for are not
# in tension here, a sharper prompt is cheaper AND more accurate than a
# generic "be complete" instruction with a large flat token budget.
_CATEGORY_CONFIG = {
    Category.FACTUAL: (
        "Answer the question directly and factually in 2-4 sentences. "
        "Be accurate and complete, but do not pad with extra examples.",
        320,
    ),
    Category.MATH: (
        "Solve the problem. Show the minimal working needed, then give the "
        "final answer clearly on its own, e.g. 'Answer: <value>'. Never "
        "omit the final answer.",
        200,
    ),
    Category.SENTIMENT: (
        "Classify the sentiment as exactly one of: positive, negative, "
        "neutral, or mixed. State the label first, then a one-sentence "
        "justification.",
        150,
    ),
    Category.SUMMARY: (
        "Summarize the text in exactly one sentence. Do not add "
        "commentary, preamble, or repeat the instructions.",
        210,
    ),
    Category.NER: (
        "Extract named entities and return ONLY a valid JSON object with "
        "exactly these keys: \"person\", \"organization\", \"location\", "
        "\"date\" - each mapped to an array of strings found in the text "
        "(use an empty array if none are found for that key, but never "
        "omit a key). Return only the JSON, no extra text.",
        380,
    ),
    Category.CODE_DEBUG: (
        "Identify the bug, then return the corrected function in a single "
        "code block. State the bug in one short sentence before the code.",
        420,
    ),
    Category.LOGIC: (
        "Reason step by step internally, but keep any shown reasoning "
        "brief. End with the final answer clearly labeled 'Answer:'.",
        260,
    ),
    Category.CODE_GEN: (
        "Implement exactly what is requested as a single, complete, "
        "working function, including the signature and body. No extra "
        "explanation unless explicitly asked for one.",
        430,
    ),
}

_TEMPERATURE = 0.0


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

        # One short retry on transient failures (timeout, connection error,
        # 429, 5xx). Without this, a single infra hiccup on Fireworks' side
        # sends that task straight to the weak local fallback model, which
        # risks the all-or-nothing accuracy gate over a problem that had
        # nothing to do with the task itself. Two attempts at 12s each stay
        # well under the harness's 30s-per-request hard limit.
        last_exc = None
        for attempt in range(2):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=12)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                last_exc = e
                if attempt == 0:
                    time.sleep(0.5)
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
            category, (_DEFAULT_SYSTEM_PROMPT, 300)
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
