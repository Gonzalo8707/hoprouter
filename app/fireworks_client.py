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
import requests


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

    def _post_chat(self, model: str, prompt: str, max_tokens: int, temperature: float):
        model_id = self._resolve_model(model)
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer in English. Be extremely concise: no "
                        "examples, no restated question, no extra "
                        "commentary. Code tasks: output only the "
                        "function/fix. Math: final answer only, minimal "
                        "working shown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=25)
        resp.raise_for_status()
        return resp.json()

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

    def chat_completion(self, model: str, prompt: str, max_tokens: int = 280,
                        temperature: float = 0.2) -> str:
        data = self._post_chat(model, prompt, max_tokens, temperature)
        return self._extract_content(data)

    def chat_completion_with_usage(self, model: str, prompt: str, max_tokens: int = 280,
                                   temperature: float = 0.2):
        """Same as chat_completion, but also returns the token usage dict
        reported by the API (prompt_tokens, completion_tokens, total_tokens).
        Useful for local evaluation to track real cost per call; not needed
        by the harness itself (it measures tokens via its own proxy)."""
        data = self._post_chat(model, prompt, max_tokens, temperature)
        text = self._extract_content(data)
        usage = data.get("usage", {})
        return text, usage
