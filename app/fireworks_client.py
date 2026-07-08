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
        self.allowed_models = [m.strip() for m in allowed.split(",") if m.strip()]

        if not self.api_key:
            raise ConfigError("FIREWORKS_API_KEY is not set in the environment")
        if not self.base_url:
            raise ConfigError("FIREWORKS_BASE_URL is not set in the environment")
        if not self.allowed_models:
            raise ConfigError("ALLOWED_MODELS is not set in the environment")

    def _resolve_model(self, requested_model: str) -> str:
        if requested_model in self.allowed_models:
            return requested_model
        # Fallback: if our preferred model isn't in the allowed list for some
        # reason (e.g. changed on launch day), use the first allowed model
        # rather than failing the whole task.
        return self.allowed_models[0]

    def chat_completion(self, model: str, prompt: str, max_tokens: int = 512,
                         temperature: float = 0.2) -> str:
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
                        "You are a precise, concise assistant. Always answer "
                        "in English, regardless of the input language. "
                        "Follow the requested output format exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def chat_completion_with_usage(self, model: str, prompt: str, max_tokens: int = 512,
                                    temperature: float = 0.2):
        """Same as chat_completion, but also returns the token usage dict
        reported by the API (prompt_tokens, completion_tokens, total_tokens).
        Useful for local evaluation to track real cost per call; not needed
        by the harness itself (it measures tokens via its own proxy)."""
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
                        "You are a precise, concise assistant. Always answer "
                        "in English, regardless of the input language. "
                        "Follow the requested output format exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        return text, usage
