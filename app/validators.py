"""
Lightweight, rule-based sanity checks for answers produced by the LOCAL
model. These are NOT accuracy checks (we can't judge correctness without
an LLM-judge) - they only catch obviously broken/malformed output, so we
can escalate to Fireworks instead of risking the accuracy gate on a
category we routed locally to save tokens.

Philosophy: the accuracy gate is all-or-nothing per the scoring rules, so
a cheap local answer that is well-formed but wrong still risks the whole
submission. These checks catch the "obviously broken" cases cheaply
(empty output, wrong structure, fallback message) - they are a floor, not
a full correctness guarantee.
"""

import json
import re

from router import Category

_FALLBACK_MARKERS = (
    "unable to process",
    "error processing task",
)

_SENTIMENT_LABELS = ("positive", "negative", "neutral", "mixed")


def _is_fallback_or_empty(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    lowered = answer.lower()
    return any(marker in lowered for marker in _FALLBACK_MARKERS)


def _valid_sentiment(answer: str) -> bool:
    lowered = answer.lower()
    return any(label in lowered for label in _SENTIMENT_LABELS)


def _valid_ner(answer: str) -> bool:
    # Expecting JSON output per the task spec. Accept either a raw JSON
    # blob or JSON embedded in surrounding text (extract the first
    # {...} or [...] block).
    match = re.search(r"(\{.*\}|\[.*\])", answer, re.DOTALL)
    if not match:
        return False
    try:
        json.loads(match.group(1))
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _valid_factual(answer: str) -> bool:
    # No strong structural signal available; just require a substantive,
    # non-trivial response.
    return len(answer.split()) >= 8


_VALIDATORS = {
    Category.SENTIMENT: _valid_sentiment,
    Category.NER: _valid_ner,
    Category.FACTUAL: _valid_factual,
}


def passes_local_safety_check(category: Category, answer: str) -> bool:
    """
    Returns True if the local answer is well-formed enough to keep.
    Returns False if we should escalate to a remote Fireworks model instead.
    """
    if _is_fallback_or_empty(answer):
        return False

    validator = _VALIDATORS.get(category)
    if validator is None:
        # No specific validator for this category (shouldn't normally
        # happen, since only LOCAL_CAPABLE categories reach this check) -
        # default to accepting a non-empty answer.
        return True

    return validator(answer)
