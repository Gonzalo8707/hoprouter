"""
HopRouter - Category classifier and routing logic.

Decides, for each incoming task, whether it should be solved by:
  - the LOCAL model (counts as 0 tokens toward the score), or
  - a REMOTE Fireworks model (counts against token efficiency).

Strategy:
  1. Classify the task into one of the 8 known categories using cheap
     heuristics (keywords, structure of the prompt) - this classification
     itself must be free (no LLM call).
  2. Based on category + a difficulty heuristic, decide the route.
  3. Categories that are usually "easy" for a small local model
     (sentiment, simple NER, short factual lookups) default to LOCAL.
  4. Categories that need stronger reasoning (multi-step math, logic
     puzzles, code debugging/generation) default to REMOTE, using the
     best-suited allowed model for that category.
"""

import re
from enum import Enum


class Category(str, Enum):
    FACTUAL = "factual_knowledge"
    MATH = "mathematical_reasoning"
    SENTIMENT = "sentiment_classification"
    SUMMARY = "text_summarisation"
    NER = "named_entity_recognition"
    CODE_DEBUG = "code_debugging"
    LOGIC = "logical_reasoning"
    CODE_GEN = "code_generation"
    UNKNOWN = "unknown"


class Route(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


# Keyword heuristics per category. Order matters: more specific categories
# are checked before generic ones.
_PATTERNS = {
    Category.CODE_DEBUG: [
        r"\bbug\b", r"\bdebug", r"\bfix (the|this) code\b", r"\berror\b.*code",
        r"traceback", r"stack trace", r"why (does|is) this code",
    ],
    Category.CODE_GEN: [
        r"\bwrite a function\b", r"\bimplement\b", r"\bwrite code\b",
        r"\bwrite a program\b", r"def \w+\(", r"\bfunction that\b",
    ],
    Category.LOGIC: [
        r"\bpuzzle\b", r"\bif .* then\b.*\bwho\b", r"\ball of the following\b",
        r"\bconstraint", r"\bdeduce\b", r"\bwho (is|owns|lives)\b",
    ],
    Category.MATH: [
        r"\bpercent", r"%", r"\bhow many\b", r"\bcalculate\b",
        r"\d+\s*[\+\-\*/]\s*\d+", r"\bsolve for\b", r"\baverage\b",
    ],
    Category.NER: [
        r"\bextract\b.*(entit|name|person|organi[sz]ation|location|date)",
        r"\bidentify all\b.*(people|organizations|locations|dates)",
        r"\bner\b",
    ],
    Category.SUMMARY: [
        r"\bsummari[sz]e\b", r"\bcondense\b", r"\bin one sentence\b",
        r"\btl;?dr\b", r"\bshorten\b",
    ],
    Category.SENTIMENT: [
        r"\bsentiment\b", r"\bpositive or negative\b", r"\bclassify.*(review|feedback|comment)",
        r"\bhow does .* feel\b",
    ],
    Category.FACTUAL: [
        r"\bwhat is\b", r"\bexplain\b", r"\bdefine\b", r"\bhow does\b",
        r"\bwhy does\b", r"\bdescribe\b",
    ],
}

# Which allowed Fireworks model to prefer per category, when routed REMOTE.
MODEL_PREFERENCE = {
    Category.CODE_DEBUG: "kimi-k2p7-code",
    Category.CODE_GEN: "kimi-k2p7-code",
    Category.LOGIC: "minimax-m3",
    Category.MATH: "minimax-m3",
    Category.NER: "gemma-4-26b-a4b-it",
    Category.SUMMARY: "gemma-4-26b-a4b-it",
    Category.SENTIMENT: "gemma-4-26b-a4b-it",
    Category.FACTUAL: "gemma-4-31b-it",
    Category.UNKNOWN: "gemma-4-31b-it",
}

# Categories the local model is expected to handle reliably on its own,
# without falling below the accuracy threshold.
LOCAL_CAPABLE = {
    Category.SENTIMENT,
    Category.NER,
    Category.FACTUAL,
}


def classify(prompt: str) -> Category:
    text = prompt.lower()
    for category, patterns in _PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text):
                return category
    return Category.UNKNOWN


def decide_route(prompt: str, category: Category) -> Route:
    """
    Decide LOCAL vs REMOTE.

    Extra heuristics beyond category:
      - Very long prompts (likely summarisation of long passages, or
        complex multi-constraint logic) push toward REMOTE even if the
        category is normally LOCAL_CAPABLE.
      - Very short, simple prompts in a LOCAL_CAPABLE category stay LOCAL.
    """
    length = len(prompt.split())

    if category in LOCAL_CAPABLE and length < 120:
        return Route.LOCAL

    return Route.REMOTE


def route_task(prompt: str):
    """Returns (category, route, preferred_model_if_remote)."""
    category = classify(prompt)
    route = decide_route(prompt, category)
    model = MODEL_PREFERENCE.get(category) if route == Route.REMOTE else None
    return category, route, model
