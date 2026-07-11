"""
HopRouter - Category classifier and routing logic.

Decides, for each incoming task, whether it should be solved by:
  - the LOCAL model (counts as 0 tokens toward the score), or
  - a REMOTE Fireworks model (counts against token efficiency).

Strategy:
  1. If the incoming task already tells us its category (some public
     example sets from the harness include a "category"/"type" field
     alongside the prompt), trust it - this is free, zero-risk, and
     removes classification error entirely for that task. See
     `resolve_category()`.
  2. Otherwise, classify the task into one of the 8 known categories using
     cheap heuristics (keywords, structure of the prompt) - this
     classification itself must be free (no LLM call).
  3. Based on category + a difficulty heuristic, decide the route.
  4. Categories that are usually "easy" for a small local model default to
     LOCAL (currently none - see LOCAL_CAPABLE below).
  5. Categories that need stronger reasoning (multi-step math, logic
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


# Maps the canonical string value of every known category to its Enum
# member, plus a couple of very likely alias spellings, for use by
# resolve_category() below. Built once at import time.
_CATEGORY_BY_VALUE = {c.value: c for c in Category if c != Category.UNKNOWN}
_CATEGORY_ALIASES = {
    # American spelling in case the harness (or a variant of it) doesn't
    # use the British "summarisation" spelling used in Category.SUMMARY.
    "text_summarization": Category.SUMMARY,
}


def resolve_category(explicit_category) -> "Category | None":
    """
    If the task dict already carries a category label, normalize it and
    return the matching Category - this is strictly more reliable than
    guessing from the prompt text, since it comes straight from the task
    spec instead of a regex heuristic. Returns None if there's nothing
    usable, so the caller can fall back to classify(prompt).
    """
    if not explicit_category or not isinstance(explicit_category, str):
        return None
    normalized = re.sub(r"[\s\-]+", "_", explicit_category.strip().lower())
    if normalized in _CATEGORY_BY_VALUE:
        return _CATEGORY_BY_VALUE[normalized]
    if normalized in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[normalized]
    return None


# Keyword heuristics per category. Order matters: more specific categories
# are checked before generic ones (FACTUAL patterns are broad, e.g.
# "what is", so FACTUAL is checked last to avoid swallowing math/NER/etc.
# prompts that happen to also contain "what is").
_PATTERNS = {
    Category.CODE_DEBUG: [
        r"\bbug\b", r"\bdebug", r"\bfix (the|this) code\b", r"\berror\b.*code",
        r"traceback", r"stack trace", r"why (does|is) this code",
        r"\bcorrect(ed)? version\b", r"\bwhat('s| is) wrong with\b", r"\bfix (it|this)\b",
        r"\bfind the (bug|issue|error)\b",
    ],
    Category.CODE_GEN: [
        r"\bwrite a function\b", r"\bimplement\b", r"\bwrite code\b",
        r"\bwrite a program\b", r"def \w+\(", r"\bfunction that\b",
        r"\bcreate a function\b", r"\bwrite a (python|javascript|java)\b",
    ],
    Category.LOGIC: [
        r"\bpuzzle\b", r"\bif .* then\b.*\bwho\b", r"\ball of the following\b",
        r"\bconstraint", r"\bdeduce\b", r"\bwho (is|owns|lives|finished)\b",
        r"\bfinished (first|last|second|third|\d\w{0,2})\b",
        r"\b(1st|2nd|3rd|4th|first|second|third|fourth)\s+place\b",
        r"\border(ed|ing)?\b.*\b(place|position|rank)\b",
        r"\bmust be true\b", r"\bwho is (the )?(oldest|youngest|tallest|shortest)\b",
    ],
    Category.MATH: [
        r"\bpercent", r"%", r"\bhow many\b", r"\bcalculate\b",
        r"\d+\s*[\+\-\*/]\s*\d+", r"\bsolve for\b", r"\baverage\b",
        r"\bhow (fast|long|much)\b", r"\bspeed\b", r"\bkm/h\b", r"\bmph\b",
        r"\d+\s*(km|kg|cm|mm|m|minutes?|hours?|liters?|dollars?)\b",
        r"\$\d+", r"\bdiscount\b", r"\bnew price\b", r"\bincrease\b.*\bprice\b",
        r"\btotal cost\b",
    ],
    Category.NER: [
        r"\bextract\b.*(entit|name|person|organi[sz]ation|location|date)",
        r"\bidentify all\b.*(people|organizations|locations|dates)",
        r"\bner\b", r"\bnamed entit",
        r"\blist all\b.*(people|person|organi[sz]ations?|locations?|dates?|entit)",
        r"\b(people|persons|organi[sz]ations?|locations?|dates?)\b.*\bmentioned\b",
    ],
    Category.SUMMARY: [
        r"\bsummari[sz]e\b", r"\bsummary\b", r"\bcondense\b", r"\bin one sentence\b",
        r"\btl;?dr\b", r"\bshorten\b", r"\bkey points\b", r"\bmain idea\b",
        r"\bmain points\b", r"\bbullet points?\b", r"\brecap\b",
        r"\bin \d+ (sentences?|bullets?|points?|words?)\b",
    ],
    Category.SENTIMENT: [
        r"\bsentiment\b", r"\bpositive or negative\b", r"\bclassify.*(review|feedback|comment)",
        r"\bhow does .* feel\b", r"\btone of\b", r"\battitude\b",
    ],
    Category.FACTUAL: [
        r"\bwhat is\b", r"\bexplain\b", r"\bdefine\b", r"\bhow does\b",
        r"\bwhy does\b", r"\bdescribe\b", r"\bwhat causes\b",
    ],
}


MODEL_PREFERENCE = {
    Category.CODE_DEBUG: "kimi-k2p7-code",
    Category.CODE_GEN: "kimi-k2p7-code",
    Category.LOGIC: "minimax-m3",
    Category.MATH: "minimax-m3",
    Category.NER: "minimax-m3",
    Category.SUMMARY: "minimax-m3",
    Category.SENTIMENT: "minimax-m3",
    Category.FACTUAL: "minimax-m3",
    Category.UNKNOWN: "minimax-m3",
}

# Categories the local model is expected to handle reliably on its own,
# without falling below the accuracy threshold.
#
# NOTE: every category was tested locally and dropped - even with prompt
# tuning, the small local model (Qwen2.5-0.5B) is not reliable enough
# against the real (hidden) harness variants, which turned out to be
# harder than our own homemade eval set. Given the accuracy gate is
# all-or-nothing, we accept the token cost of routing everything remotely
# rather than risk an incomplete/wrong answer. The local model remains
# wired in only as an emergency fallback (see main.py) for when Fireworks
# itself is unreachable.
LOCAL_CAPABLE = set()


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

    With LOCAL_CAPABLE currently empty, this always returns REMOTE - kept
    as-is (rather than special-cased away) so re-enabling any category for
    local handling later is a one-line change.
    """
    length = len(prompt.split())

    if category in LOCAL_CAPABLE and length < 120:
        return Route.LOCAL

    return Route.REMOTE


def route_task(prompt: str, explicit_category=None):
    """
    Returns (category, route, preferred_model_if_remote).

    `explicit_category` is an optional category label taken straight from
    the task (e.g. task.get("category")), if the input schema provides
    one. When present and recognized, it is used instead of the regex
    classifier - see resolve_category().
    """
    category = resolve_category(explicit_category) or classify(prompt)
    route = decide_route(prompt, category)
    model = MODEL_PREFERENCE.get(category) if route == Route.REMOTE else None
    return category, route, model
