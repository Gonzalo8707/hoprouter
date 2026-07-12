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
# History: with the old Qwen2.5-0.5B every category was tested and
# dropped (the 0.5B caused the 73.7% gate failure - too weak for the
# hidden harness variants). The local model is now Qwen2.5-3B-Instruct
# (Q4_K_M via llama.cpp, see local_model.py) - the largest model that
# fits the scoring environment's 4GB RAM / 2 vCPU budget (7B would be
# OOM-killed; 2-3B 4-bit is what the Participant Guide recommends) -
# so the four "simple" categories are routed local again.
# Every local answer still passes through validators.py, and anything
# malformed/incomplete escalates to Fireworks (see main.py) - so the
# accuracy floor is the remote design, and local routing only removes
# tokens from tasks where the 3B produced a well-formed answer.
#
# Reasoning-heavy categories (math, logic, both code categories) stay
# remote: that's where hidden reasoning earns its cost.
LOCAL_CAPABLE = {Category.FACTUAL, Category.SENTIMENT,
                 Category.SUMMARY, Category.NER}

# Per-category prompt-length ceilings (in words) for the local route.
# Longer prompts go remote: they carry more content to get right (long
# passages to summarize, more entities to extract) AND cost more
# prompt-processing time - on the scoring host's 2 vCPUs, prompt eval
# alone eats into the 30s budget, so ceilings are sized to leave the
# generation phase most of the deadline. SUMMARY/NER legitimately
# include a passage in the prompt, so their ceilings are higher than a
# bare question's - a single global threshold would send nearly every
# summary remote and waste the local model where it saves the most.
_LOCAL_MAX_WORDS = {
    Category.FACTUAL: 100,
    Category.SENTIMENT: 150,
    Category.SUMMARY: 300,
    Category.NER: 180,
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

    A LOCAL_CAPABLE category goes local only while the prompt stays under
    that category's length ceiling (see _LOCAL_MAX_WORDS); anything
    longer, and every non-local-capable category, goes remote.
    """
    length = len(prompt.split())

    if category in LOCAL_CAPABLE and length <= _LOCAL_MAX_WORDS[category]:
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
