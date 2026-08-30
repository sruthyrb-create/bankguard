"""Output-side guardrails: tool-call schema check, grounding check, PII-leak check, action policy.

All four are plain functions over strings and dicts, so they are unit-tested
without a model. The agent decides what to do with a violation (deny the tool
call, ask the model to repair its answer, or redact).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from bankguard.guardrails.pii import mask_pii

# ---------------------------------------------------------------- schema check

# Formats the tool schemas don't state (Groq rejects some JSON-schema keywords,
# so the schemas sent to the model stay simple and the stricter check lives here).
ID_PATTERNS = {"account_id": re.compile(r"^ACC-\d{4}$"), "transaction_id": re.compile(r"^TXN-\d{6}$")}
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
    "object": (dict,), "array": (list,), "null": (type(None),),
}


def check_tool_args(name: str, args: Any, schemas: list[dict[str, Any]]) -> list[str]:
    """Validate a tool call against its JSON schema; return a list of problems (empty = valid)."""
    spec = next((s["function"] for s in schemas if s["function"]["name"] == name), None)
    if spec is None:
        return [f"unknown tool {name!r}"]
    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]
    params = spec["parameters"]
    props: dict[str, Any] = params.get("properties", {})
    problems = [f"missing required parameter {r!r}" for r in params.get("required", []) if r not in args]
    if params.get("additionalProperties") is False:
        problems += [f"unexpected parameter {k!r}" for k in args if k not in props]
    for key, value in args.items():
        if key not in props:
            continue
        types = props[key].get("type", [])
        types = [types] if isinstance(types, str) else types
        ok_types = tuple(t for name_ in types for t in _JSON_TYPES.get(name_, ()))
        if ok_types and (not isinstance(value, ok_types) or (isinstance(value, bool) and "boolean" not in types)):
            problems.append(f"{key!r} must be {' or '.join(types)}, got {type(value).__name__}")
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            lo, hi = props[key].get("minimum"), props[key].get("maximum")
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                problems.append(f"{key!r} must be between {lo} and {hi}")
        if key in ID_PATTERNS and isinstance(value, str) and not ID_PATTERNS[key].match(value):
            problems.append(f"{key!r} has the wrong format: {value!r}")
        if isinstance(value, str) and key in ("reason", "query") and not value.strip():
            problems.append(f"{key!r} must not be empty")
    return problems


# ------------------------------------------------------------- grounding check

# Chunk ids look like 1005.11(c)#1 or 1005.2. Accept [id] and the full-width 【id】 gpt-oss also uses.
_ID = r"1005\.\d+[a-z]?(?:\([a-z0-9]+\))*(?:#\d+)?"
CITATION = re.compile(rf"\[\s*({_ID})\s*\]|【\s*({_ID})\s*】")


@dataclass
class GroundingResult:
    cited: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unsupported


def check_grounding(answer: str, retrieved_ids: set[str]) -> GroundingResult:
    """Every cited chunk id must have been retrieved (the agent passes ids seen in this conversation).

    A citation without the #n suffix (e.g. [1005.11(c)]) counts as supported if
    some retrieved chunk of that paragraph exists, since the model often drops it.
    """
    cited = [a or b for a, b in CITATION.findall(answer)]
    bases = {r.split("#")[0] for r in retrieved_ids}
    unsupported = [c for c in cited if c not in retrieved_ids and not ("#" not in c and c in bases)]
    return GroundingResult(cited, sorted(set(unsupported)))


# ----------------------------------------------------------- day-count check

# gpt-oss writes non-breaking hyphens and narrow spaces ("10‑day", "45 days"); normalise them first.
_DASHES = re.compile(r"[-\u2010-\u2015\u2212]")
_DAY_COUNT = re.compile(r"\b(\d{1,3}) (?:(business|calendar) )?days?\b")
_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
          "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_WORD_NUM = {w: i for i, w in enumerate(_UNITS)} | {w: 10 * i for i, w in enumerate(_TENS) if w}
_NUMBER_WORDS = re.compile(
    rf"\b({'|'.join(_TENS[2:])})(?: ({'|'.join(_UNITS[1:10])}))?\b|\b({'|'.join(_UNITS)})\b")


def _words_to_digits(text: str) -> str:
    """'three' -> '3', 'forty five' -> '45' (hyphens are already spaces). Regulation E spells out small
    numbers ("three business days"), while models often write digits."""
    def repl(m: re.Match[str]) -> str:
        if m.group(3):
            return str(_WORD_NUM[m.group(3)])
        return str(_WORD_NUM[m.group(1)] + (_WORD_NUM[m.group(2)] if m.group(2) else 0))
    return _NUMBER_WORDS.sub(repl, text)


def day_counts(text: str) -> set[tuple[int, str]]:
    """Return {(n, unit)} for every "n days", "three business days", "n-calendar-day" in the text.

    unit is "business", "calendar" or "plain" (no qualifier).
    """
    norm = _words_to_digits(" ".join(_DASHES.sub(" ", text.lower()).split()))
    return {(int(m.group(1)), m.group(2) or "plain") for m in _DAY_COUNT.finditer(norm)}


def check_day_counts(answer: str, sources: list[str]) -> list[str]:
    """Day counts the answer qualifies as business or calendar days must match the sources.

    "N business days" needs "N business days" in a source. "N calendar days" needs
    "N calendar days" or an unqualified "N days" (Regulation E counts plain days as
    calendar days). Unqualified counts in the answer are not checked.
    Sources are the tool outputs seen in this conversation (policy text, dispute timelines).
    """
    found: set[tuple[int, str]] = set().union(*(day_counts(s) for s in sources)) if sources else set()
    bad = []
    for n, unit in day_counts(answer):
        if unit == "business" and (n, "business") not in found:
            bad.append(f"{n} business days")
        elif unit == "calendar" and not {(n, "calendar"), (n, "plain")} & found:
            bad.append(f"{n} calendar days")
    return sorted(bad)


# ------------------------------------------------------------- PII-leak check

_ACCOUNT_ID = re.compile(r"\bACC-\d{4}\b")


@dataclass
class LeakResult:
    text: str
    found: dict[str, int] = field(default_factory=dict)

    @property
    def leaked(self) -> bool:
        return bool(self.found)


def redact_leaks(answer: str, own_account: str) -> LeakResult:
    """Mask PII and any account id that is not the session's own account."""
    masked = mask_pii(answer)
    found = dict(masked.found)
    n_other = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal n_other
        if m.group(0) == own_account:
            return m.group(0)
        n_other += 1
        return "[OTHER_ACCOUNT]"

    text = _ACCOUNT_ID.sub(repl, masked.text)
    if n_other:
        found["OTHER_ACCOUNT"] = n_other
    return LeakResult(text, found)


# --------------------------------------------------------------- action policy

GUARDED_TOOLS = {"open_dispute"}
_AFFIRM = re.compile(
    r"\b(?:yes|yeah|yep|yup|sure|ok(?:ay)?|confirm(?:ed)?|correct|go ahead|proceed|please do|do it|"
    r"(?:open|file|start|raise|submit) (?:a |the )?dispute|dispute (?:it|that|this|them|both|txn-\d{6}))\b", re.I)
# A refusal or postponement of the action itself; "I don't recognise this charge" is not one.
_NEGATE = re.compile(
    r"^\W*(?:no|nope|nah)\b|\b(?:don[’']?t|do not|not yet|no need to|rather not)\s+(?:\w+\s+){0,2}?"
    r"(?:open|file|dispute|do|proceed|go ahead|start|submit|raise)\b|\b(?:hold off|cancel|never ?mind|"
    r"wait)\b", re.I)
_TXN = re.compile(r"\bTXN-\d{6}\b", re.I)


@dataclass
class ActionDecision:
    allowed: bool
    reason: str


def check_action(tool: str, args: dict[str, Any], messages: list[dict[str, Any]]) -> ActionDecision:
    """Allow a write action only if the customer explicitly confirmed it for this transaction.

    Explicit confirmation means: the latest user message is affirmative (and not a
    negation), and the transaction id appears either in that user message or as the
    only transaction id in the assistant message right before it (the assistant named
    it and asked). If that question named several transactions, the user must name one.
    """
    if tool not in GUARDED_TOOLS:
        return ActionDecision(True, "not a guarded tool")
    txn = str(args.get("transaction_id", "")).upper()
    user_idx = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=None)
    if user_idx is None:
        return ActionDecision(False, "no user message")
    user_text = str(messages[user_idx].get("content") or "")
    if _NEGATE.search(user_text):
        return ActionDecision(False, "the latest user message declines or postpones")
    if not _AFFIRM.search(user_text):
        return ActionDecision(False, "the latest user message does not explicitly confirm a dispute")
    if txn in {t.upper() for t in _TXN.findall(user_text)}:
        return ActionDecision(True, "user confirmed and named the transaction")
    prev_assistant = next((m for m in reversed(messages[:user_idx])
                           if m.get("role") == "assistant" and m.get("content")), None)
    named = {t.upper() for t in _TXN.findall(str(prev_assistant["content"]))} if prev_assistant else set()
    if named == {txn}:
        return ActionDecision(True, "user confirmed the transaction the assistant named")
    if txn in named:
        return ActionDecision(False, "the assistant's question named several transactions; the customer "
                                     f"must confirm {txn} by id")
    return ActionDecision(False, f"{txn or 'the transaction'} was not named by the user or in the "
                                 "assistant's confirmation question")
