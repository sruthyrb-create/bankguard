"""Regex PII detection and masking (input masking and output leak check share this).

Masks: full card numbers (any 13-19 digit run, with or without spaces/dashes),
US SSNs, US phone numbers and email addresses. Things that must NOT be masked
in this domain: transaction ids (TXN-000123), account ids (ACC-0001), dates
(2026-10-31), amounts ($1,234.56) and card last-4 ("card ending 1234").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Order matters: cards before phones (a card run contains phone-shaped pieces),
# SSNs before phones (a 9-digit SSN is never a phone, but keep it explicit).
_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_SSN_DASHED = re.compile(r"(?<![\w-])\d{3}[- ]\d{2}[- ]\d{4}(?![\w-])")
_SSN_CONTEXT = re.compile(r"(?i)\b(?:ssn|social security(?: number| no\.?)?)\b\D{0,20}?(\d{9})(?!\d)")
_PHONE = re.compile(
    r"(?<![\w-])(?:\+?1[ .-]?|001[ .-]?)?(?:\(\d{3}\)\s?|\d{3}[ .-]?)\d{3}[ .-]?\d{4}"
    r"(?:\s?(?:x|ext\.?)\s?\d{1,5})?(?![\w-])")
_EMAIL = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")

PII_TYPES = ("CARD", "SSN", "PHONE", "EMAIL")

# gpt-oss often writes ids with a non-breaking hyphen (U+2011): "TXN‑040737". Every check that matches ids,
# phone numbers or dates runs on text with all dash variants turned into "-", or it silently misses them.
_DASHES = re.compile(r"[\u2010-\u2015\u2212\ufe58\ufe63\uff0d]")


def normalize_dashes(text: str) -> str:
    return _DASHES.sub("-", text)


@dataclass
class PIIResult:
    """Masked text plus a count of what was found (never the values themselves)."""

    text: str
    found: dict[str, int] = field(default_factory=dict)

    @property
    def any(self) -> bool:
        return bool(self.found)


def mask_pii(text: str) -> PIIResult:
    """Replace PII with [CARD], [SSN], [PHONE], [EMAIL] placeholders."""
    found: dict[str, int] = {}
    text = normalize_dashes(text)

    def sub(pattern: re.Pattern[str], label: str, s: str, group: int = 0) -> str:
        def repl(m: re.Match[str]) -> str:
            found[label] = found.get(label, 0) + 1
            if group:
                start, end = m.span(group)
                return m.group(0)[: start - m.start()] + f"[{label}]" + m.group(0)[end - m.start():]
            return f"[{label}]"
        return pattern.sub(repl, s)

    out = sub(_EMAIL, "EMAIL", text)
    out = sub(_CARD, "CARD", out)
    out = sub(_SSN_DASHED, "SSN", out)
    out = sub(_SSN_CONTEXT, "SSN", out, group=1)
    out = sub(_PHONE, "PHONE", out)
    return PIIResult(out, found)
