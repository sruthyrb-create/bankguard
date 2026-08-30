"""Rule-based topic filter (input guardrail): investment advice and other customers' data.

The rules look for *intent*, not topic words: "is a stock purchase on my debit
card covered by Reg E?" and "a crypto exchange charged me twice" are normal
support questions and must pass. Every rule here is a possible false refusal,
so the dev set includes benign messages that use the same words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ASSET = r"(?:stocks?|shares?|equit(?:y|ies)|bonds?|funds?|etfs?|index funds?|mutual funds?|crypto(?:currency)?|" \
         r"bitcoin|btc|ethereum|coins?|401\s?\(?k\)?|ira|roth|portfolio|options)"

INVESTMENT_PATTERNS = [
    re.compile(rf"\bshould i (?:invest|buy|sell|move|put|hold|short)\b.*\b{_ASSET}", re.I),
    re.compile(rf"\bshould i (?:invest|move .* into)\b", re.I),
    re.compile(r"\b(?:where|how) (?:should|can|do) i invest\b", re.I),
    re.compile(r"\binvest(?:ing)? (?:in|my|our)\b", re.I),
    re.compile(rf"\b(?:which|what) {_ASSET} (?:should|to) i (?:buy|sell|pick|get)\b", re.I),
    re.compile(rf"\b(?:which|what) {_ASSET} (?:is|are) (?:the )?best\b", re.I),
    re.compile(rf"\bbest {_ASSET}\b", re.I),
    re.compile(r"\b(?:good|bad|smart|safe) investment\b", re.I),
    re.compile(r"\b(?:stock|crypto|trading|investment) (?:tips?|picks?|advice|recommendations?)\b", re.I),
]

_PERSON = r"(?:neighbou?r|friend|wife|husband|spouse|partner|brother|sister|mother|father|mom|mum|dad|son|" \
          r"daughter|boss|coworker|co-worker|colleague|roommate|ex|cousin|uncle|aunt|tenant|landlord)"

OTHER_CUSTOMER_PATTERNS = [
    re.compile(r"\b(?:another|other|different) customers?\b|\b(?:every|each) customer'?s\b|\ball customers'", re.I),
    re.compile(r"\b(?:which|what) customers?\b", re.I),
    re.compile(r"\bcustomers? (?:who|that|with|in|from|living)\b", re.I),
    re.compile(r"\b(?:list|database) of customers\b|\bcustomer (?:list|database|records)\b", re.I),
    re.compile(r"\bsomeone else'?s\b|\bsomebody else'?s\b", re.I),
    re.compile(rf"\b{_PERSON}'?s (?:account|transactions?|card|balance|statement|charges?|disputes?)\b", re.I),
]

_ACCOUNT_ID = re.compile(r"\bACC-\d{4}\b", re.I)

REFUSALS = {
    "investment_advice": "I can't give investment advice. I can help with your Northwind Bank account, "
                         "transactions, unusual charges, disputes and Regulation E questions.",
    "other_customer": "I can only help with your own account, so I can't look up or discuss other "
                      "customers' information.",
}


@dataclass
class TopicResult:
    category: str | None  # None, "investment_advice" or "other_customer"
    rule: str | None = None

    @property
    def blocked(self) -> bool:
        return self.category is not None


def check_topic(text: str, own_account: str) -> TopicResult:
    """Return the first off-limits category the message matches, if any."""
    for acc in _ACCOUNT_ID.findall(text):
        if acc.upper() != own_account.upper():
            return TopicResult("other_customer", f"account_id:{acc.upper()}")
    for p in OTHER_CUSTOMER_PATTERNS:
        if p.search(text):
            return TopicResult("other_customer", p.pattern[:60])
    for p in INVESTMENT_PATTERNS:
        if p.search(text):
            return TopicResult("investment_advice", p.pattern[:60])
    return TopicResult(None)
