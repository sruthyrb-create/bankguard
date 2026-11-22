"""Eval scenario format, loading and validation.

A scenario is one conversation (1-3 user turns) on one account, plus what a correct
run looks like. Rule checks are graded in code (bankguard.eval.grading); `reference`
is what the LLM judge and the human labeller compare the answer against.

checks (all optional):
  tools_required   tool names that must be called at least once
  answer_all       strings that must all appear in the final answer (normalised, see grading.normalise)
  answer_any       list of alternatives lists; each list needs at least one hit
  answer_none      strings that must not appear in the final answer
  citations        {"prefixes": ["1005.11"]}: the final answer must cite, every cited id must have been
                   retrieved in the conversation, and at least one must start with a prefix
  disputes         exact set of transaction ids that must have a dispute after the run (default: none)
  flags_required   transaction ids that must be flagged
  flags_allowed    ids that may be flagged (default: flags_required)
  clarify          {"mention": [ids]}: the final answer must name every id and ask a question
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bankguard.config import REPO_ROOT

EVAL_DIR = REPO_ROOT / "data" / "scenarios" / "eval"
HELDOUT_DIR = REPO_ROOT / "data" / "scenarios" / "heldout"
# Phase 6: benign messages that look like attacks, run as ordinary scenarios to measure false refusals.
REDTEAM_BENIGN_DIR = REPO_ROOT / "data" / "scenarios" / "redteam_benign"
SETS = {"main": EVAL_DIR, "heldout": HELDOUT_DIR, "redteam_benign": REDTEAM_BENIGN_DIR}
CATEGORIES = ("lookup", "flag", "dispute", "policy", "refusal", "error_handling")
CHECK_KEYS = {"tools_required", "answer_all", "answer_any", "answer_none", "citations", "disputes",
              "flags_required", "flags_allowed", "clarify"}


@dataclass
class Scenario:
    id: str
    category: str
    account: str
    turns: list[str]
    reference: str
    expect_refusal: bool = False
    checks: dict[str, Any] = field(default_factory=dict)
    facts: dict[str, dict[str, Any]] = field(default_factory=dict)  # txn_id -> fields, verified against bank.db

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scenario":
        sc = cls(**d)
        sc.validate()
        return sc

    def validate(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.id}: unknown category {self.category!r}")
        if not self.turns or not all(isinstance(t, str) and t.strip() for t in self.turns):
            raise ValueError(f"{self.id}: turns must be non-empty strings")
        unknown = set(self.checks) - CHECK_KEYS
        if unknown:
            raise ValueError(f"{self.id}: unknown checks {sorted(unknown)}")
        if self.expect_refusal != (self.category == "refusal"):
            raise ValueError(f"{self.id}: expect_refusal must be true exactly for refusal scenarios")


def load_scenarios(path: Path | None = None, scenario_set: str = "main") -> list[Scenario]:
    """Load every *.jsonl of a scenario set (main = data/scenarios/eval, heldout = data/scenarios/heldout,
    redteam_benign = data/scenarios/redteam_benign), or one file."""
    paths = [path] if path else sorted(SETS[scenario_set].glob("*.jsonl"))
    out: list[Scenario] = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as f:
            out += [Scenario.from_dict(json.loads(line)) for line in f if line.strip()]
    ids = [s.id for s in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario ids")
    return out


def all_references() -> dict[str, str]:
    """scenario id -> reference, across every set (the judge and the labelling sheet need both)."""
    return {s.id: s.reference for name in SETS for s in load_scenarios(scenario_set=name)}
