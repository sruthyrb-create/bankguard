"""Input and output guardrails, each switchable in configs/guardrails.yaml.

Input (before the model sees the message):
  pii_masking           mask SSN, card, phone, email (never blocks)
  injection_classifier  block if P(injection) >= threshold
  topic_filter          block investment advice and other customers' data
Output (inside the agent loop):
  schema_check          tool-call arguments must match the tool schema; answer must be non-empty
  grounding_check       cited Reg E chunk ids must have been retrieved in this conversation
  day_count_check       "N business/calendar days" in the answer must match policy text or tool output seen
  id_check              (v2) every transaction id in the answer must be on the account or quoted by the customer
  action_claim_check    (v2) flags/disputes the answer claims must match successful tool calls in the conversation
  pii_leak_check        redact PII and other customers' account ids from the answer
  action_policy         open_dispute only after explicit customer confirmation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bankguard.guardrails.injection import HFInjectionClassifier, InjectionScorer
from bankguard.guardrails.output import (ActionDecision, GroundingResult, LeakResult, check_action,
                                         check_action_claims, check_day_counts, check_grounding, check_tool_args,
                                         claimed_actions, redact_leaks, unknown_txn_ids)
from bankguard.guardrails.pii import mask_pii
from bankguard.guardrails.topic import REFUSALS, check_topic

INJECTION_REFUSAL = ("I can't follow instructions that try to change how I work. I'm happy to help with "
                     "your account, transactions, unusual charges, disputes or Regulation E questions.")
GROUNDING_FALLBACK = ("I couldn't back up that policy answer with the Regulation E text I retrieved, so I "
                      "won't guess. Could you ask the question again, or ask about one rule at a time?")
EMPTY_FALLBACK = "Sorry, I couldn't produce an answer. Could you rephrase your question?"
ID_FALLBACK = ("I couldn't verify one of the transaction ids in my answer against your account, so I won't guess. "
               "Could you tell me the transaction id or describe the charge (date, merchant, amount)?")

__all__ = ["GuardrailConfig", "Guardrails", "InputDecision", "check_action", "check_grounding",
           "check_tool_args", "check_day_counts", "check_action_claims", "claimed_actions", "unknown_txn_ids",
           "redact_leaks", "mask_pii", "check_topic", "ActionDecision",
           "GroundingResult", "LeakResult", "INJECTION_REFUSAL", "GROUNDING_FALLBACK", "EMPTY_FALLBACK", "ID_FALLBACK"]


@dataclass
class GuardrailConfig:
    pii_masking: bool = False
    injection_classifier: bool = False
    injection_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    injection_revision: str = "main"
    injection_threshold: float | None = None
    topic_filter: bool = False
    schema_check: bool = False
    grounding_check: bool = False
    grounding_allow_subparagraphs: bool = False
    day_count_check: bool = False
    id_check: bool = False              # v2: transaction ids in the answer must be on the account
    action_claim_check: bool = False    # v2: claimed flags/disputes must match successful tool calls
    pii_leak_check: bool = False
    action_policy: bool = False
    max_repairs: int = 1

    @classmethod
    def from_yaml(cls, data: dict[str, Any]) -> "GuardrailConfig":
        """Build from the dict in configs/guardrails.yaml (sections `input` and `output`)."""
        flat = {**data.get("input", {}), **data.get("output", {})}
        unknown = set(flat) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown guardrail settings: {sorted(unknown)}")
        return cls(**flat)

    @classmethod
    def all_off(cls) -> "GuardrailConfig":
        return cls()

    def enabled(self) -> list[str]:
        names = ["pii_masking", "injection_classifier", "topic_filter", "schema_check", "grounding_check",
                 "day_count_check", "id_check", "action_claim_check", "pii_leak_check", "action_policy"]
        return [n for n in names if getattr(self, n)]


@dataclass
class InputDecision:
    text: str                       # what the model will see (masked if pii_masking is on)
    blocked_by: str | None = None   # guardrail name, or None
    refusal: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


class Guardrails:
    """Holds the config and the (optional) injection classifier; the agent calls into it."""

    def __init__(self, config: GuardrailConfig, scorer: InjectionScorer | None = None) -> None:
        if config.injection_classifier:
            if config.injection_threshold is None:
                raise ValueError("injection_threshold is not set in configs/guardrails.yaml; "
                                 "run `make tune-guardrails` first (or switch the classifier off).")
            if scorer is None:
                scorer = HFInjectionClassifier(config.injection_model, config.injection_revision)
        self.config, self.scorer = config, scorer

    @classmethod
    def from_config(cls, data: dict[str, Any] | None = None, scorer: InjectionScorer | None = None) -> "Guardrails":
        """Load configs/guardrails.yaml (or a given dict)."""
        if data is None:
            from bankguard.config import load_yaml
            data = load_yaml("guardrails")
        return cls(GuardrailConfig.from_yaml(data), scorer)

    def check_input(self, text: str, own_account: str) -> InputDecision:
        """Run the input guardrails: mask PII, injection classifier, topic filter.

        The classifier scores the original text, not the masked one: it runs locally, so the PII never
        leaves the machine, and masking pushed benign scores up on the dev set (0.982 -> 0.999, 0.012 -> 0.092).
        The model, traces and history only see masked text.
        """
        cfg, events, original = self.config, [], text
        if cfg.pii_masking:
            masked = mask_pii(text)
            if masked.any:
                events.append({"guardrail": "pii_masking", "action": "mask", "found": masked.found})
            text = masked.text
        if cfg.injection_classifier and self.scorer is not None:
            score = self.scorer.score(original)
            blocked = score >= float(cfg.injection_threshold)  # type: ignore[arg-type]
            events.append({"guardrail": "injection_classifier", "action": "block" if blocked else "pass",
                           "score": round(score, 4), "threshold": cfg.injection_threshold})
            if blocked:
                return InputDecision(text, "injection_classifier", INJECTION_REFUSAL, events)
        if cfg.topic_filter:
            topic = check_topic(text, own_account)
            if topic.blocked:
                events.append({"guardrail": "topic_filter", "action": "block", "category": topic.category,
                               "rule": topic.rule})
                return InputDecision(text, "topic_filter", REFUSALS[topic.category], events)  # type: ignore[index]
        return InputDecision(text, None, None, events)
