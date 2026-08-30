"""Prompt-injection classifier (input guardrail).

`InjectionScorer` is anything with `score(text) -> float` (probability that the
text is an injection). The real one wraps a Hugging Face sequence classifier;
tests use a fake, so no model download is needed to run the test suite.
"""

from __future__ import annotations

from typing import Protocol


class InjectionScorer(Protocol):
    name: str

    def score(self, text: str) -> float: ...


class HFInjectionClassifier:
    """protectai/deberta-v3-base-prompt-injection-v2 (or any binary classifier with an INJECTION label).

    Inputs longer than max_length tokens are truncated, so an injection placed
    after ~512 tokens is not seen. That limit is logged in DECISIONS.md.
    """

    def __init__(self, model_name: str, revision: str = "main", device: str | None = None,
                 max_length: int = 512) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = f"{model_name}@{revision}"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
        self.model.to(self.device).eval()
        self.max_length = max_length
        labels = {v.upper(): int(k) for k, v in self.model.config.id2label.items()}
        if "INJECTION" not in labels:
            raise ValueError(f"{model_name} has no INJECTION label (labels: {list(labels)})")
        self.injection_idx = labels["INJECTION"]

    def score(self, text: str) -> float:
        import torch

        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)[0]
        return float(probs[self.injection_idx])

