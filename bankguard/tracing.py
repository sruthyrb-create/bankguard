"""Append-only JSONL trace logger (schema in the spec, section 14)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bankguard.config import REPO_ROOT

TRACE_DIR = REPO_ROOT / "traces"


class Tracer:
    """Writes one JSON object per line to traces/<run_id>.jsonl."""

    def __init__(self, run_id: str, path: str | Path | None = None) -> None:
        self.run_id = run_id
        self.path = Path(path) if path else TRACE_DIR / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, session_id: str, step: int, event: str, **fields: Any) -> dict[str, Any]:
        """Record one event (llm_call | tool_call | guardrail | final | error) and return it."""
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                  "run_id": self.run_id, "session_id": session_id, "step": step, "event": event, **fields}
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return record


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    """Load a trace file back into a list of events."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
