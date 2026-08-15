"""Phase 0 environment check: Python, GPU, .env, Groq key and model IDs.

Run with: python scripts/check_env.py
Exits non-zero if a required check fails.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bankguard.config import REPO_ROOT, get_secret, load_env, load_yaml  # noqa: E402


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'OK ' if ok else 'FAIL'}] {label}{' - ' + detail if detail else ''}")
    return ok


def main() -> int:
    results: list[bool] = []
    results.append(check("Python 3.11", sys.version_info[:2] == (3, 11), sys.version.split()[0]))

    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        )
        results.append(check("GPU visible", out.returncode == 0, out.stdout.strip()))
    else:
        results.append(check("GPU visible", False, "nvidia-smi not found"))

    load_env()
    results.append(check(".env file present", (REPO_ROOT / ".env").exists()))
    key = get_secret("GROQ_API_KEY")
    results.append(check("GROQ_API_KEY set", key is not None))

    models = load_yaml("models")
    load_yaml("guardrails")
    results.append(check("configs load", True))

    if key:
        from openai import OpenAI

        client = OpenAI(base_url=models["groq"]["base_url"], api_key=key)
        try:
            available = {m.id for m in client.models.list().data}
        except Exception as exc:  # network or auth problem
            results.append(check("Groq API reachable", False, repr(exc)))
        else:
            results.append(check("Groq API reachable", True, f"{len(available)} models"))
            for role in ("agent_model", "judge_model"):
                mid = models["groq"][role]
                results.append(check(f"Groq {role} '{mid}' available", mid in available))
            print("\nGroq models on your account (pick the judge from a different family):")
            for mid in sorted(available):
                print("   ", mid)

    print()
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
