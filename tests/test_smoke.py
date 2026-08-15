"""Phase 0 smoke tests: config loads and repo rules hold."""

from pathlib import Path

from bankguard.config import REPO_ROOT, load_yaml

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".toml", ".cfg", ""}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "traces", ".venv"}


def test_configs_load() -> None:
    models = load_yaml("models")
    guards = load_yaml("guardrails")
    assert "groq" in models and "local" in models
    assert set(guards) == {"input", "output"}


def test_env_example_has_groq_key() -> None:
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GROQ_API_KEY=" in text


def test_no_real_bank_branding() -> None:
    """Rule 3: the repo only ever mentions the fictional Northwind Bank."""
    banned = ["capital" + " one", "capital" + "one"]  # split so this file doesn't match itself
    hits: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts) or not path.is_file():
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        hits += [f"{path.relative_to(REPO_ROOT)}: {b}" for b in banned if b in text]
    assert not hits, f"Real bank branding found: {hits}"
