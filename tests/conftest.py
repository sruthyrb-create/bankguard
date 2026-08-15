"""Shared fixtures: one small synthetic database per test session."""

import shutil
from pathlib import Path

import pytest

from bankguard.synth import SynthConfig, generate

SMALL = SynthConfig(seed=7, n_customers=20, days=30)


@pytest.fixture(scope="session")
def base_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("db") / "bank.db"
    generate(path, SMALL)
    return path


@pytest.fixture()
def db(base_db: Path, tmp_path: Path) -> Path:
    """A fresh copy per test, so write tools don't leak between tests."""
    path = tmp_path / "bank.db"
    shutil.copy(base_db, path)
    return path
