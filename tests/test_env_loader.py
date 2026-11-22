"""Tests for the stdlib .env loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.app.env_loader import _parse_dotenv, load_dotenv


def test_parse_dotenv_handles_comments_quotes_and_export() -> None:
    text = """
# comment
GROQ_API_KEY=gsk_test
export STT_PROVIDER="groq"
EMPTY=
  SPACED  =  value with spaces  
QUOTED='single quoted'
"""
    parsed = _parse_dotenv(text)
    assert parsed["GROQ_API_KEY"] == "gsk_test"
    assert parsed["STT_PROVIDER"] == "groq"
    assert parsed["EMPTY"] == ""
    assert parsed["SPACED"] == "value with spaces"
    assert parsed["QUOTED"] == "single quoted"


def test_load_dotenv_does_not_overwrite_existing_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("NEW_KEY=from_file\nEXISTING=from_file\n", encoding="utf-8")
    monkeypatch.setenv("EXISTING", "from_shell")
    monkeypatch.delenv("NEW_KEY", raising=False)

    applied = load_dotenv(env_file)

    assert applied == ["NEW_KEY"]
    assert os.environ["NEW_KEY"] == "from_file"
    assert os.environ["EXISTING"] == "from_shell"


def test_load_dotenv_returns_empty_when_file_missing(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "absent.env") == []
