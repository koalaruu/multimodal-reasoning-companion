"""Load a local .env file into os.environ without third-party dependencies.

Existing environment variables win over file values, matching the behavior
users expect when they export a variable in the shell to override a file.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def load_dotenv(path: Path | None = None) -> list[str]:
    """Load variables from a .env file. Returns the names that were set.

    Values already present in os.environ are never overwritten.
    """

    env_path = _REPO_ROOT / ".env" if path is None else path
    if not env_path.is_file():
        return []

    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return []

    applied: list[str] = []
    for key, value in _parse_dotenv(text).items():
        if key in os.environ:
            continue
        os.environ[key] = value
        applied.append(key)
    return applied
