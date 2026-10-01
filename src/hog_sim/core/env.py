"""Read settings such as API keys from a local ``.env`` file.

Setting environment variables is awkward on Windows, so the game also looks for a ``.env``
file in the current directory (``KEY=value`` per line, ``#`` comments). Variables already
set in the environment win over the file. ``.env`` is git-ignored.
"""

from __future__ import annotations

import os
from pathlib import Path


def parse_env(text: str) -> dict[str, str]:
    values = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            values[key] = value
    return values


def load_env(path: str | Path = ".env") -> list[str]:
    """Set variables from ``path`` that are not already set. Returns the names it set."""
    path = Path(path)
    if not path.is_file():
        return []
    loaded = []
    for key, value in parse_env(path.read_text(encoding="utf-8-sig")).items():
        # An empty value (as in .env.example) would still count as "set", and an empty
        # API key outranks other sign-in methods, so skip blanks.
        if value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
