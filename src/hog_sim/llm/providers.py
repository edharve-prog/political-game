"""Choose how the game reaches Claude.

- ``api``: the Anthropic API through the SDK. Signs in with ``ANTHROPIC_API_KEY`` or, with no
  key, an ``ant auth login`` profile (browser OAuth, no key to copy). Billed per token to the
  Anthropic Console account.
- ``claude-code``: the local Claude Code CLI in headless mode, signed in with the player's
  Claude account. Counts against that plan's limits. For playing on your own machine.
- ``auto``: ``api`` when API credentials exist, otherwise ``claude-code`` when it is
  installed.

Every provider returns an ``LLMClient``, so the rest of the game does not care which.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Literal

from hog_sim.llm.client import LLMClient, LLMError

Provider = Literal["auto", "api", "claude-code"]
PROVIDERS: tuple[str, ...] = ("auto", "api", "claude-code")

NO_BACKEND_HELP = """\
--llm needs a way to reach Claude. Pick one:
  1. Your Claude subscription: install Claude Code (https://claude.com/claude-code),
     run `claude` once and sign in. Then: hog-sim --llm --provider claude-code
  2. An API key from https://console.anthropic.com: put this line in a file called .env
     next to pyproject.toml:
         ANTHROPIC_API_KEY=sk-ant-...
  3. API sign-in without a key: install the `ant` CLI and run `ant auth login`."""


def anthropic_config_dir() -> Path:
    if os.environ.get("ANTHROPIC_CONFIG_DIR"):
        return Path(os.environ["ANTHROPIC_CONFIG_DIR"])
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "Anthropic"
    return Path.home() / ".config" / "anthropic"


def has_api_credentials() -> bool:
    """A key, a token, or an `ant auth login` profile the SDK will pick up."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    if os.environ.get("ANTHROPIC_PROFILE"):
        return True
    creds = anthropic_config_dir() / "credentials"
    return creds.is_dir() and any(creds.glob("*.json"))


def resolve_provider(provider: str) -> str:
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider {provider!r}; use one of {', '.join(PROVIDERS)}")
    if provider != "auto":
        return provider
    if has_api_credentials():
        return "api"
    if shutil.which("claude"):
        return "claude-code"
    raise LLMError(NO_BACKEND_HELP)


def describe(provider: str) -> str:
    if provider == "claude-code":
        return "your Claude Code sign-in (uses your Claude plan's limits)"
    return "the Anthropic API (billed per token to your Console account)"


def make_client(provider: str = "auto") -> tuple[LLMClient, str]:
    """A client for ``provider`` and the resolved provider name."""
    resolved = resolve_provider(provider)
    if resolved == "claude-code":
        from hog_sim.llm.claude_code import ClaudeCodeClient

        return ClaudeCodeClient(), resolved
    try:
        import anthropic  # noqa: F401
    except ImportError:
        raise LLMError(
            "The API provider needs the Claude SDK. Install it with: uv sync --extra llm"
        ) from None
    if not has_api_credentials():
        raise LLMError(NO_BACKEND_HELP)
    from hog_sim.llm.client import AnthropicClient

    return AnthropicClient(), resolved
