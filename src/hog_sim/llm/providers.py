"""Choose how the game reaches a model: Claude by default, or OpenAI Codex.

- ``api``: the Anthropic API through the SDK. Signs in with ``ANTHROPIC_API_KEY`` or, with no
  key, an ``ant auth login`` profile (browser OAuth, no key to copy). Billed per token to the
  Anthropic Console account.
- ``claude-code`` (the default): the local Claude Code CLI in headless mode, signed in with
  the player's Claude account. Counts against that plan's limits. For playing on your own
  machine. When Claude Code is missing or signed out and API credentials exist, the game
  falls back to ``api`` and says so.
- ``codex``: the local OpenAI Codex CLI in non-interactive mode (``codex exec``), signed in
  with the player's ChatGPT account via ``codex login``. Counts against that plan's Codex
  limits. For playing on your own machine. Only used when asked for; no fallback.
- ``auto``: kept for older ``.env`` files; the same as ``claude-code``.

Every provider returns an ``LLMClient``, so the rest of the game does not care which.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from hog_sim.llm.client import LLMClient, LLMError

Provider = Literal["auto", "api", "claude-code", "codex"]
PROVIDERS: tuple[str, ...] = ("claude-code", "api", "codex", "auto")
DEFAULT_PROVIDER = "claude-code"

NO_BACKEND_HELP = """\
The game needs a way to reach Claude. Pick one:
  1. Your Claude subscription: install Claude Code (https://claude.com/claude-code),
     run `claude` once and sign in. Then: hog-sim
  2. An API key from https://console.anthropic.com: put this line in a file called .env
     next to pyproject.toml:
         ANTHROPIC_API_KEY=sk-ant-...
  3. API sign-in without a key: install the `ant` CLI and run `ant auth login`.
  4. Your ChatGPT plan through Codex: npm install -g @openai/codex, run `codex login`
     and sign in with ChatGPT. Then: hog-sim --provider codex"""


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


def claude_code_signed_in(
    executable: str, runner: Callable[..., Any] = subprocess.run
) -> bool | None:
    """Whether Claude Code reports a sign-in; None when it can't say (older versions)."""
    from hog_sim.llm.claude_code import subscription_env

    try:
        proc = runner(
            [executable, "auth", "status"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            env=subscription_env(),
        )
        return bool(json.loads(proc.stdout)["loggedIn"])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return None


def choose_provider(
    provider: str, runner: Callable[..., Any] = subprocess.run
) -> tuple[str, str | None]:
    """The provider to use and, when it differs from the one asked for, a note saying why."""
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider {provider!r}; use one of {', '.join(PROVIDERS)}")
    if provider == "api":
        return "api", None
    if provider == "codex":
        return _choose_codex(runner)
    executable = shutil.which("claude")
    if executable is None:
        problem = "Claude Code isn't installed"
    elif claude_code_signed_in(executable, runner) is False:
        problem = "Claude Code isn't signed in"
    else:
        return "claude-code", None
    if has_api_credentials():
        return "api", f"{problem}, so the game is falling back to the Anthropic API."
    if executable is None:
        raise LLMError(NO_BACKEND_HELP)
    from hog_sim.llm.claude_code import SIGN_IN_HELP

    raise LLMError(f"{problem}. {SIGN_IN_HELP}")


def _choose_codex(runner: Callable[..., Any]) -> tuple[str, str | None]:
    """Codex was asked for by name, so a missing or signed-out Codex is an error, not a fallback."""
    from hog_sim.llm.codex import INSTALL_HELP, SIGN_IN_HELP, codex_signed_in

    executable = shutil.which("codex")
    if executable is None:
        raise LLMError(f"Codex isn't installed. {INSTALL_HELP}. {SIGN_IN_HELP}")
    if codex_signed_in(executable, runner) is False:
        raise LLMError(f"Codex isn't signed in. {SIGN_IN_HELP}")
    return "codex", None


def resolve_provider(provider: str) -> str:
    return choose_provider(provider)[0]


def describe(provider: str) -> str:
    if provider == "claude-code":
        return "your Claude Code sign-in (uses your Claude plan's limits)"
    if provider == "codex":
        return "your Codex ChatGPT sign-in (uses your ChatGPT plan's Codex limits)"
    return "the Anthropic API (billed per token to your Console account)"


def make_client(provider: str = DEFAULT_PROVIDER) -> tuple[LLMClient, str, str | None]:
    """A client for ``provider``, the provider actually used, and any fallback note."""
    resolved, note = choose_provider(provider)
    if resolved == "claude-code":
        from hog_sim.llm.claude_code import ClaudeCodeClient

        return ClaudeCodeClient(), resolved, note
    if resolved == "codex":
        from hog_sim.llm.codex import CodexClient

        return CodexClient(), resolved, note
    try:
        import anthropic  # noqa: F401
    except ImportError:
        raise LLMError(
            "The API provider needs the Claude SDK. Install it with: uv sync --extra llm"
        ) from None
    if not has_api_credentials():
        raise LLMError(NO_BACKEND_HELP)
    from hog_sim.llm.client import AnthropicClient

    return AnthropicClient(), resolved, note
