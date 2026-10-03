"""Run LLM requests through the local OpenAI Codex CLI, signed in with ChatGPT.

``codex exec`` is Codex's own non-interactive mode. It uses whatever account Codex is signed
in with (``codex login`` opens the ChatGPT sign-in in a browser), so a player with a ChatGPT
plan that includes Codex can play without an OpenAI API key; calls count against that plan's
Codex limits. Like the Claude Code route, it is meant for playing on your own machine.

Codex is a coding agent rather than a plain model endpoint, so each call runs in an empty
scratch folder with a read-only sandbox and is told to answer with JSON only. The model is
Codex's own default unless ``HOG_SIM_CODEX_MODEL`` names one; the game's Claude model names
mean nothing to it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from typing import Any

from hog_sim.llm.claude_code import _flatten, _strip_fence
from hog_sim.llm.client import (
    CallRecord,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMUnavailable,
    Usage,
    UsageLog,
)

SIGN_IN_HELP = "Run `codex login` and sign in with ChatGPT, then try again."
INSTALL_HELP = "Install it with: npm install -g @openai/codex"
MODEL_ENV = "HOG_SIM_CODEX_MODEL"

# Codex prefers an API key over the ChatGPT sign-in when one is in the environment. This
# route exists to use the player's ChatGPT plan, so keep API keys away from it.
HIDDEN_ENV = ("OPENAI_API_KEY", "CODEX_API_KEY")

# Codex takes minimal/low/medium/high/xhigh; the game also has "max".
EFFORTS = {"low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "xhigh"}

ANSWER_RULES = (
    "Answer directly from your own knowledge. Do not run commands, read files or browse. "
    "Reply with only a JSON object matching this schema, with no other text:"
)


def chatgpt_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in HIDDEN_ENV}


def codex_signed_in(executable: str, runner: Callable[..., Any] = subprocess.run) -> bool | None:
    """Whether ``codex login status`` reports a sign-in; None when it can't say."""
    try:
        proc = runner(
            [executable, "login", "status"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            env=chatgpt_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.returncode == 0


class CodexClient:
    subscription = True  # calls count against the ChatGPT plan, not per-token billing
    name = "Codex"

    def __init__(
        self,
        executable: str | None = None,
        model: str | None = None,
        timeout_s: float = 600,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        resolved = executable or shutil.which("codex")
        if resolved is None:
            raise LLMError(f"Codex is not installed or not on PATH. {INSTALL_HELP}. {SIGN_IN_HELP}")
        self.executable = resolved
        self.model = model if model is not None else os.environ.get(MODEL_ENV) or None
        self.timeout_s = timeout_s
        self.runner = runner
        self.usage = UsageLog()

    def command(self, request: LLMRequest, workdir: str) -> tuple[list[str], str]:
        """The argv and stdin for a request. Free text goes on stdin (Windows .cmd shims)."""
        argv = [
            self.executable,
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--cd",
            workdir,
            "-c",
            f'model_reasoning_effort="{EFFORTS.get(request.effort, "medium")}"',
        ]
        if self.model:
            argv += ["--model", self.model]
        argv.append("-")
        stdin = (
            f"{request.system}\n\n{ANSWER_RULES}\n{json.dumps(request.output_schema)}\n\n"
            f"{_flatten(request)}"
        )
        return argv, stdin

    def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="hog-sim-codex-") as workdir:
            argv, stdin = self.command(request, workdir)
            try:
                proc = self.runner(
                    argv,
                    input=stdin,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=self.timeout_s,
                    env=chatgpt_env(),
                )
            except subprocess.TimeoutExpired as exc:
                raise LLMUnavailable(
                    f"Codex: no answer within {self.timeout_s:.0f}s; try again"
                ) from exc
        text, usage, failure = _read_events(proc.stdout or "")
        if text is None:
            detail = failure or (proc.stderr or "").strip()[-500:] or "no output"
            raise LLMUnavailable(f"Codex failed: {detail}. {SIGN_IN_HELP}")
        served = self.model or "codex-default"
        response = LLMResponse(
            text=_strip_fence(text),
            usage=Usage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
            served_model=f"codex/{served}",
        )
        self.usage.add(
            CallRecord(
                # Prefixed so cost estimates read zero: usage comes out of the ChatGPT plan.
                model=f"codex/{served}",
                schema_name=request.schema_name,
                prompt_version=request.prompt_version,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_s=time.perf_counter() - start,
                served_model=response.served_model,
            )
        )
        return response


def _read_events(stdout: str) -> tuple[str | None, dict[str, Any], str | None]:
    """The last agent message, the turn's token usage, and why the turn failed, if it did.

    ``codex exec --json`` prints one event per line. ``error`` events also cover retries
    that later succeed, so they only explain a turn that produced no answer.
    """
    text: str | None = None
    usage: dict[str, Any] = {}
    failure: str | None = None
    last_error: str | None = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        item = event.get("item") or {}
        if kind == "item.completed" and item.get("type") == "agent_message":
            text = item.get("text", "")
        elif kind == "turn.completed":
            usage = event.get("usage") or {}
        elif kind == "turn.failed":
            failure = (event.get("error") or {}).get("message") or "the turn failed"
        elif kind == "error":
            last_error = event.get("message") or last_error
    return text, usage, failure or last_error
