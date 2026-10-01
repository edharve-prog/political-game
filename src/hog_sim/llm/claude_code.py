"""Run LLM requests through the local Claude Code CLI instead of the API.

``claude -p`` is Claude Code's own headless mode. It uses whatever account Claude Code is
signed in with, so a player with a Claude subscription can play without an API key; calls
count against that plan's usage limits rather than being billed per token. It is meant for
playing on your own machine: anyone else running the game needs their own sign-in.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from typing import Any

from hog_sim.llm.client import (
    CallRecord,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMUnavailable,
    Usage,
    UsageLog,
)

SIGN_IN_HELP = "Run `claude` once and sign in with /login, then try again."

# Claude Code prefers an API key over its own sign-in when one is in the environment. This
# route exists to use the player's subscription, so keep API credentials away from it.
HIDDEN_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def subscription_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in HIDDEN_ENV}


class ClaudeCodeClient:
    subscription = True  # calls count against the Claude plan, not per-token billing

    def __init__(
        self,
        executable: str | None = None,
        timeout_s: float = 600,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        resolved = executable or shutil.which("claude")
        if resolved is None:
            raise LLMError(
                "Claude Code is not installed or not on PATH. Install it from "
                f"https://claude.com/claude-code. {SIGN_IN_HELP}"
            )
        self.executable = resolved
        self.timeout_s = timeout_s
        self.runner = runner
        self.usage = UsageLog()

    @property
    def _via_batch_file(self) -> bool:
        # npm installs a .cmd shim on Windows; cmd.exe mangles quotes and newlines in
        # arguments, so free text goes on stdin instead of the command line.
        return self.executable.lower().endswith((".cmd", ".bat"))

    def command(self, request: LLMRequest) -> tuple[list[str], str]:
        """The argv and stdin for a request."""
        conversation = _flatten(request)
        argv = [
            self.executable,
            "-p",
            "--output-format",
            "json",
            "--model",
            request.model,
            "--effort",
            request.effort,
            "--tools",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
        ]
        if self._via_batch_file:
            stdin = (
                f"{request.system}\n\nReply with only a JSON object matching this schema:\n"
                f"{json.dumps(request.output_schema)}\n\n{conversation}"
            )
        else:
            argv += [
                "--system-prompt",
                request.system,
                "--json-schema",
                json.dumps(request.output_schema),
            ]
            stdin = conversation
        return argv, stdin

    def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        argv, stdin = self.command(request)
        try:
            proc = self.runner(
                argv,
                input=stdin,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout_s,
                env=subscription_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMUnavailable(
                f"Claude Code: no answer within {self.timeout_s:.0f}s; try again"
            ) from exc
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            detail = (proc.stderr or proc.stdout or "").strip()[:500]
            raise LLMUnavailable(
                f"Claude Code failed: {detail or 'no output'}. {SIGN_IN_HELP}"
            ) from None
        if out.get("is_error"):
            raise LLMUnavailable(f"Claude Code: {out.get('result')}. {SIGN_IN_HELP}")

        structured = out.get("structured_output")
        text = json.dumps(structured) if structured is not None else _strip_fence(out["result"])
        usage = out.get("usage") or {}
        response = LLMResponse(
            text=text,
            usage=Usage(
                input_tokens=usage.get("input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
            served_model=_main_model(out.get("modelUsage") or {}, request.model),
        )
        self.usage.add(
            CallRecord(
                # Prefixed so cost estimates read zero: usage comes out of the Claude plan.
                model=f"claude-code/{request.model}",
                schema_name=request.schema_name,
                prompt_version=request.prompt_version,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_s=time.perf_counter() - start,
                served_model=response.served_model,
            )
        )
        return response


def _flatten(request: LLMRequest) -> str:
    """Headless mode takes one prompt, so earlier attempts and feedback are inlined."""
    if len(request.messages) == 1:
        return request.messages[0].content
    parts = []
    for m in request.messages:
        label = "Your earlier reply" if m.role == "assistant" else "User"
        parts.append(f"[{label}]\n{m.content}")
    return "\n\n".join(parts)


def _main_model(model_usage: dict[str, Any], fallback: str) -> str:
    """The model that produced the most output (Claude Code also makes small helper calls)."""
    if not model_usage:
        return fallback
    return max(model_usage, key=lambda m: model_usage[m].get("outputTokens", 0))


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()
