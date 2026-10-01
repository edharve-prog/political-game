"""LLM client interface, test clients, and the validate-and-retry loop.

Every LLM call in the game goes through ``structured_call``: the client returns JSON text,
the text is parsed into a Pydantic model, an optional semantic check runs (e.g. "every node
id exists in the world"), and anything that fails is sent back to the model with the error
for another attempt. Nothing unvalidated ever leaves this module.

Clients:
- ``AnthropicClient`` calls the Claude API. Needs the ``llm`` extra and ``ANTHROPIC_API_KEY``.
- ``FakeClient`` answers from a function or a queue of canned replies. For tests.
- ``RecordingClient`` caches responses on disk keyed by the request. Wrapping a live client
  records a cassette; with no inner client it replays one, so tests never need a key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, Field, ValidationError

from hog_sim.core.models import Model

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# USD per million tokens (input, output). Used for cost logging only.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class ModelConfig(Model):
    """Which model and effort each kind of call uses.

    Narrative work (scenarios) gets more effort than mapping text to actions. Swap in a
    cheaper model for either here once there is real cost data.
    """

    scenario_model: str = "claude-opus-5-5"
    scenario_effort: Effort = "medium"
    interpret_model: str = "claude-opus-5-5"
    interpret_effort: Effort = "low"
    max_tokens: int = 16000
    max_attempts: int = Field(3, ge=1)


# --- Requests and responses ------------------------------------------------


class Message(Model):
    role: Literal["user", "assistant"]
    content: str


class LLMRequest(Model):
    model: str
    system: str
    messages: list[Message]
    output_schema: dict[str, Any] = Field(description="JSON schema of the expected output")
    schema_name: str
    schema_fingerprint: str = Field("", description="Hash of the Pydantic schema")
    prompt_version: str
    effort: Effort = "medium"
    max_tokens: int = 16000

    def cache_key(self) -> str:
        # The wire schema depends on whether the SDK is installed, so key on the fingerprint
        # instead; recordings made with the SDK still replay in CI without it.
        payload = self.model_dump(mode="json", exclude={"output_schema"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


class Usage(Model):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(Model):
    text: str
    usage: Usage = Field(default_factory=Usage)
    served_model: str | None = None  # the model that answered, if the client reports it


class CallRecord(Model):
    model: str
    schema_name: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool = False
    served_model: str | None = None

    @property
    def cost_usd(self) -> float:
        if self.cached:
            return 0.0
        price_in, price_out = PRICES.get(self.model, (0.0, 0.0))
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000


class UsageLog:
    """Cost and latency of every call a client made."""

    def __init__(self) -> None:
        self.records: list[CallRecord] = []

    def add(self, record: CallRecord) -> None:
        self.records.append(record)
        log.info(
            "llm call %s/%s v%s: %d in, %d out, %.2fs, $%.4f%s",
            record.model,
            record.schema_name,
            record.prompt_version,
            record.input_tokens,
            record.output_tokens,
            record.latency_s,
            record.cost_usd,
            " (cached)" if record.cached else "",
        )

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)


class LLMClient(Protocol):
    usage: UsageLog

    def complete(self, request: LLMRequest) -> LLMResponse: ...


# --- Errors ----------------------------------------------------------------


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


class LLMOutputError(LLMError):
    """The model kept returning output that failed validation."""

    def __init__(self, schema_name: str, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(
            f"{schema_name}: no valid output after {len(errors)} attempts; last error: {errors[-1]}"
        )


class CassetteMiss(LLMError):
    pass


class LLMUnavailable(LLMError):
    """Claude could not be reached: no credit, bad credentials, rate limits, outages."""


OTHER_ROUTE_HINT = "To use your Claude subscription instead, run with --provider claude-code."


def describe_api_error(exc: Exception) -> str:
    """One line saying what went wrong with an Anthropic API call and what to do about it."""
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    message = ""
    if isinstance(body, dict):
        message = (body.get("error") or {}).get("message", "")
    message = message or str(exc)
    if "credit balance" in message.lower():
        what = "your Anthropic API account has no credit left (Console > Plans & Billing)"
    elif status == 401:
        what = (
            "the API key or sign-in was rejected; check ANTHROPIC_API_KEY or run `ant auth login`"
        )
    elif status == 403:
        what = f"the API account is not allowed to make this call ({message})"
    elif status == 429:
        what = "the API rate limit was hit; wait a minute and try again"
    elif status is not None and status >= 500:
        what = f"the Anthropic API had a server error ({status}); try again shortly"
    elif status is None:
        what = f"could not reach the Anthropic API ({message})"
    else:
        what = f"the Anthropic API rejected the request ({status}: {message})"
    return f"Anthropic API: {what}. {OTHER_ROUTE_HINT}"


# --- Clients ---------------------------------------------------------------


def _record(client: LLMClient, request: LLMRequest, response: LLMResponse, start: float) -> None:
    client.usage.add(
        CallRecord(
            model=request.model,
            schema_name=request.schema_name,
            prompt_version=request.prompt_version,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=time.perf_counter() - start,
            served_model=response.served_model,
        )
    )


class FakeClient:
    """Answers from ``responder(request)`` or, failing that, a queue of replies.

    Replies may be strings (returned as-is) or Pydantic models / dicts (serialised to JSON).
    Every request is kept in ``requests`` so tests can inspect prompts.
    """

    def __init__(
        self,
        replies: Iterable[str | BaseModel | dict] = (),
        responder: Callable[[LLMRequest], str | BaseModel | dict] | None = None,
    ) -> None:
        self.replies = list(replies)
        self.responder = responder
        self.requests: list[LLMRequest] = []
        self.usage = UsageLog()

    def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        self.requests.append(request)
        if self.responder is not None:
            reply = self.responder(request)
        elif self.replies:
            reply = self.replies.pop(0)
        else:
            raise LLMError("FakeClient has no replies left")
        response = LLMResponse(text=_to_text(reply))
        _record(self, request, response, start)
        return response


def _to_text(reply: str | BaseModel | dict) -> str:
    if isinstance(reply, BaseModel):
        return reply.model_dump_json()
    if isinstance(reply, dict):
        return json.dumps(reply)
    return reply


class RecordingClient:
    """Response cache on disk, one JSON file mapping request keys to responses.

    With ``inner`` set, misses go to the inner client and are saved (record mode). Without it,
    misses raise ``CassetteMiss`` (replay mode). Changing a prompt changes the key, so stale
    recordings are never served for a new prompt version.
    """

    def __init__(self, path: str | Path, inner: LLMClient | None = None) -> None:
        self.path = Path(path)
        self.inner = inner
        self.usage = UsageLog()
        self._store: dict[str, dict[str, Any]] = (
            json.loads(self.path.read_text()) if self.path.exists() else {}
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = request.cache_key()
        if key in self._store:
            response = LLMResponse.model_validate(self._store[key]["response"])
            self.usage.add(
                CallRecord(
                    model=request.model,
                    schema_name=request.schema_name,
                    prompt_version=request.prompt_version,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_s=0.0,
                    cached=True,
                )
            )
            return response
        if self.inner is None:
            raise CassetteMiss(f"no recording for {request.schema_name} request {key}")
        response = self.inner.complete(request)
        self.usage.records.extend(self.inner.usage.records[-1:])
        self._store[key] = {
            "schema_name": request.schema_name,
            "prompt_version": request.prompt_version,
            "last_message": request.messages[-1].content[-500:],
            "response": response.model_dump(mode="json"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._store, indent=2, sort_keys=True))
        return response


class AnthropicClient:
    """Claude API client using structured outputs, so replies are JSON matching the schema.

    Refusals are retried server-side on another model (``fallbacks="default"``); a refusal
    from the whole chain raises ``LLMRefusal``. Thinking is adaptive (always on for these
    models) and depth is set per request with ``effort``.
    """

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, client: Any = None) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on install
                raise LLMError("install the llm extra: uv sync --extra llm") from exc
            client = anthropic.Anthropic()
        self._client = client
        self.usage = UsageLog()

    def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        try:
            response = self._client.beta.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=[m.model_dump() for m in request.messages],
                output_config={
                    "effort": request.effort,
                    "format": {"type": "json_schema", "schema": request.output_schema},
                },
                betas=[self.FALLBACK_BETA],
                fallbacks="default",
            )
        except Exception as exc:
            if not _is_api_error(exc):
                raise
            raise LLMUnavailable(describe_api_error(exc)) from exc
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(f"{request.schema_name} refused: {details}")
        text = "".join(b.text for b in response.content if b.type == "text")
        result = LLMResponse(
            text=text,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            served_model=getattr(response, "model", None),
        )
        _record(self, request, result, start)
        if response.stop_reason == "max_tokens":
            log.warning("%s hit max_tokens; output is likely truncated", request.schema_name)
        return result


def _is_api_error(exc: Exception) -> bool:
    try:
        import anthropic
    except ImportError:  # pragma: no cover - only a stubbed client gets here without the SDK
        return False
    return isinstance(exc, anthropic.APIError)


# --- Structured calls ------------------------------------------------------


def output_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for structured outputs.

    Uses the SDK's transform when installed (it moves numeric bounds into descriptions, which
    the API accepts); falls back to Pydantic's own schema so tests run without the SDK.
    Bounds are enforced by Pydantic validation either way.
    """
    try:
        from anthropic import transform_schema
    except ImportError:
        return model.model_json_schema()
    return transform_schema(model)


def structured_call(
    client: LLMClient,
    *,
    output_type: type[T],
    system: str,
    prompt: str,
    model: str,
    prompt_version: str,
    effort: Effort = "medium",
    max_tokens: int = 16000,
    max_attempts: int = 3,
    check: Callable[[T], list[str]] | None = None,
) -> T:
    """Call the model until it returns output that parses into ``output_type`` and passes
    ``check`` (which returns a list of problems, empty when fine).

    Failed attempts are shown back to the model with the problems, so it can correct itself.
    Raises ``LLMOutputError`` after ``max_attempts``.
    """
    messages = [Message(role="user", content=prompt)]
    schema = output_schema(output_type)
    fingerprint = hashlib.sha256(
        json.dumps(output_type.model_json_schema(), sort_keys=True).encode()
    ).hexdigest()[:12]
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        request = LLMRequest(
            model=model,
            system=system,
            messages=messages,
            output_schema=schema,
            schema_name=output_type.__name__,
            schema_fingerprint=fingerprint,
            prompt_version=prompt_version,
            effort=effort,
            max_tokens=max_tokens,
        )
        text = client.complete(request).text
        try:
            result = output_type.model_validate_json(text)
            problems = check(result) if check else []
        except ValidationError as exc:
            problems = [_short_validation_error(exc)]
        if not problems:
            return result
        error = "; ".join(problems)
        errors.append(error)
        log.warning("%s attempt %d rejected: %s", output_type.__name__, attempt, error)
        messages = [
            *messages,
            Message(role="assistant", content=text),
            Message(
                role="user",
                content=(
                    "That output was rejected:\n"
                    + "\n".join(f"- {p}" for p in problems)
                    + "\nReturn a corrected JSON object."
                ),
            ),
        ]
    raise LLMOutputError(output_type.__name__, errors)


def _short_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(x) for x in err["loc"]) or "(root)"
        parts.append(f"{loc}: {err['msg']}")
    return "invalid output: " + "; ".join(parts)
