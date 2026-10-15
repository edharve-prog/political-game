"""A per-turn cap on model calls (review of 13 Oct 2026, O10).

``BudgetedClient`` wraps a client and refuses a call once the turn being played has made
``limit`` calls, raising ``LLMBudgetExceeded`` before anything is sent. The game treats it like
any other model failure: the player can allow another ``limit`` calls for the turn or play
that step offline. Retries inside a call (a reply that failed validation) count too, since each
one costs as much as the first.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hog_sim.llm.client import LLMClient, LLMError, LLMRequest, LLMResponse

DEFAULT_LIMIT = 25
BUDGET_ENV = "HOG_SIM_CALL_BUDGET"


class LLMBudgetExceeded(LLMError):
    """This turn has made as many model calls as its budget allows."""

    def __init__(self, client: BudgetedClient) -> None:
        self.client = client
        super().__init__(
            f"this turn has used its {client.allowed} model calls; trying again allows "
            f"another {client.limit} (set the cap with --call-budget or {BUDGET_ENV}; 0 means "
            "no cap)"
        )


class BudgetedClient:
    """Counts calls per turn (``turn_of`` returns the turn being played) and refuses those over
    ``limit``. ``extend()`` allows another ``limit`` this turn. Other attributes are the wrapped
    client's."""

    def __init__(self, inner: LLMClient, limit: int = DEFAULT_LIMIT) -> None:
        self.inner = inner
        self.limit = limit
        self.turn_of: Callable[[], int | None] = lambda: None
        self._turn: int | None = None
        self.calls = 0
        self.allowed = limit

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def extend(self) -> None:
        self.allowed += self.limit

    def complete(self, request: LLMRequest) -> LLMResponse:
        turn = self.turn_of()
        if turn != self._turn:
            self._turn, self.calls, self.allowed = turn, 0, self.limit
        if self.limit and self.calls >= self.allowed:
            raise LLMBudgetExceeded(self)
        self.calls += 1
        return self.inner.complete(request)
