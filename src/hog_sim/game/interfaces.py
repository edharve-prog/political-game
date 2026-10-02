"""The pluggable parts of a turn.

The game loop only talks to these protocols, so the stubs in ``game/stubs.py`` can be
swapped for the LLM-backed versions from Project 5 (scenarios, interpretation) and
Project 6 (outcome forecasting) without changing the loop.
"""

from __future__ import annotations

from typing import Protocol

from hog_sim.core.models import Delivery, Outcome, PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.game.records import TurnRecord
from hog_sim.world.propagation import DeltaDistribution


class ScenarioSource(Protocol):
    def next_scenario(self, state: WorldState, history: list[TurnRecord]) -> Scenario: ...


class Interpreter(Protocol):
    """May also offer ``last_delivery() -> Delivery | None`` for how the last response was
    delivered (story RB-4); the game falls back to a plain ``Delivery()``."""

    def interpret(self, text: str, state: WorldState, scenario: Scenario) -> list[PolicyAction]: ...


class Forecaster(Protocol):
    def forecast(
        self,
        state: WorldState,
        scenario: Scenario,
        actions: list[PolicyAction],
        engine: DeltaDistribution,
        delivery: Delivery | None = None,
    ) -> list[Outcome]: ...


class NeedsClarification(Exception):  # noqa: N818 - reads as a message, not an error
    """Raised by an interpreter before any state changes when the response is too vague.

    The interface shows ``question`` and asks again; the turn is not played.
    """

    def __init__(self, question: str) -> None:
        super().__init__(question)
        self.question = question
