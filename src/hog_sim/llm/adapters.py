"""Adapters that plug the LLM layer into the game loop's protocols.

``LLMScenarioSource`` satisfies ``ScenarioSource`` and ``LLMInterpreter`` satisfies
``Interpreter`` (both in ``game/interfaces.py`` on the Project 8 branch). They only rely
on the shape of the turn history (``.scenario.title``, ``.outcome.narrative``), so this
module does not import the game package.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from hog_sim.core.models import Delivery, Pledge, PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.llm.client import LLMClient, ModelConfig
from hog_sim.llm.interpreter import Interpretation, interpret
from hog_sim.llm.scenario_gen import GeneratedScenario, generate_scenario
from hog_sim.llm.summary import summarise_state
from hog_sim.policy.feasibility import Role
from hog_sim.world.storylines import (
    final_storylines,
    must_open_new,
    open_storylines,
    storylines_text,
)


def recent_events(history: Sequence[Any], limit: int = 8) -> list[str]:
    """One line per past turn: the scenario title, the leader's response and what came of it."""
    events = []
    for record in list(history)[-limit:]:
        line = f"Turn {record.turn}: {record.scenario.title}"
        response = getattr(record, "response", None)
        if response:
            text = " ".join(response.split())
            line += f"; the leader responded: {text[:200]}"
        outcome = getattr(record, "outcome", None)
        if outcome is not None:
            line += f" -> {outcome.narrative}"
        events.append(line)
    return events


class LLMScenarioSource:
    """``recall(state)`` (optional) returns precedent lines from the knowledge store."""

    def __init__(
        self,
        client: LLMClient,
        role: Role = "prime_minister",
        config: ModelConfig | None = None,
        recall: Callable[[WorldState], list[str]] | None = None,
    ) -> None:
        self.client = client
        self.role = role
        self.config = config
        self.recall = recall

    def next_scenario(self, state: WorldState, history: Sequence[Any]) -> GeneratedScenario:
        precedents = self.recall(state) if self.recall else None
        summary = summarise_state(state, self.role, recent_events(history), precedents=precedents)
        recent = [r.scenario for r in history]
        return generate_scenario(
            summary,
            self.client,
            self.config,
            recent=recent,
            storylines=open_storylines(state),
            storylines_prompt=storylines_text(state),
            final=final_storylines(state),
            open_new=must_open_new(state, history),
        )


class LLMInterpreter:
    """Interprets the player's text into the actions they asked for.

    It does not judge them: the game applies feasibility, diminishing returns and political
    capital in one place (``policy.limits.constrain``), the same in every mode, so blocked
    actions stay in the turn record with a note saying why. The full interpretation of the
    last call is kept on ``last_interpretation`` so an interface can show the clarifying
    question or the measures that could not be mapped.
    """

    def __init__(
        self, client: LLMClient, role: Role = "prime_minister", config: ModelConfig | None = None
    ) -> None:
        self.client = client
        self.role = role
        self.config = config
        self.last_interpretation: Interpretation | None = None

    def interpret(self, text: str, state: WorldState, scenario: Scenario) -> list[PolicyAction]:
        summary = summarise_state(state, self.role)
        result = interpret(text, summary, self.client, scenario=scenario, config=self.config)
        self.last_interpretation = result
        return list(result.actions)

    def last_delivery(self) -> Delivery | None:
        """How the last response was delivered (story RB-4), or None before any call."""
        return self.last_interpretation.delivery if self.last_interpretation else None

    def last_pledges(self) -> list[Pledge]:
        """Promises the last response made (story SD-5)."""
        if self.last_interpretation is None:
            return []
        return [p.to_pledge() for p in self.last_interpretation.pledges]

    def last_sacked(self) -> list[str]:
        """Ministers the last response sacked (story SD-4)."""
        return list(self.last_interpretation.sacked) if self.last_interpretation else []
