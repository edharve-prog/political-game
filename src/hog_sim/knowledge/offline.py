"""Offline plug-ins that reuse what Claude wrote in earlier games.

When the offline scenario library serves a scenario Claude generated before (SD-3 loads
them from the store), these reuse Claude's work on it:

* ``StoredInterpreter``: the stored interpretation of the same or a closely matching
  response, re-checked against today's world and feasibility; otherwise the fallback
  (keyword matching).
* ``StoredForecaster``: the stored candidate outcomes for the same kind of response
  (same action kinds, targets and directions), with the engine's current numbers in place
  of the stored ones and every graph change re-validated; otherwise the fallback.

Both satisfy the game loop's protocols and make no LLM calls.
"""

from __future__ import annotations

from hog_sim.core.models import Outcome, PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.game.interfaces import Forecaster, Interpreter
from hog_sim.knowledge.entries import action_signature, normalise_text, scenario_id
from hog_sim.knowledge.store import KnowledgeStore
from hog_sim.policy.feasibility import Role, check_feasibility
from hog_sim.world.changes import check_graph_change
from hog_sim.world.propagation import DeltaDistribution

MATCH_THRESHOLD = 0.6


def _overlap(a: str, b: str) -> float:
    x, y = set(normalise_text(a).split()), set(normalise_text(b).split())
    return len(x & y) / len(x | y) if x | y else 0.0


class StoredInterpreter:
    def __init__(
        self, store: KnowledgeStore, fallback: Interpreter, role: Role = "prime_minister"
    ) -> None:
        self.store = store
        self.fallback = fallback
        self.role = role
        self.last_source: str | None = None  # "stored" or "fallback", for the interface

    def interpret(self, text: str, state: WorldState, scenario: Scenario) -> list[PolicyAction]:
        entries = self.store.interpretations(scenario_id(scenario))
        best, best_score = None, 0.0
        for entry in entries:
            score = (
                1.0
                if normalise_text(entry.response) == normalise_text(text)
                else _overlap(entry.response, text)
            )
            if score > best_score:
                best, best_score = entry, score
        if best is not None and best_score >= MATCH_THRESHOLD:
            known = set(state.node_ids())
            actions = [a for a in best.actions if a.target in known]
            feasible = check_feasibility(actions, state, self.role).feasible_actions
            if feasible:
                self.last_source = "stored"
                return [a.model_copy(deep=True) for a in feasible]
        self.last_source = "fallback"
        return self.fallback.interpret(text, state, scenario)


class StoredForecaster:
    def __init__(self, store: KnowledgeStore, fallback: Forecaster) -> None:
        self.store = store
        self.fallback = fallback
        self.last_source: str | None = None

    def forecast(
        self,
        state: WorldState,
        scenario: Scenario,
        actions: list[PolicyAction],
        engine: DeltaDistribution,
    ) -> list[Outcome]:
        signature = action_signature(actions)
        matches = [
            e
            for e in self.store.outcomes(scenario_id(scenario))
            if action_signature(e.actions) == signature
        ]
        if not matches:
            self.last_source = "fallback"
            return self.fallback.forecast(state, scenario, actions, engine)
        self.last_source = "stored"
        entry = matches[-1]
        expected = {
            n: round(v, 3) for n, v in engine.at(0).items() if n.startswith("indicator:") and v
        }
        known = set(state.node_ids())
        candidates = [_adapt(c, state, known, expected) for c in entry.candidates]
        total = sum(c.probability for c in candidates) or 1.0
        for c in candidates:
            c.probability = c.probability / total if total else 1 / len(candidates)
        return candidates


def _adapt(outcome: Outcome, state: WorldState, known: set[str], expected: dict) -> Outcome:
    """A stored outcome made safe for today's world: engine numbers, known nodes only."""
    events = []
    for event in outcome.approval_events:
        effects = {g: v for g, v in event.group_effects.items() if g in state.groups}
        if effects:
            events.append(event.model_copy(update={"group_effects": effects, "age_turns": 0}))
    return outcome.model_copy(
        deep=True,
        update={
            "indicator_deltas": dict(expected),
            "approval_events": events,
            "shocks": [s for s in outcome.shocks if s.node in known],
            "graph_changes": [g for g in outcome.graph_changes if not check_graph_change(state, g)],
            "scores": {**outcome.scores, "reused": 1.0},
        },
    )
