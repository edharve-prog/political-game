"""Offline plug-ins that reuse what Claude wrote in earlier games.

When the offline scenario library serves a scenario Claude generated before (SD-3 loads
them from the store), these reuse Claude's work on it:

* ``StoredInterpreter``: the stored interpretation of the same or a closely matching
  response, kept to the nodes today's world has; otherwise the fallback (keyword
  matching). Like every interpreter it reports what was asked for: the game decides what
  is feasible, so a stored request that is now blocked shows up as blocked rather than
  being swapped for a different policy.
* ``StoredForecaster``: the stored candidate outcomes for the same kind of response
  (same action kinds, targets, directions and size), re-scored against today's engine
  forecast, delivery and resistance, with every graph change re-validated. Each candidate
  keeps the indicator changes it claimed, so its narrative and numbers still agree. A stored
  set none of whose candidates fits today's engine (best consistency below
  ``REUSE_MIN_FIT``) was written for a different world and is not reused; an older set that
  fits is tried, otherwise the fallback.

Both satisfy the game loop's protocols and make no LLM calls.
"""

from __future__ import annotations

from hog_sim.core.models import Delivery, Outcome, PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.forecasting.candidates import CLAIM_TURN
from hog_sim.forecasting.scoring import (
    CandidateScores,
    ScoreWeights,
    base_rate,
    combine,
    consistency,
)
from hog_sim.game.interfaces import Forecaster, Interpreter
from hog_sim.knowledge.entries import action_signature, normalise_text, scenario_id
from hog_sim.knowledge.store import KnowledgeStore
from hog_sim.policy.feasibility import Role, check_feasibility
from hog_sim.world.changes import check_graph_change
from hog_sim.world.propagation import DeltaDistribution

MATCH_THRESHOLD = 0.6
# A stored candidate set is reused only if at least one candidate's claims fit today's engine
# about this well: a geometric-mean |z| of about 1.8 spreads (see forecasting/scoring.py).
REUSE_MIN_FIT = 0.2


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
        self._delivery: Delivery | None = None

    def last_delivery(self) -> Delivery | None:
        """Delivery is read from the player's own words each turn, never from storage."""
        return self._delivery

    def interpret(self, text: str, state: WorldState, scenario: Scenario) -> list[PolicyAction]:
        reader = getattr(self.fallback, "delivery_for", None)
        self._delivery = reader(text, state) if reader else None
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
            if actions:
                self.last_source = "stored"
                return [a.model_copy(deep=True) for a in actions]
        self.last_source = "fallback"
        return self.fallback.interpret(text, state, scenario)


class StoredForecaster:
    def __init__(
        self, store: KnowledgeStore, fallback: Forecaster, role: Role = "prime_minister"
    ) -> None:
        self.store = store
        self.fallback = fallback
        self.role = role
        self.last_source: str | None = None

    def forecast(
        self,
        state: WorldState,
        scenario: Scenario,
        actions: list[PolicyAction],
        engine: DeltaDistribution,
        delivery: Delivery | None = None,
        limits: list[str] | None = None,
        sacked: list[str] | None = None,
    ) -> list[Outcome]:
        signature = action_signature(actions)
        matches = [
            e
            for e in self.store.outcomes(scenario_id(scenario))
            if action_signature(e.actions) == signature
        ]
        known = set(state.node_ids())
        turn = min(CLAIM_TURN, engine.horizon - 1)
        resistance = check_feasibility(list(actions), state, self.role).resistance
        for entry in reversed(matches):  # newest first
            candidates = [_adapt(c, state, known) for c in entry.candidates]
            scores = [
                CandidateScores(
                    consistency=consistency(c.indicator_deltas, engine, turn),
                    judge=c.scores.get("judge", c.probability),
                    base_rate=base_rate(c.events, delivery, resistance),
                    self_reported=c.scores.get("self_reported", c.probability),
                )
                for c in candidates
            ]
            if max(s.consistency for s in scores) < REUSE_MIN_FIT:
                continue  # written for a world whose engine moved differently
            self.last_source = "stored"
            rescored = combine(scores, ScoreWeights())
            return [
                c.model_copy(
                    update={
                        "probability": s.probability,
                        "scores": {**s.model_dump(exclude={"probability"}), "reused": 1.0},
                    }
                )
                for c, s in zip(candidates, rescored, strict=True)
            ]
        self.last_source = "fallback"
        return self.fallback.forecast(
            state, scenario, actions, engine, delivery=delivery, limits=limits, sacked=sacked
        )


def _adapt(outcome: Outcome, state: WorldState, known: set[str]) -> Outcome:
    """A stored outcome made safe for today's world: known nodes and valid changes only.

    Its claimed indicator changes are kept (not replaced by today's engine numbers), so they
    still match its narrative; the caller re-scores how well they fit today's engine."""
    events = []
    for event in outcome.approval_events:
        effects = {g: v for g, v in event.group_effects.items() if g in state.groups}
        if effects:
            events.append(event.model_copy(update={"group_effects": effects, "age_turns": 0}))
    return outcome.model_copy(
        deep=True,
        update={
            "indicator_deltas": {n: v for n, v in outcome.indicator_deltas.items() if n in known},
            "approval_events": events,
            "shocks": [s for s in outcome.shocks if s.node in known],
            "graph_changes": [g for g in outcome.graph_changes if not check_graph_change(state, g)],
        },
    )
