"""Offline plug-ins that reuse what Claude wrote in earlier games.

When the offline scenario library serves a scenario Claude generated before (SD-3 loads
them from the store), these reuse Claude's work on it:

* ``StoredInterpreter``: the stored interpretation of the same or a closely matching
  response, kept to the nodes today's world has; otherwise the fallback (keyword
  matching). A close match must also say the same thing: the same negations, directions,
  numbers and durations, so "do not increase public spending" never reuses "increase
  public spending". Like every interpreter it reports what was asked for: the game decides
  what is feasible, so a stored request that is now blocked shows up as blocked rather
  than being swapped for a different policy. Delivery, pledges and sackings are read from
  the player's own words each turn, never from storage.
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

import re

from hog_sim.core.models import Delivery, Outcome, Pledge, PolicyAction, Scenario
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
from hog_sim.world.changes import action_changes, apply_graph_changes, validate_graph_changes
from hog_sim.world.propagation import DeltaDistribution

MATCH_THRESHOLD = 0.6
# A stored candidate set is reused only if at least one candidate's claims fit today's engine
# about this well: a geometric-mean |z| of about 1.8 spreads (see forecasting/scoring.py).
REUSE_MIN_FIT = 0.2


def _overlap(a: str, b: str) -> float:
    x, y = set(normalise_text(a).split()), set(normalise_text(b).split())
    return len(x & y) / len(x | y) if x | y else 0.0


# Words that change what a response means however few of them there are. Two responses
# that share most of their words are only the same request when these agree.
_NEGATION = re.compile(r"\b(?:no|not|never|nor|none|neither|without|cannot)\b|n['’]t\b")
_DIRECTION = [
    (
        re.compile(
            r"\b(?:(?:rais|increas|boost|expand|extend|higher|hike|lift|doubl|tripl|introduc|"
            r"grow)\w*|more|up|add|adds|adding)\b"
        ),
        "+",
    ),
    (
        re.compile(
            r"\b(?:(?:cut|reduc|lower|slash|scrap|abolish|freez|halv|decreas|remov|withdraw|"
            r"shrink|trim)\w*|less|fewer|down|end|ends|ending|stop|stops|drop|drops)\b"
        ),
        "-",
    ),
]
_NUMBER = re.compile(
    r"\d+(?:\.\d+)?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"fifteen|twenty|thirty|forty|fifty|hundred|thousand|half|third|quarter)\b"
)
_UNIT = re.compile(
    r"\b(?:day|week|month|quarter|year|decade|parliament|term|temporar|permanent|indefinite|"
    r"percent|billion|bn|million|pence|pound)\w*|%"
)


def intent_key(text: str) -> tuple:
    """What two responses must share to mean the same thing: how many negations, the
    directions asked for (in order), and every number and unit of size or duration."""
    t = text.lower()
    hits = sorted((m.start(), sign) for pattern, sign in _DIRECTION for m in pattern.finditer(t))
    return (
        len(_NEGATION.findall(t)),
        tuple(sign for _, sign in hits),
        tuple(_NUMBER.findall(t)),
        tuple(sorted(_UNIT.findall(t))),
    )


def same_request(stored: str, text: str) -> float:
    """How closely ``text`` matches a stored response: 1.0 when identical, the word overlap
    when the two agree on ``intent_key``, and 0.0 when they don't."""
    if normalise_text(stored) == normalise_text(text):
        return 1.0
    if intent_key(stored) != intent_key(text):
        return 0.0
    return _overlap(stored, text)


class StoredInterpreter:
    def __init__(
        self, store: KnowledgeStore, fallback: Interpreter, role: Role = "prime_minister"
    ) -> None:
        self.store = store
        self.fallback = fallback
        self.role = role
        self.last_source: str | None = None  # "stored" or "fallback", for the interface
        self._delivery: Delivery | None = None
        self._pledges: list[Pledge] = []
        self._sacked: list[str] = []
        self._used_fallback = False

    # Delivery, pledges and sackings are read from the player's own words each turn, never
    # from storage: from the fallback's own reading when it interpreted the turn, and from
    # its readers when a stored interpretation did.

    def last_delivery(self) -> Delivery | None:
        if self._used_fallback:
            return getattr(self.fallback, "last_delivery", lambda: None)()
        return self._delivery

    def last_pledges(self) -> list[Pledge]:
        if self._used_fallback:
            return list(getattr(self.fallback, "last_pledges", lambda: [])())
        return list(self._pledges)

    def last_sacked(self) -> list[str]:
        if self._used_fallback:
            return list(getattr(self.fallback, "last_sacked", lambda: [])())
        return list(self._sacked)

    def _read_metadata(self, text: str, state: WorldState) -> None:
        def read(name: str, *args: object) -> object:
            reader = getattr(self.fallback, name, None)
            return reader(*args) if reader else None

        self._delivery = read("delivery_for", text, state)  # type: ignore[assignment]
        self._pledges = list(read("pledges_for", text) or [])  # type: ignore[call-overload]
        self._sacked = list(read("sacked_for", text, state) or [])  # type: ignore[call-overload]

    def interpret(self, text: str, state: WorldState, scenario: Scenario) -> list[PolicyAction]:
        self._used_fallback = False
        self._read_metadata(text, state)
        entries = self.store.interpretations(scenario_id(scenario))
        best, best_score = None, 0.0
        for entry in entries:
            score = same_request(entry.response, text)
            if score > best_score:
                best, best_score = entry, score
        if best is not None and best_score >= MATCH_THRESHOLD:
            known = set(state.node_ids())
            actions = [a for a in best.actions if a.target in known]
            if actions:
                self.last_source = "stored"
                return [a.model_copy(deep=True) for a in actions]
        self.last_source = "fallback"
        self._used_fallback = True
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
        # Graph changes are checked against the world as resolve() meets it, after the
        # player's own moves.
        staged = apply_graph_changes(state, action_changes(state, actions), trusted=True)
        turn = min(CLAIM_TURN, engine.horizon - 1)
        resistance = check_feasibility(list(actions), state, self.role).resistance
        for entry in reversed(matches):  # newest first
            candidates = [_adapt(c, staged, known) for c in entry.candidates]
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
            # All or nothing, as in resolve(): a stored set that no longer fits is dropped.
            "graph_changes": []
            if validate_graph_changes(state, outcome.graph_changes)
            else list(outcome.graph_changes),
        },
    )
