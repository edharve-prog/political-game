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
  (same action kinds, targets and directions), with the engine's current numbers in place
  of the stored ones and every graph change re-validated; otherwise the fallback.

Both satisfy the game loop's protocols and make no LLM calls.
"""

from __future__ import annotations

import re

from hog_sim.core.models import Delivery, Outcome, Pledge, PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.game.interfaces import Forecaster, Interpreter
from hog_sim.knowledge.entries import action_signature, normalise_text, scenario_id
from hog_sim.knowledge.store import KnowledgeStore
from hog_sim.policy.feasibility import Role
from hog_sim.world.changes import check_graph_change
from hog_sim.world.propagation import DeltaDistribution

MATCH_THRESHOLD = 0.6


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
        if not matches:
            self.last_source = "fallback"
            return self.fallback.forecast(
                state, scenario, actions, engine, delivery=delivery, limits=limits, sacked=sacked
            )
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
