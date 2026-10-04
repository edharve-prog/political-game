"""Rule-based stand-ins for the LLM parts, so the game is playable and testable offline.

Project 5 replaces ``CannedScenarios`` and ``KeywordInterpreter``; Project 6 replaces
``EngineForecaster``. All three satisfy the protocols in ``game/interfaces.py``.
"""

from __future__ import annotations

import re

from hog_sim.core.config import make_rng
from hog_sim.core.models import (
    ApprovalEvent,
    Delivery,
    Outcome,
    Pledge,
    PolicyAction,
    Scenario,
    Shock,
)
from hog_sim.core.state import WorldState
from hog_sim.game.records import TurnRecord
from hog_sim.policy.pledges import broken_by
from hog_sim.population.popularity import policy_events, target_approval
from hog_sim.world.cast import active_cast
from hog_sim.world.graph import build_graph, exposed_groups
from hog_sim.world.propagation import DeltaDistribution, apply_deltas

_SCENARIOS = [
    Scenario(
        title="Gas prices spike",
        briefing="A cold snap and supply outages in Europe push wholesale gas prices up sharply.",
        affected_nodes=["indicator:energy_prices", "sector:energy"],
        urgency=0.8,
        suggested_options=["Cap household bills", "Tax energy windfall profits", "Wait it out"],
        shocks=[Shock(node="indicator:energy_prices", delta=1.5)],
    ),
    Scenario(
        title="Inflation surprise",
        briefing=(
            "Monthly figures show prices rising faster than forecast; markets expect rates to rise."
        ),
        affected_nodes=["indicator:inflation", "indicator:interest_rate"],
        urgency=0.6,
        suggested_options=["Cut public spending", "Back the central bank", "Do nothing"],
        shocks=[Shock(node="indicator:inflation", delta=0.5)],
    ),
    Scenario(
        title="Housing crunch",
        briefing="Rents and house prices hit records and young voters are angry.",
        affected_nodes=["indicator:house_prices", "sector:housing", "group:young_renters"],
        urgency=0.5,
        suggested_options=["Spend on housebuilding", "Deregulate planning", "Do nothing"],
        shocks=[Shock(node="indicator:house_prices", delta=0.8)],
    ),
    Scenario(
        title="Trade dispute with China",
        briefing="China slaps tariffs on British exports after a diplomatic row.",
        affected_nodes=["country:china", "sector:manufacturing"],
        urgency=0.6,
        suggested_options=["Negotiate", "Retaliate", "Support manufacturers"],
        shocks=[Shock(node="sector:manufacturing", delta=-0.8)],
    ),
    Scenario(
        title="Public sector pay row",
        briefing="Unions threaten strikes across schools and hospitals over pay.",
        affected_nodes=["sector:public", "group:public_workers"],
        urgency=0.7,
        suggested_options=["Fund a pay rise", "Hold firm", "Negotiate"],
    ),
]


class CannedScenarios:
    """Picks a scenario from a fixed list, seeded and without immediate repeats."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def next_scenario(self, state: WorldState, history: list[TurnRecord]) -> Scenario:
        last = history[-1].scenario.title if history else None
        options = [s for s in _SCENARIOS if s.title != last]
        return make_rng(self.seed, state.turn, "scenario").choice(options).model_copy(deep=True)


# Keyword -> node id. Checked in order; first match wins.
_TARGETS = [
    (r"\bbills?\b|price cap|\bcap\b", "indicator:energy_prices"),
    (r"energy|gas|electric", "sector:energy"),
    (r"bank|financ|city", "sector:finance"),
    (r"manufactur|industr|factor|export", "sector:manufacturing"),
    (r"hous|home|rent|planning|build", "sector:housing"),
    (r"public|nhs|hospital|school|pay|union|wages?", "sector:public"),
    (r"china", "country:china"),
    (r"\beu\b|europe", "country:eu"),
]
_SIZE = [(r"huge|major|big|massive", 0.9), (r"small|modest|slight", 0.25)]


class KeywordInterpreter:
    """Turns free text into PolicyActions with keyword rules: one per line of a package built
    from "Option:" and "Also:" lines (backlog story RB-1), otherwise one for the whole text."""

    def __init__(self) -> None:
        self._delivery: Delivery | None = None
        self._pledges: list[Pledge] = []
        self._sacked: list[str] = []

    def last_delivery(self) -> Delivery | None:
        return self._delivery

    def last_pledges(self) -> list[Pledge]:
        return list(self._pledges)

    def last_sacked(self) -> list[str]:
        return list(self._sacked)

    def delivery_for(self, text: str, state: WorldState) -> Delivery:
        return keyword_delivery(text, state)

    def interpret(self, text: str, state: WorldState, scenario: Scenario) -> list[PolicyAction]:
        self._delivery = keyword_delivery(text, state)
        self._pledges, text = keyword_pledges(text)
        self._sacked, text = keyword_sackings(text, state)
        parts = re.findall(r"^\s*(?:Option|Also):\s*(.+)$", text, re.M)
        if len(parts) < 2:
            return self._one(text, state, scenario)
        actions = [a for part in parts for a in self._one(part, state, scenario)]
        return [a for a in actions if a.kind != "do_nothing"] or actions[:1]

    def _one(self, text: str, state: WorldState, scenario: Scenario) -> list[PolicyAction]:
        t = text.lower()
        if not t.strip() or re.search(r"nothing|wait|hold firm", t):
            return [PolicyAction(kind="do_nothing", target=state.player_country, magnitude=0)]
        target = next((node for pat, node in _TARGETS if re.search(pat, t)), None)
        if target is None:
            target = next(
                (n for n in scenario.affected_nodes if not n.startswith("indicator:")),
                state.player_country,
            )
        size = next((m for pat, m in _SIZE if re.search(pat, t)), 0.5)

        if target.startswith("indicator:"):
            # Aimed at a price itself: the player wants it down unless they say otherwise.
            kind = "regulate" if re.search(r"\bcap|freez|limit|regulat", t) else "spend"
            size = size if re.search(r"\braise|\bincrease|\bhigher", t) else -size
        elif target.startswith("country:"):
            kind = "diplomatic"
            size = -size if re.search(r"retaliat|sanction|tariff", t) else size
        elif re.search(r"\bcut\w*\b.*\btax|\btax\w*\b.*\bcut", t):
            kind, size = "tax", -size
        elif re.search(r"tax|levy|windfall", t):
            kind = "tax"
        elif re.search(r"deregulat|scrap rules|relax", t):
            kind = "deregulate"
        elif re.search(r"regulat|cap\b|ban|limit", t):
            kind = "regulate"
        elif re.search(r"\bcut|austerity|squeeze", t):
            kind, size = "spend", -size
        elif re.search(r"spend|invest|fund|subsid|support|build|pay rise", t):
            kind = "spend"
        else:
            kind = "communicate"
        return [PolicyAction(kind=kind, target=target, magnitude=size, rationale=f"from: {text}")]


# Who "consulting X" means, for the keyword interpreter's delivery (story RB-4).
_CONSULT_WHO = [
    (r"union|workers|staff|nurses|teachers", "group:public_workers"),
    (r"business|industry|employers|firms|cbi", "group:business"),
    (r"pensioner", "group:pensioners"),
    (r"renter|tenant", "group:young_renters"),
    (r"\bmps\b|backbench|parliament|party", "institution:legislature"),
    (r"bank of england|the bank|governor", "institution:central_bank"),
]


def keyword_delivery(text: str, state: WorldState) -> Delivery:
    t = text.lower()
    consulted = []
    if re.search(r"consult|talks? with|negotiat|meet|sit down with|work with", t):
        known = set(state.node_ids())
        consulted = [n for pat, n in _CONSULT_WHO if re.search(pat, t) and n in known]
    phased = re.search(r"phase|gradual|staged|stagger|over (?:the next )?\w+ (?:years|months)", t)
    return Delivery(consulted=consulted, speed="phased" if phased else "immediate")


_NEGATION = r"\b(?:no|not|never|won't|will not|nor)\b"


def keyword_pledges(text: str) -> tuple[list[Pledge], str]:
    """Promises in ``text`` ("we will not raise taxes", "no cuts to schools"), and the text
    with those sentences taken out so they don't also read as actions (story SD-5)."""
    pledges, kept = [], []
    for sentence in re.split(r"(?<=[.;!])\s+|\n", text):
        t = sentence.lower()
        words = re.sub(r"^\s*(?:Option|Also):\s*", "", sentence).strip(" .;!")
        pledge = None
        if re.search(_NEGATION, t):
            if re.search(r"\btax", t) and re.search(r"rais|increas|\bnew\b|\brise|put up", t):
                pledge = Pledge(text=words, kind="tax", direction="up")
            elif re.search(r"\bcuts?\b", t):
                target = next((node for pat, node in _TARGETS if re.search(pat, t)), None)
                pledge = Pledge(text=words, kind="spend", direction="down", target=target)
        if pledge:
            pledges.append(pledge)
        else:
            kept.append(sentence)
    return pledges, "\n".join(kept) if pledges else text


def keyword_sackings(text: str, state: WorldState) -> tuple[list[str], str]:
    """Ministers ``text`` sacks ("sack the Chancellor"), matched on a word of their role, and
    the text with those sentences taken out (story SD-4)."""
    sacked, kept = [], []
    for sentence in re.split(r"(?<=[.;!])\s+|\n", text):
        t = sentence.lower()
        hits = []
        if re.search(r"\b(?:sack|fire|dismiss|sacking|replace)\b", t):
            for character in active_cast(state):
                words = [w for w in re.findall(r"[a-z]+", character.role.lower()) if len(w) > 4]
                if character.minister and any(w in t for w in words[:1]):
                    hits.append(character.id)
        if hits:
            sacked += hits
        else:
            kept.append(sentence)
    return sacked, "\n".join(kept) if sacked else text


class EngineForecaster:
    """Three candidates around the engine's expectation: as expected, backlash, welcomed.

    The engine decides who feels what: groups whose target approval the forecast raises are
    the ones who welcome the response, and groups it lowers are the ones who push back. The
    two turns are equally likely, so acting is not penalised on average (EB-7) and a response
    that helps people gets amplified rather than punished.
    """

    TURN = 2  # judge the response by the engine's forecast this many turns out
    THRESHOLD = 0.002  # smaller moves in target approval count as unaffected
    EFFECT = 0.03  # size of the reaction for groups the forecast moves
    FALLBACK_EFFECT = 0.02  # size when the forecast moves nobody, felt by the exposed groups

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
        graph = build_graph(state)
        exposed: set[str] = set()
        for action in actions:
            if action.kind != "do_nothing":
                exposed |= exposed_groups(graph, action.target)
                if action.target in state.groups:
                    exposed.add(action.target)
        exposed |= {n for n in scenario.affected_nodes if n.startswith("group:")}
        expected = {
            n: round(v, 3) for n, v in engine.at(0).items() if n.startswith("indicator:") and v
        }

        projected = apply_deltas(state, engine.at(min(self.TURN, engine.horizon - 1)))
        projected.events = [*state.events, *policy_events(actions)]
        before, after = target_approval(state, state), target_approval(projected, state)
        change = {g: after[g] - before[g] for g in state.groups}
        winners = sorted(g for g, c in change.items() if c > self.THRESHOLD)
        losers = sorted(g for g, c in change.items() if c < -self.THRESHOLD)

        def event(name: str, groups: list[str], size: float) -> list[ApprovalEvent]:
            if groups:
                return [ApprovalEvent(name=name, group_effects={g: size for g in groups})]
            if exposed:
                fallback = self.FALLBACK_EFFECT if size > 0 else -self.FALLBACK_EFFECT
                return [
                    ApprovalEvent(name=name, group_effects={g: fallback for g in sorted(exposed)})
                ]
            return []

        # Consulting first takes some of the risk of a backlash away (story RB-4), without
        # making a warm welcome any likelier.
        backlash = 0.12 if delivery is not None and delivery.consulted else 0.2
        broken = "".join(
            f' Critics say it breaks the pledge "{p.text}".' for p in broken_by(state, actions)
        )
        broken += "".join(
            f" {state.characters[c].name} is sacked as {state.characters[c].role}."
            for c in sacked or []
            if c in state.characters
        )
        outcomes = [
            Outcome(
                narrative=f"{scenario.title}: the response lands broadly as expected.",
                indicator_deltas=expected,
                probability=0.8 - backlash,
            ),
            Outcome(
                narrative=f"{scenario.title}: the response sparks a backlash from those affected.",
                indicator_deltas=expected,
                approval_events=event("Backlash", losers, -self.EFFECT),
                probability=backlash,
            ),
            Outcome(
                narrative=f"{scenario.title}: the response is welcomed as decisive.",
                indicator_deltas=expected,
                approval_events=event("Seen as decisive", winners, self.EFFECT),
                probability=0.2,
            ),
        ]
        for outcome in outcomes:
            outcome.narrative += broken
        return outcomes
