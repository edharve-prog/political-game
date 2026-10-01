"""Rule-based stand-ins for the LLM parts, so the game is playable and testable offline.

Project 5 replaces ``CannedScenarios`` and ``KeywordInterpreter``; Project 6 replaces
``EngineForecaster``. All three satisfy the protocols in ``game/interfaces.py``.
"""

from __future__ import annotations

import re

from hog_sim.core.config import make_rng
from hog_sim.core.models import ApprovalEvent, Outcome, PolicyAction, Scenario, Shock
from hog_sim.core.state import WorldState
from hog_sim.game.records import TurnRecord
from hog_sim.world.graph import build_graph, exposed_groups
from hog_sim.world.propagation import DeltaDistribution

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
    (r"energy|gas|electric|bills?", "sector:energy"),
    (r"bank|financ|city", "sector:finance"),
    (r"manufactur|industr|factor|export", "sector:manufacturing"),
    (r"hous|home|rent|planning|build", "sector:housing"),
    (r"public|nhs|hospital|school|pay|union|wages?", "sector:public"),
    (r"china", "country:china"),
    (r"\beu\b|europe", "country:eu"),
]
_SIZE = [(r"huge|major|big|massive", 0.9), (r"small|modest|slight", 0.25)]


class KeywordInterpreter:
    """Turns free text into at most one PolicyAction with keyword rules."""

    def interpret(self, text: str, state: WorldState, scenario: Scenario) -> list[PolicyAction]:
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

        if target.startswith("country:"):
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


class EngineForecaster:
    """Three candidates around the engine's expectation: as expected, backlash, welcomed."""

    def forecast(
        self,
        state: WorldState,
        scenario: Scenario,
        actions: list[PolicyAction],
        engine: DeltaDistribution,
    ) -> list[Outcome]:
        graph = build_graph(state)
        groups: set[str] = set()
        for action in actions:
            if action.kind != "do_nothing":
                groups |= exposed_groups(graph, action.target)
        groups |= {n for n in scenario.affected_nodes if n.startswith("group:")}
        expected = {
            n: round(v, 3) for n, v in engine.at(0).items() if n.startswith("indicator:") and v
        }

        def event(name: str, size: float) -> list[ApprovalEvent]:
            if not groups:
                return []
            return [ApprovalEvent(name=name, group_effects={g: size for g in sorted(groups)})]

        return [
            Outcome(
                narrative=f"{scenario.title}: the response lands broadly as expected.",
                indicator_deltas=expected,
                probability=0.6,
            ),
            Outcome(
                narrative=f"{scenario.title}: the response sparks a backlash from those affected.",
                indicator_deltas=expected,
                approval_events=event("Backlash", -0.03),
                probability=0.25,
            ),
            Outcome(
                narrative=f"{scenario.title}: the response is welcomed as decisive.",
                indicator_deltas=expected,
                approval_events=event("Seen as decisive", 0.02),
                probability=0.15,
            ),
        ]
