"""What tagged outcome events do to the world (Project 16, EB-5).

An outcome's ``events`` are tags such as ``strike`` or ``market_selloff`` (see
``forecasting.scoring.EventTag``). Each tag brings a fixed set of shocks, in standard steps,
so a narrative that says "markets sold off" moves Bank Rate and finance as well. Nodes the
world doesn't have are skipped. Reputation-only events (media coverage, foreign praise) have
no economic effect: their weight falls on approval through the outcome's group effects.

An outcome cannot move one node further than ``MAX_OUTCOME_STEPS`` a turn, however its
shocks are written: a tag listed twice counts once, and ``outcome_shocks`` scales the shocks
on each node (the candidate's own and its tags') so their total stays within the cap. Without
that, five copies of a one-step shock were a five-step shock.
"""

from __future__ import annotations

from collections import defaultdict

from hog_sim.core.models import Outcome, Shock
from hog_sim.core.state import WorldState

EVENT_SHOCKS: dict[str, dict[str, float]] = {
    "strike": {"sector:public": -0.5},
    "protest": {"institution:legislature": -0.2},
    "backbench_rebellion": {"institution:legislature": -0.5},
    "market_selloff": {"indicator:gilt_yield": 0.5, "sector:finance": -0.5},
    "market_rally": {"indicator:gilt_yield": -0.3, "sector:finance": 0.3},
    "capital_flight": {"sector:finance": -0.8, "indicator:gilt_yield": 0.4},
    "business_investment": {"sector:manufacturing": 0.5},
    "foreign_retaliation": {"sector:manufacturing": -0.4},
}


MAX_OUTCOME_STEPS = 1.0  # most an outcome moves any one node, standard steps a turn


def event_shocks(events: list[str], state: WorldState) -> list[Shock]:
    """Shocks for an outcome's event tags (each tag once), for the nodes ``state`` has."""
    known = set(state.node_ids())
    return [
        Shock(node=node, delta=delta)
        for tag in dict.fromkeys(events)
        for node, delta in EVENT_SHOCKS.get(tag, {}).items()
        if node in known
    ]


def cap_shocks(shocks: list[Shock], cap: float = MAX_OUTCOME_STEPS) -> list[Shock]:
    """The shocks with each node's total scaled down to ``cap`` standard steps a turn.

    A node's total is the sum of its shocks' deltas; when that is past the cap, each of them
    is scaled by the same share, so their timing and shape are kept."""
    totals: dict[str, float] = defaultdict(float)
    for shock in shocks:
        totals[shock.node] += shock.delta
    capped = []
    for shock in shocks:
        total = abs(totals[shock.node])
        share = cap / total if total > cap else 1.0
        if shock.delta * share:
            capped.append(shock.model_copy(update={"delta": shock.delta * share}))
    return capped


def outcome_shocks(outcome: Outcome, state: WorldState) -> list[Shock]:
    """Everything an outcome pushes: its own shocks and its event tags', capped per node."""
    known = set(state.node_ids())
    own = [s for s in outcome.shocks if s.node in known]
    return cap_shocks(own + event_shocks(outcome.events, state))
