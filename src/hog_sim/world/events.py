"""What tagged outcome events do to the world (Project 16, EB-5).

An outcome's ``events`` are tags such as ``strike`` or ``market_selloff`` (see
``forecasting.scoring.EventTag``). Each tag brings a fixed set of shocks, in standard steps,
so a narrative that says "markets sold off" moves Bank Rate and finance as well. Nodes the
world doesn't have are skipped. Reputation-only events (media coverage, foreign praise) have
no economic effect: their weight falls on approval through the outcome's group effects.
"""

from __future__ import annotations

from hog_sim.core.models import Shock
from hog_sim.core.state import WorldState

EVENT_SHOCKS: dict[str, dict[str, float]] = {
    "strike": {"sector:public": -0.5},
    "protest": {"institution:legislature": -0.2},
    "backbench_rebellion": {"institution:legislature": -0.5},
    "market_selloff": {"indicator:interest_rate": 0.3, "sector:finance": -0.5},
    "market_rally": {"indicator:interest_rate": -0.2, "sector:finance": 0.3},
    "capital_flight": {"sector:finance": -0.8, "indicator:interest_rate": 0.2},
    "business_investment": {"sector:manufacturing": 0.5},
    "foreign_retaliation": {"sector:manufacturing": -0.4},
}


def event_shocks(events: list[str], state: WorldState) -> list[Shock]:
    """Shocks for an outcome's event tags, for the nodes ``state`` has."""
    known = set(state.node_ids())
    return [
        Shock(node=node, delta=delta)
        for tag in events
        for node, delta in EVENT_SHOCKS.get(tag, {}).items()
        if node in known
    ]
