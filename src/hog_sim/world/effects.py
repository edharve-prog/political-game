"""What a turn's decisions do to chosen nodes, on their own (E11, D5).

The engine's central projection of this turn's actions and nothing else: their shocks run
from a quiet world with the edge weights at their central values, over ``EFFECT_HORIZON``
turns, and each node's move is its average deviation over those turns, in standard steps. It
is the forecast's ``response_at`` without the Monte Carlo draws, so ``resolve`` can use it and
replay still reproduces it. A population group is not an engine node: its move is the
approval the actions win or lose it directly (``policy_events``), in standard steps.

Rules that judge a decision by what it does, not by what it names, use it: the cast's loyalty
(``world/cast.py``) and whether an in-tray item was helped (``world/storylines.py``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import fmean

from hog_sim.core.models import PolicyAction
from hog_sim.core.state import WorldState
from hog_sim.population.popularity import policy_events
from hog_sim.world.propagation import actions_to_shocks, scale, simulate

EFFECT_HORIZON = 4  # turns of a decision's effect that judge it


def decision_effects(
    state: WorldState,
    actions: Sequence[PolicyAction],
    nodes: Iterable[str],
    horizon: int = EFFECT_HORIZON,
) -> dict[str, float]:
    """Standard steps each of ``nodes`` moves on average over ``horizon`` turns from
    ``actions`` alone, in ``state`` (the world the turn is played in). Unknown nodes move 0."""
    nodes = list(dict.fromkeys(nodes))
    actions = [a for a in actions if a.kind != "do_nothing"]
    if not actions or not nodes:
        return dict.fromkeys(nodes, 0.0)
    shocks = actions_to_shocks(actions, state)
    path = simulate(state, shocks, horizon) if shocks else {}
    events = policy_events(actions, state) if any(n in state.groups for n in nodes) else []
    moves = {}
    for node_id in nodes:
        if node_id in state.groups:
            approval = sum(e.group_effects.get(node_id, 0.0) for e in events)
            moves[node_id] = approval / scale(state, node_id)
        elif node_id in path:
            moves[node_id] = fmean(path[node_id])
        else:
            moves[node_id] = 0.0
    return moves
