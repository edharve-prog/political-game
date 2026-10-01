"""Group approval, national approval, vote intention and elections.

Each turn every group's approval moves part of the way towards a target:

    target = lean
           + K * sum(CARES_ABOUT weight * indicator change in standard steps)
           + K * sum(EMPLOYS weight * sector output change in standard steps)
           + K * sum(INFLUENCES weight * institution support above or below 0.5, in steps)
           + sum(active event effects, halving every half_life_turns)
           - DEBT_PENALTY * (deficit above DEFICIT_TOLERANCE, in % of GDP)

The last term is a loss of fiscal credibility felt by every group: borrowing is tolerated up
to a point, then each extra point of deficit costs approval across the board, so spending
on everything cannot buy an election.

Changes are measured against a ``reference`` state (the start state, or a rolling
snapshot chosen by the game loop). ``K`` converts one weighted standard step into
approval points. Partial adjustment gives the electorate memory: a shock is felt over
several turns rather than all at once, and approval drifts back to lean once it passes.
"""

from __future__ import annotations

from hog_sim.core.models import EdgeKind, Model
from hog_sim.core.state import WorldState
from hog_sim.world.propagation import METRICS, scale

K = 0.05  # approval per weighted standard step
ADJUST = 0.5  # share of the gap to target closed each turn
EVENT_FLOOR = 1e-3  # events weaker than this are dropped
DEFICIT_ID = "indicator:deficit"
DEFICIT_TOLERANCE = 6.0  # % of GDP voters accept before credibility suffers
DEBT_PENALTY = 0.015  # approval lost per point of deficit above the tolerance


def _steps(state: WorldState, reference: WorldState, node_id: str) -> float:
    node, ref = state.node(node_id), reference.node(node_id)
    field = METRICS[node.kind]
    return (getattr(node, field) - getattr(ref, field)) / scale(reference, node_id)


def _event_weight(age: int, half_life: float) -> float:
    return 0.5 ** (age / half_life)


def target_approval(state: WorldState, reference: WorldState) -> dict[str, float]:
    targets = {gid: g.lean for gid, g in state.groups.items()}
    for edge in state.edges:
        if edge.kind == EdgeKind.CARES_ABOUT and edge.source in targets:
            targets[edge.source] += K * edge.weight * _steps(state, reference, edge.target)
        elif edge.kind == EdgeKind.EMPLOYS and edge.target in targets:
            targets[edge.target] += K * edge.weight * _steps(state, reference, edge.source)
        elif edge.kind == EdgeKind.INFLUENCES and edge.target in targets:
            institution = state.institutions.get(edge.source)
            if institution is not None:
                targets[edge.target] += K * edge.weight * (institution.support - 0.5) / 0.1
    deficit = state.indicators.get(DEFICIT_ID)
    if deficit is not None and deficit.value > DEFICIT_TOLERANCE:
        penalty = DEBT_PENALTY * (deficit.value - DEFICIT_TOLERANCE)
        targets = {gid: t - penalty for gid, t in targets.items()}
    for event in state.events:
        w = _event_weight(event.age_turns, event.half_life_turns)
        for gid, effect in event.group_effects.items():
            if gid in targets:
                targets[gid] += effect * w
    return {gid: min(1.0, max(0.0, t)) for gid, t in targets.items()}


def step_approval(state: WorldState, reference: WorldState) -> WorldState:
    """Advance approval one turn and age events. Returns a new state."""
    targets = target_approval(state, reference)
    new = state.snapshot()
    for gid, group in new.groups.items():
        group.approval = min(
            1.0, max(0.0, group.approval + ADJUST * (targets[gid] - group.approval))
        )
    for event in new.events:
        event.age_turns += 1
    new.events = [
        e
        for e in new.events
        if max(map(abs, e.group_effects.values()), default=0)
        * _event_weight(e.age_turns, e.half_life_turns)
        >= EVENT_FLOOR
    ]
    return new


def national_approval(state: WorldState) -> float:
    """Population-weighted approval across all groups."""
    total = sum(g.population_share for g in state.groups.values())
    return sum(g.population_share * g.approval for g in state.groups.values()) / total


def vote_intention(state: WorldState) -> float:
    """Government vote share: approval weighted by population share and turnout."""
    weights = {gid: g.population_share * g.turnout for gid, g in state.groups.items()}
    total = sum(weights.values())
    return sum(weights[gid] * g.approval for gid, g in state.groups.items()) / total


def seat_share(vote: float, exponent: float = 3.0) -> float:
    """Two-party cube law: small vote leads become large seat leads."""
    a, b = vote**exponent, (1 - vote) ** exponent
    return a / (a + b)


class ElectionResult(Model):
    vote_share: float
    seats: int
    total_seats: int

    @property
    def majority(self) -> bool:
        return self.seats > self.total_seats // 2


def run_election(state: WorldState, total_seats: int = 650) -> ElectionResult:
    vote = vote_intention(state)
    return ElectionResult(
        vote_share=vote, seats=round(seat_share(vote) * total_seats), total_seats=total_seats
    )
