"""Group approval, national approval, vote intention and elections.

Each turn every group's approval moves part of the way towards a target:

    target = lean
           + K * sum(CARES_ABOUT weight * indicator change in standard steps)
           + K * sum(EMPLOYS weight * sector output change in standard steps)
           + K * sum(INFLUENCES weight * institution support above or below 0.5, in steps)
           + sum(active event effects, held for hold_turns then halving every half_life_turns),
             capped at +-EVENT_CAP per group
           - DEBT_PENALTY * (deficit above DEFICIT_TOLERANCE, in % of GDP)

The last term is a loss of fiscal credibility felt by every group: borrowing is tolerated up
to a point, then each extra point of deficit costs approval across the board, so spending
on everything cannot buy an election.

Changes are measured against what voters are used to: each node's baseline starts at its
level in the ``reference`` state (the start) and drifts towards the current level with a
half-life of ``HABIT_HALF_LIFE`` turns (EB-11). A price rise hurts most when it is new; a
year later it is partly the new normal. ``K`` converts one weighted standard step into
approval points. Partial adjustment gives the electorate memory: a shock is felt over
several turns rather than all at once, and approval drifts back to lean once it passes.
"""

from __future__ import annotations

from collections.abc import Mapping

from hog_sim.core.models import ApprovalEvent, EdgeKind, Model, PolicyAction
from hog_sim.core.state import WorldState
from hog_sim.world.propagation import METRICS, action_factor, scale, unfunded_tax_cut

K = 0.05  # approval per weighted standard step
ADJUST = 0.5  # share of the gap to target closed each turn
EVENT_FLOOR = 1e-3  # events weaker than this are dropped
EVENT_CAP = 0.15  # most that all active events together can move a group's target (EB-6)
DEFICIT_ID = "indicator:deficit"
DEFICIT_TOLERANCE = 5.0  # % of GDP voters accept before credibility suffers
DEBT_PENALTY = 0.03  # approval lost per point of deficit above the tolerance
HABIT_HALF_LIFE = 12  # turns for voters to take half of a lasting change as normal (EB-11)
HABITUATION = 1 - 0.5 ** (1 / HABIT_HALF_LIFE)  # share of the gap a baseline closes each turn


def _level(state: WorldState, node_id: str) -> float:
    node = state.node(node_id)
    return getattr(node, METRICS[node.kind])


def _steps(
    state: WorldState,
    reference: WorldState,
    node_id: str,
    baselines: Mapping[str, float] | None = None,
) -> float:
    base = (baselines or {}).get(node_id)
    if base is None:
        base = _level(reference, node_id)
    return (_level(state, node_id) - base) / scale(reference, node_id)


def _felt_nodes(state: WorldState) -> set[str]:
    """Nodes whose changes voters feel directly: what groups care about or work in."""
    nodes = set()
    for edge in state.edges:
        if edge.kind == EdgeKind.CARES_ABOUT and edge.source in state.groups:
            nodes.add(edge.target)
        elif edge.kind == EdgeKind.EMPLOYS and edge.target in state.groups:
            nodes.add(edge.source)
    return nodes


def adapt_baselines(state: WorldState, reference: WorldState) -> dict[str, float]:
    """Move each felt node's baseline ``HABITUATION`` of the way to its current level."""
    baselines = {}
    for node_id in sorted(_felt_nodes(state)):
        base = state.baselines.get(node_id, _level(reference, node_id))
        baselines[node_id] = base + HABITUATION * (_level(state, node_id) - base)
    return baselines


def _event_weight(age: int, half_life: float, hold: int = 0) -> float:
    return 0.5 ** (max(0, age - hold) / half_life)


# Approval every group loses per step of unfunded tax cut, when markets turn on the plan.
CONFIDENCE_APPROVAL = 0.12


# A policy aimed straight at a group (a pension rise, a tax on landlords) moves that group's
# target approval by this much per standard step, for as long as the policy runs.
GROUP_STEP = 0.05


def policy_events(
    actions: list[PolicyAction], state: WorldState | None = None
) -> list[ApprovalEvent]:
    """Approval events for actions targeting population groups.

    The effect holds at full strength for the action's duration, then fades with the usual
    half-life, so a lasting benefit keeps its voters for as long as it is paid for. Given the
    state, an unfunded tax cut that spooks markets also costs every group approval.
    """
    events = []
    loss = unfunded_tax_cut(actions, state)
    groups = state.groups if state is not None else {}
    if loss and groups:
        events.append(
            ApprovalEvent(
                name="Markets lose confidence in the government's plans",
                group_effects={g: -CONFIDENCE_APPROVAL * loss for g in groups},
            )
        )
    for action in actions:
        if not action.target.startswith("group:"):
            continue
        effect = action_factor(action) * action.magnitude * GROUP_STEP
        if effect:
            events.append(
                ApprovalEvent(
                    name=f"{action.kind} {action.target}",
                    group_effects={action.target: effect},
                    # Felt at full strength on the turn it starts and each turn after it
                    # while it runs: duration_turns updates in all (EC-7).
                    hold_turns=action.duration_turns - 1,
                )
            )
    return events


def target_approval(
    state: WorldState,
    reference: WorldState,
    baselines: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Each group's target approval, with changes measured from ``baselines`` where given
    and from ``reference`` otherwise."""
    targets = {gid: g.lean for gid, g in state.groups.items()}
    for edge in state.edges:
        if edge.kind == EdgeKind.CARES_ABOUT and edge.source in targets:
            steps = _steps(state, reference, edge.target, baselines)
            targets[edge.source] += K * edge.weight * steps
        elif edge.kind == EdgeKind.EMPLOYS and edge.target in targets:
            steps = _steps(state, reference, edge.source, baselines)
            targets[edge.target] += K * edge.weight * steps
        elif edge.kind == EdgeKind.INFLUENCES and edge.target in targets:
            institution = state.institutions.get(edge.source)
            if institution is not None:
                targets[edge.target] += K * edge.weight * (institution.support - 0.5) / 0.1
    deficit = state.indicators.get(DEFICIT_ID)
    if deficit is not None and deficit.value > DEFICIT_TOLERANCE:
        penalty = DEBT_PENALTY * (deficit.value - DEFICIT_TOLERANCE)
        targets = {gid: t - penalty for gid, t in targets.items()}
    felt = {gid: 0.0 for gid in targets}
    for event in state.events:
        w = _event_weight(event.age_turns, event.half_life_turns, event.hold_turns)
        for gid, effect in event.group_effects.items():
            if gid in felt:
                felt[gid] += effect * w
    for gid, total in felt.items():
        targets[gid] += max(-EVENT_CAP, min(EVENT_CAP, total))
    return {gid: min(1.0, max(0.0, t)) for gid, t in targets.items()}


def step_approval(state: WorldState, reference: WorldState) -> WorldState:
    """Advance approval one turn, let voters get used to the new levels, and age events.

    Approval is judged against the baselines voters held coming into the turn; the
    baselines then adapt. Returns a new state."""
    targets = target_approval(state, reference, state.baselines)
    new = state.snapshot()
    new.baselines = adapt_baselines(state, reference)
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
        * _event_weight(e.age_turns, e.half_life_turns, e.hold_turns)
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
