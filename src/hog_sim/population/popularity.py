"""Group approval, national approval, vote intention and elections.

Each turn every group's approval moves part of the way towards a target:

    target = lean
           + K * sum(CARES_ABOUT weight * indicator change in standard steps)
           + K * sum(EMPLOYS weight * sector output change in standard steps)
           + K * sum(INFLUENCES weight * institution support above or below 0.5, in steps)
           + capped(gains from active events) - capped(losses from active events)
           - DEBT_PENALTY * (deficit above DEFICIT_TOLERANCE, in % of GDP)

Events are held for hold_turns, then halve every half_life_turns. A group's gains and its
losses are capped separately, each below ``EVENT_CAP`` (``cap_events``), so a pile of good news
cannot absorb a later backlash: bad news always lowers the target (E8).

The deficit term is a loss of fiscal credibility felt by every group: borrowing is tolerated up
to a point, then each extra point of deficit costs approval across the board, so spending
on everything cannot buy an election.

Changes are measured against what voters are used to: each node's baseline starts at its
level in the ``reference`` state (the start) and drifts towards the current level with a
half-life of ``HABIT_HALF_LIFE`` turns (EB-11). A price rise hurts most when it is new; a
year later it is partly the new normal. ``K`` converts one weighted standard step into
approval points. Partial adjustment gives the electorate memory: a shock is felt over
several turns rather than all at once, and approval drifts back to lean once it passes.

Turnout is not fixed (O8): a group turns out more the further its approval sits from its
lean, in either direction, because enthusiasm and anger both bring people to the polls. Its
effective turnout is ``turnout * (1 + TURNOUT_SWING * |approval - lean|)``, at most
``MAX_TURNOUT``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from hog_sim.core.models import ApprovalEvent, EdgeKind, Group, Model, PolicyAction
from hog_sim.core.state import WorldState
from hog_sim.policy.credibility import approval_shift
from hog_sim.world.levers import (
    LAW_HOLD,
    RULE_KINDS,
    delivers_for,
    deregulation_losers,
    law_counterparties,
)
from hog_sim.world.propagation import METRICS, action_factor, scale, unfunded_tax_cut

K = 0.05  # approval per weighted standard step
ADJUST = 0.5  # share of the gap to target closed each turn
EVENT_FLOOR = 1e-3  # events weaker than this are dropped
EVENT_CAP = 0.15  # most that a group's active gains, or its losses, move its target (EB-6)
EVENT_KNEE = 0.10  # gains or losses up to this count in full; beyond it they taper to the cap
TURNOUT_SWING = 2.0  # turnout rises 2% for each approval point away from lean (O8)
MAX_TURNOUT = 0.95
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
# Fairness (E6): money handed to one group (spending on it, a tax cut for it) costs every
# other group this share of what it wins, for as long as it runs: the others see someone
# else's pension rise or tax break, not their own.
FAIRNESS = 0.25


def policy_events(
    actions: list[PolicyAction], state: WorldState | None = None
) -> list[ApprovalEvent]:
    """Approval events for actions targeting population groups.

    The effect holds at full strength for the action's duration, then fades with the usual
    half-life, so a lasting benefit keeps its voters for as long as it is paid for. Given the
    state, money handed to one group costs the others ``FAIRNESS`` of what it wins. A law for
    a group also costs its counterparty, and holds only ``LAW_HOLD`` turns unless the package
    puts money behind the group; deregulating a sector costs the workers whose protections it
    loosens (``world.levers``). An unfunded tax cut that spooks markets costs every group
    approval.
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
        hold = action.duration_turns - 1
        losers = deregulation_losers(action)
        if losers:
            events.append(
                ApprovalEvent(
                    name=f"{action.kind} {action.target}: protections loosened",
                    group_effects={
                        g: steps * action.magnitude * GROUP_STEP
                        for g, steps in losers.items()
                        if g in groups or state is None
                    },
                    hold_turns=hold,
                )
            )
        if not action.target.startswith("group:"):
            continue
        effect = action_factor(action) * action.magnitude * GROUP_STEP
        if not effect:
            continue
        effects = {action.target: effect}
        if action.kind in ("tax", "spend") and effect > 0:
            effects |= {g: -FAIRNESS * effect for g in groups if g != action.target}
        if action.kind in RULE_KINDS:
            # A law for one group has a losing side, and without money behind it its appeal
            # wears thin (O2).
            effects |= {
                g: share * effect
                for g, share in law_counterparties(action).items()
                if g != action.target and (g in groups or state is None)
            }
            if not delivers_for(action.target, actions):
                hold = min(hold, LAW_HOLD)
        events.append(
            ApprovalEvent(
                name=f"{action.kind} {action.target}",
                group_effects=effects,
                # Felt at full strength on the turn it starts and each turn after it while it
                # runs: duration_turns updates in all (EC-7).
                hold_turns=hold,
            )
        )
    return events


def cap_events(total: float) -> float:
    """``total`` gains (or, negated, losses) from events, as they move a target: in full up to
    ``EVENT_KNEE``, then tapering so they approach ``EVENT_CAP`` but never reach it. Each extra
    event still counts for something, so piling on news moves a group less and less, but the
    cap never hides news of the other sign (E8)."""
    sign = 1.0 if total >= 0 else -1.0
    size = abs(total)
    if size <= EVENT_KNEE:
        return total
    room = EVENT_CAP - EVENT_KNEE
    return sign * (EVENT_KNEE + room * (1 - math.exp(-(size - EVENT_KNEE) / room)))


@dataclass(slots=True)
class Term:
    """One part of a group's target approval (E20): ``kind`` is lean, cares, employs,
    influences, credibility (D9), debt or events, ``node`` the node it comes from (empty
    for lean, credibility and events) and ``value`` its contribution in approval (0-1)."""

    kind: str
    value: float
    node: str = ""
    events: list[tuple[str, float]] = field(default_factory=list)  # kind "events": each one


def approval_terms(
    state: WorldState,
    reference: WorldState,
    baselines: Mapping[str, float] | None = None,
) -> dict[str, list[Term]]:
    """Each group's target approval split into its terms, before the 0-1 clamp. The terms
    add up (in order) to what ``target_approval`` returns, so explanations name the real
    causes: lean, each cared-about indicator, each employing sector, each institution, the
    government's credibility, the debt penalty and the capped events in force."""
    terms = {gid: [Term(kind="lean", value=g.lean)] for gid, g in state.groups.items()}
    for edge in state.edges:
        if edge.kind == EdgeKind.CARES_ABOUT and edge.source in terms:
            steps = _steps(state, reference, edge.target, baselines)
            terms[edge.source].append(
                Term(kind="cares", node=edge.target, value=K * edge.weight * steps)
            )
        elif edge.kind == EdgeKind.EMPLOYS and edge.target in terms:
            steps = _steps(state, reference, edge.source, baselines)
            terms[edge.target].append(
                Term(kind="employs", node=edge.source, value=K * edge.weight * steps)
            )
        elif edge.kind == EdgeKind.INFLUENCES and edge.target in terms:
            institution = state.institutions.get(edge.source)
            if institution is not None:
                value = K * edge.weight * (institution.support - 0.5) / 0.1
                terms[edge.target].append(Term(kind="influences", node=edge.source, value=value))
    shift = approval_shift(state.credibility)
    if shift:
        for group_terms in terms.values():
            group_terms.append(Term(kind="credibility", value=shift))
    deficit = state.indicators.get(DEFICIT_ID)
    if deficit is not None and deficit.value > DEFICIT_TOLERANCE:
        penalty = DEBT_PENALTY * (deficit.value - DEFICIT_TOLERANCE)
        for group_terms in terms.values():
            group_terms.append(Term(kind="debt", node=DEFICIT_ID, value=-penalty))
    felt: dict[str, list[tuple[str, float]]] = {gid: [] for gid in terms}
    for event in state.events:
        w = _event_weight(event.age_turns, event.half_life_turns, event.hold_turns)
        for gid, effect in event.group_effects.items():
            if gid in felt:
                felt[gid].append((event.name, effect * w))
    for gid, events in felt.items():
        gains = sum(e for _, e in events if e > 0)
        losses = sum(e for _, e in events if e < 0)
        capped = cap_events(gains) + cap_events(losses)
        terms[gid].append(Term(kind="events", value=capped, events=events))
    return terms


def target_approval(
    state: WorldState,
    reference: WorldState,
    baselines: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Each group's target approval, with changes measured from ``baselines`` where given
    and from ``reference`` otherwise."""
    targets = {}
    for gid, group_terms in approval_terms(state, reference, baselines).items():
        total = 0.0
        for term in group_terms:
            total += term.value
        targets[gid] = min(1.0, max(0.0, total))
    return targets


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


def effective_turnout(group: Group) -> float:
    """Turnout raised by enthusiasm or anger: the further approval is from lean, either way,
    the more of the group votes."""
    swing = 1 + TURNOUT_SWING * abs(group.approval - group.lean)
    return min(MAX_TURNOUT, max(group.turnout, group.turnout * swing))


def vote_intention(state: WorldState) -> float:
    """Government vote share: approval weighted by population share and effective turnout."""
    weights = {gid: g.population_share * effective_turnout(g) for gid, g in state.groups.items()}
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

    @property
    def hung(self) -> bool:
        """Exactly half the seats: neither side has a majority (an even house, a tied vote)."""
        return self.seats * 2 == self.total_seats

    @property
    def verdict(self) -> str:
        if self.majority:
            return "You win a majority."
        if self.hung:
            return "Hung parliament: you are level with the opposition and short of a majority."
        return "You lose: the opposition wins a majority."


def run_election(state: WorldState, total_seats: int = 650) -> ElectionResult:
    if not state.groups:
        raise ValueError("an election needs population groups to vote")
    vote = vote_intention(state)
    return ElectionResult(
        vote_share=vote, seats=round(seat_share(vote) * total_seats), total_seats=total_seats
    )
