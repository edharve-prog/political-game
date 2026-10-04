"""Pledges: promises the leader makes, remembered across turns (backlog story SD-5).

The interpreter extracts a pledge from what the player says ("we will not raise taxes").
From the next turn on, an applied action that the pledge rules out breaks it, in
``resolve()``: the pledge is marked broken and the groups who cared take an approval hit.
Prompts show every pledge, kept or broken, so scenarios and outcomes can refer back to it.
"""

from __future__ import annotations

from collections.abc import Sequence

from hog_sim.core.models import ApprovalEvent, Pledge, PolicyAction
from hog_sim.core.state import WorldState

BROKEN_HIT = 0.05  # approval lost by each group that cared
BROKEN_HIT_ALL = 0.02  # lost by every group when the pledge names none
BROKEN_HALF_LIFE = 4.0
BROKEN_HOLD = 1


def breaks(pledge: Pledge, action: PolicyAction) -> bool:
    if action.kind != pledge.kind or action.magnitude == 0:
        return False
    if pledge.target is not None and action.target != pledge.target:
        return False
    if pledge.direction == "up":
        return action.magnitude > 0
    if pledge.direction == "down":
        return action.magnitude < 0
    return True


def kept_pledges(state: WorldState) -> list[Pledge]:
    return [p for p in state.pledges if p.broken_turn is None]


def broken_by(state: WorldState, actions: Sequence[PolicyAction]) -> list[Pledge]:
    """The kept pledges that ``actions`` would break."""
    return [p for p in kept_pledges(state) if any(breaks(p, a) for a in actions)]


def broken_event(pledge: Pledge, state: WorldState) -> ApprovalEvent:
    groups = [g for g in pledge.groups if g in state.groups]
    effects = (
        {g: -BROKEN_HIT for g in groups} if groups else {g: -BROKEN_HIT_ALL for g in state.groups}
    )
    return ApprovalEvent(
        name=f"Broke pledge: {pledge.text}",
        group_effects=effects,
        half_life_turns=BROKEN_HALF_LIFE,
        hold_turns=BROKEN_HOLD,
    )


def apply_pledges(
    state: WorldState, turn: int, actions: Sequence[PolicyAction], made: Sequence[Pledge]
) -> WorldState:
    """Break the kept pledges that ``actions`` rule out, then add the pledges ``made`` this
    turn (a promise repeating a kept one is not added twice). Deterministic."""
    new = state.snapshot()
    broken = {id(p) for p in broken_by(new, actions)}
    for pledge in new.pledges:
        if id(pledge) in broken:
            pledge.broken_turn = turn
            new.events.append(broken_event(pledge, new))
    rules = {p.rule() for p in kept_pledges(new)}
    for pledge in made:
        if pledge.rule() not in rules:
            new.pledges.append(pledge.model_copy(update={"made_turn": turn, "broken_turn": None}))
            rules.add(pledge.rule())
    return new


def pledge_line(pledge: Pledge) -> str:
    state = (
        f"BROKEN on turn {pledge.broken_turn}" if pledge.broken_turn is not None else "kept so far"
    )
    return f'"{pledge.text}" (made turn {pledge.made_turn}, {state})'
