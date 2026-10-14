"""Fiscal events (review of 13 October, O6): the Budget is the turn to change taxes and spending.

Outside a Budget turn (``world.calendar``), every tax or spending measure costs
``OFF_BUDGET_SURCHARGE`` more political capital per unit, for the time a Chancellor spends
defending an emergency statement: the limits add it (``policy.limits.capital_cost``) and
say when the next Budget falls. On a Budget turn there is no surcharge.

An independent scorecard, like the one an official forecaster publishes with each Budget,
costs every tax and spending measure in the package: what it adds to the deficit each turn
it is paid, for how many turns, and over its life, with the package's total. The review
screen shows it whenever the package changes taxes or spending.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from hog_sim.core.models import Model, PolicyAction
from hog_sim.core.state import WorldState
from hog_sim.world.calendar import calendar_event
from hog_sim.world.propagation import FISCAL_NODE, fiscal_cost, fiscal_turns, scale

OFF_BUDGET_SURCHARGE = 0.5  # extra capital per unit for a tax or spending change off-Budget
FISCAL_KINDS = {"tax", "spend"}
BUDGET = "budget"
_LOOKAHEAD = 24


def is_budget(turn: int) -> bool:
    event = calendar_event(turn)
    return event is not None and event.kind == BUDGET


def next_budget(turn: int) -> int | None:
    """The first Budget turn after ``turn``, if one falls within two years."""
    return next((t for t in range(turn + 1, turn + _LOOKAHEAD + 1) if is_budget(t)), None)


def off_budget(action: PolicyAction, state: WorldState) -> bool:
    """Whether ``action`` changes taxes or spending outside a Budget turn."""
    return action.kind in FISCAL_KINDS and bool(action.magnitude) and not is_budget(state.turn)


def off_budget_note(actions: Sequence[PolicyAction], state: WorldState) -> str | None:
    """The limits' note when the package changes taxes or spending outside the Budget."""
    if not any(off_budget(a, state) for a in actions):
        return None
    when = next_budget(state.turn)
    later = f"; the next Budget is turn {when}" if when is not None else ""
    return (
        f"Outside the Budget: tax and spending changes cost {OFF_BUDGET_SURCHARGE:g} more "
        f"political capital per unit this turn{later}"
    )


class ScoreLine(Model):
    """One measure on the scorecard: points of GDP added to the deficit each turn it is
    paid (negative when it raises money), for ``turns`` turns."""

    action: PolicyAction
    per_turn: float
    turns: int

    @property
    def total(self) -> float:
        return self.per_turn * self.turns


def scorecard(actions: Sequence[PolicyAction], state: WorldState) -> list[ScoreLine]:
    """The independent scorecard for the package's tax and spending measures."""
    if FISCAL_NODE not in state.indicators:
        return []
    step = scale(state, FISCAL_NODE)
    lines = []
    for action in actions:
        if action.kind not in FISCAL_KINDS or not action.magnitude:
            continue
        per_turn = fiscal_cost(action, state) * step
        lines.append(ScoreLine(action=action, per_turn=per_turn, turns=fiscal_turns(action, state)))
    return lines


def scorecard_lines(
    actions: Sequence[PolicyAction],
    state: WorldState,
    describe: Callable[[PolicyAction], str] = lambda a: f"{a.kind} {a.target}",
) -> list[str]:
    """The scorecard as lines for the review screen; empty when nothing is scored."""
    card = scorecard(actions, state)
    if not card:
        return []
    lines = ["Independent scorecard (deficit, % of GDP; + borrows, - raises money):"]
    for line in card:
        lines.append(
            f"  - {describe(line.action)}: {line.per_turn:+.2f} a turn for {line.turns} "
            f"turns, {line.total:+.1f} in all"
        )
    total = sum(line.total for line in card)
    if total > 0.05:
        verdict = "adds to borrowing"
    elif total < -0.05:
        verdict = "cuts borrowing"
    else:
        verdict = "is paid for"
    lines.append(f"  Package: {total:+.1f} over its life, so it {verdict}")
    return lines
