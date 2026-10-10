"""What the player can actually do in one turn, applied by the game loop to every mode.

Four limits, in order (Project 16, EB-2 and EB-3; Project 17, EC-5):

1. **Feasibility**: actions ``check_feasibility`` blocks are dropped.
2. **Diminishing returns**: an action of the same kind on the same target as one taken in
   the last ``REPEAT_WINDOW`` turns has its magnitude halved for each such use, so hammering
   one lever stops paying. A policy meant to keep running should say so with its duration.
3. **Political capital**: a turn's package costs the sum of its magnitudes (speeches cost a
   quarter, but whipping the legislature costs in full, and a measure forced through without a
   majority costs ``FORCE_CAPITAL`` times as much, EB-14). Above ``capital`` a turn, every
   measure is scaled down to fit.
4. **Deficit ceiling**: the deficit this package would leave (today's, plus what is
   already landing this turn, plus every tax and spending move in the package) may not pass
   ``DEFICIT_LIMIT``. Borrowing measures are scaled down to fit; tax rises and cuts in the
   same package make room for them.

The game is the only place these run: interpreters report what the player asked for.

Each limit that bites adds a plain-English note for the player. The game logs the actions
that were actually applied, so replay stays exact.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from pydantic import Field

from hog_sim.core.models import Model, PolicyAction
from hog_sim.core.state import WorldState
from hog_sim.policy.feasibility import (
    DEFICIT_LIMIT,
    FORCE_CAPITAL,
    LEGISLATURE_ID,
    Role,
    check_feasibility,
)
from hog_sim.world.propagation import FISCAL_NODE, fiscal_cost, scale

REPEAT_WINDOW = 4
REPEAT_DECAY = 0.5
CAPITAL_COST = {"communicate": 0.25, "do_nothing": 0.0}


class Constrained(Model):
    actions: list[PolicyAction]
    notes: list[str] = Field(default_factory=list)


def capital_cost(
    actions: Sequence[PolicyAction], state: WorldState | None = None, role: Role = "prime_minister"
) -> float:
    """Political capital the package uses. Whipping the legislature is hard graft, not a
    speech; given the state, measures forced through without a majority cost more."""
    forced = (
        [c.forced for c in check_feasibility(list(actions), state, role).checks]
        if state is not None
        else [False] * len(actions)
    )
    total = 0.0
    for action, force in zip(actions, forced, strict=True):
        unit = 1.0 if action.target == LEGISLATURE_ID else CAPITAL_COST.get(action.kind, 1.0)
        total += unit * abs(action.magnitude) * (FORCE_CAPITAL if force else 1.0)
    return total


def constrain(
    actions: Sequence[PolicyAction],
    state: WorldState,
    past_actions: Sequence[Sequence[PolicyAction]],
    role: Role = "prime_minister",
    capital: float = 1.5,
) -> Constrained:
    """Apply feasibility, diminishing returns and the capital budget to one turn's actions.

    ``past_actions`` holds the applied actions of earlier turns, oldest first.
    """
    report = check_feasibility(list(actions), state, role)
    notes = [
        f"Blocked: {c.action.kind} {c.action.target} ({'; '.join(c.blockers)})"
        for c in report.checks
        if not c.feasible
    ]

    recent = Counter(
        (a.kind, a.target)
        for turn in past_actions[-REPEAT_WINDOW:]
        for a in turn
        if a.kind != "do_nothing"
    )
    kept = []
    for action in report.feasible_actions:
        # Whipping the legislature is routine graft that has to be kept up, so it does not
        # wear out with repetition (EB-14).
        whip = action.kind == "communicate" and action.target == LEGISLATURE_ID
        uses = (
            recent[(action.kind, action.target)] if action.kind != "do_nothing" and not whip else 0
        )
        if uses:
            factor = REPEAT_DECAY**uses
            action = action.model_copy(update={"magnitude": action.magnitude * factor})
            notes.append(
                f"Diminishing returns: {action.kind} {action.target} was already used {uses} "
                f"time(s) in the last {REPEAT_WINDOW} turns, so it has {factor:.0%} of its effect"
            )
        kept.append(action)

    notes += [
        f"Forced through: {c.action.kind} {c.action.target} ({'; '.join(c.warnings)})"
        for c in report.checks
        if c.forced
    ]
    used = capital_cost(kept, state, role)
    if used > capital:
        share = capital / used
        kept = [a.model_copy(update={"magnitude": a.magnitude * share}) for a in kept]
        notes.append(
            f"Political capital: the package needed {used:.2f} against {capital:.2f} a turn, "
            f"so every measure was scaled to {share:.0%}"
        )
    kept, note = fit_deficit(kept, state)
    if note:
        notes.append(note)
    return Constrained(actions=kept, notes=notes)


def fit_deficit(
    actions: list[PolicyAction], state: WorldState, limit: float = DEFICIT_LIMIT
) -> tuple[list[PolicyAction], str | None]:
    """Scale the package's borrowing so the projected deficit stays within ``limit``."""
    deficit = state.indicators.get(FISCAL_NODE)
    if deficit is None:
        return actions, None
    step = scale(state, FISCAL_NODE)
    costs = [fiscal_cost(a, state) * step for a in actions]
    loosening = sum(c for c in costs if c > 0)
    landing = state.pending.get(state.turn, {}).get(FISCAL_NODE, 0.0)
    projected = deficit.value + landing + sum(costs)
    if loosening <= 0 or projected <= limit + 1e-9:
        return actions, None
    share = max(0.0, loosening - (projected - limit)) / loosening
    fitted = [
        a.model_copy(update={"magnitude": a.magnitude * share}) if c > 0 else a
        for a, c in zip(actions, costs, strict=True)
    ]
    fitted = [a for a, c in zip(fitted, costs, strict=True) if c <= 0 or share > 0]
    return fitted, (
        f"Deficit limit: the package would take the deficit to {projected:.1f}% of GDP, past "
        f"the {limit:g}% limit, so its borrowing was scaled to {share:.0%}"
    )
