"""What the player can actually do in one turn, applied by the game loop to every mode.

Three limits, in order (Project 16, EB-2 and EB-3):

1. **Feasibility**: actions ``check_feasibility`` blocks are dropped.
2. **Diminishing returns**: an action of the same kind on the same target as one taken in
   the last ``REPEAT_WINDOW`` turns has its magnitude halved for each such use, so hammering
   one lever stops paying. A policy meant to keep running should say so with its duration.
3. **Political capital**: a turn's package costs the sum of its magnitudes (speeches cost a
   quarter). Above ``capital`` a turn, every measure is scaled down to fit.

Each limit that bites adds a plain-English note for the player. The game logs the actions
that were actually applied, so replay stays exact.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from pydantic import Field

from hog_sim.core.models import Model, PolicyAction
from hog_sim.core.state import WorldState
from hog_sim.policy.feasibility import Role, check_feasibility

REPEAT_WINDOW = 4
REPEAT_DECAY = 0.5
CAPITAL_COST = {"communicate": 0.25, "do_nothing": 0.0}


class Constrained(Model):
    actions: list[PolicyAction]
    notes: list[str] = Field(default_factory=list)


def capital_cost(actions: Sequence[PolicyAction]) -> float:
    return sum(CAPITAL_COST.get(a.kind, 1.0) * abs(a.magnitude) for a in actions)


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
        uses = recent[(action.kind, action.target)] if action.kind != "do_nothing" else 0
        if uses:
            factor = REPEAT_DECAY**uses
            action = action.model_copy(update={"magnitude": action.magnitude * factor})
            notes.append(
                f"Diminishing returns: {action.kind} {action.target} was already used {uses} "
                f"time(s) in the last {REPEAT_WINDOW} turns, so it has {factor:.0%} of its effect"
            )
        kept.append(action)

    used = capital_cost(kept)
    if used > capital:
        share = capital / used
        kept = [a.model_copy(update={"magnitude": a.magnitude * share}) for a in kept]
        notes.append(
            f"Political capital: the package needed {used:.2f} against {capital:.2f} a turn, "
            f"so every measure was scaled to {share:.0%}"
        )
    return Constrained(actions=kept, notes=notes)
