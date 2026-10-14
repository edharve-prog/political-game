"""What the player can actually do in one turn, applied by the game loop to every mode.

Four limits, in order (Project 16, EB-2 and EB-3; Project 17, EC-5), after cuts to wage
subsidies are held to the schemes actually running (``fit_job_cuts``):

1. **Feasibility**: actions ``check_feasibility`` blocks are dropped.
2. **Diminishing returns**: an action of the same kind on the same target, in the same
   direction, as one taken in the last ``REPEAT_WINDOW`` turns has its magnitude halved for
   each such use, so hammering one lever stops paying. A policy meant to keep running should
   say so with its duration. Going the other way on a measure still running is a U-turn, not
   a repeat: it keeps its full size but costs ``UTURN_CAPITAL`` more capital, for the
   credibility it spends (E12).
3. **Political capital**: a turn's package costs the sum of its magnitudes (speeches cost a
   quarter, but whipping the legislature costs in full, and a measure forced through without a
   majority costs ``FORCE_CAPITAL`` times as much, EB-14), times each measure's lifetime
   (``lifetime_factor``): every turn a measure runs after its first adds ``UPKEEP`` of its
   first turn's cost, charged when it is decided, so a longer measure is never free (E1,
   O1). Above ``capital`` a turn, every measure is scaled down to fit.
4. **Deficit ceiling**: the deficit this package would leave (today's, plus what is
   already landing this turn, plus every tax and spending move in the package) may not pass
   ``DEFICIT_LIMIT``, and once past it the package may not add to it. Borrowing measures
   are scaled down to fit; tax rises and cuts in the same package make room for them for
   as long as they run, so a funded package goes ahead even above the limit (feasibility
   only blocks borrowing there when nothing in the package pays for it to the end).

Before any of them, durations are held to ``MAX_TURNS`` for their kind: a speech is news
for a few turns, not a policy that runs for two years.

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
from hog_sim.world.propagation import (
    FISCAL_NODE,
    JOB_PROTECTION,
    cuts_jobs_cover,
    fiscal_cost,
    fiscal_turns,
    in_flight,
    job_cover_left,
    scale,
)

REPEAT_WINDOW = 4
REPEAT_DECAY = 0.5
CAPITAL_COST = {"communicate": 0.25, "do_nothing": 0.0}
# Upkeep (E1, O1): each turn a measure runs after its first costs this share of its first
# turn's capital. Capital does not carry over between turns yet, so the whole lifetime is
# charged on the turn the measure is decided; a 12-turn measure costs 2.1 times a one-off.
UPKEEP = 0.1
# Longest a measure of each kind may run, in turns (E1). A speech is news, not a programme.
MAX_TURNS = {"communicate": 3}
# A U-turn (E12) spends credibility: going the other way on a measure still running costs
# this much more capital per unit than the measure alone would.
UTURN_CAPITAL = 1.0


class Constrained(Model):
    actions: list[PolicyAction]
    notes: list[str] = Field(default_factory=list)


def lifetime_factor(action: PolicyAction) -> float:
    """How many first turns' worth of capital a measure costs over its life (``UPKEEP``)."""
    return 1 + UPKEEP * (action.duration_turns - 1)


def _direction(action: PolicyAction) -> float:
    """Which way a measure goes: the tax or spending's own direction, else the magnitude's."""
    return action.instrument_sign


def _running(past_actions: Sequence[Sequence[PolicyAction]]) -> list[PolicyAction]:
    """Earlier turns' actions still in force this turn (``past_actions`` ends last turn)."""
    now = len(past_actions)
    return [
        a
        for i, turn in enumerate(past_actions)
        for a in turn
        if a.kind != "do_nothing" and i + a.duration_turns > now
    ]


def is_u_turn(action: PolicyAction, past_actions: Sequence[Sequence[PolicyAction]]) -> bool:
    """Whether ``action`` goes the other way on a measure of the same kind and target that
    is still running (E12)."""
    way = _direction(action)
    return bool(way) and any(
        a.kind == action.kind and a.target == action.target and _direction(a) == -way
        for a in _running(past_actions)
    )


def capital_cost(
    actions: Sequence[PolicyAction],
    state: WorldState | None = None,
    role: Role = "prime_minister",
    past_actions: Sequence[Sequence[PolicyAction]] = (),
) -> float:
    """Political capital the package uses over the life of its measures. Whipping the
    legislature is hard graft, not a speech; given the state, measures forced through
    without a majority cost more; given earlier turns' actions, so do U-turns."""
    forced = (
        [c.forced for c in check_feasibility(list(actions), state, role).checks]
        if state is not None
        else [False] * len(actions)
    )
    total = 0.0
    for action, force in zip(actions, forced, strict=True):
        unit = 1.0 if action.target == LEGISLATURE_ID else CAPITAL_COST.get(action.kind, 1.0)
        if is_u_turn(action, past_actions):
            unit += UTURN_CAPITAL
        unit *= FORCE_CAPITAL if force else 1.0
        total += unit * abs(action.magnitude) * lifetime_factor(action)
    return total


def fit_durations(actions: Sequence[PolicyAction]) -> tuple[list[PolicyAction], list[str]]:
    """Hold each measure to the longest its kind may run (``MAX_TURNS``)."""
    kept, notes = [], []
    for action in actions:
        cap = MAX_TURNS.get(action.kind)
        if cap is not None and action.duration_turns > cap:
            notes.append(
                f"Duration: a {action.kind} on {action.target} can run at most {cap} turns, "
                f"not {action.duration_turns}"
            )
            action = action.model_copy(update={"duration_turns": cap})
        kept.append(action)
    return kept, notes


class LifetimeCost(Model):
    """What one measure costs over its whole life, as the review screen shows it (O1)."""

    capital: float
    deficit: float = Field(0.0, description="Points of GDP it adds to the deficit while paid")
    deficit_turns: int = Field(0, description="Turns the deficit cost is paid for")

    @property
    def deficit_turn_pts(self) -> float:
        """The deficit cost summed over the turns it is paid, in points of GDP x turns."""
        return self.deficit * self.deficit_turns


def lifetime_cost(action: PolicyAction, state: WorldState) -> LifetimeCost:
    """Capital and deficit ``action`` costs over its life, as decided (before this turn's
    capital budget scales the package)."""
    deficit, turns = 0.0, 0
    if FISCAL_NODE in state.indicators:
        deficit = fiscal_cost(action, state) * scale(state, FISCAL_NODE)
        turns = fiscal_turns(action, state)
    return LifetimeCost(
        capital=capital_cost([action], state),
        deficit=deficit,
        deficit_turns=turns if deficit else 0,
    )


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
    actions, notes = fit_durations(actions)
    cut, cut_notes = fit_job_cuts(actions, state)
    actions, notes = cut, notes + cut_notes
    report = check_feasibility(list(actions), state, role)
    notes += [
        f"Blocked: {c.action.kind} {c.action.target} ({'; '.join(c.blockers)})"
        for c in report.checks
        if not c.feasible
    ]

    recent = Counter(
        (a.kind, a.target, _direction(a))
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
            recent[(action.kind, action.target, _direction(action))]
            if action.kind != "do_nothing" and not whip
            else 0
        )
        if is_u_turn(action, past_actions):
            notes.append(
                f"U-turn: {action.kind} {action.target} reverses a measure still running, which "
                "costs credibility and extra political capital"
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
    used = capital_cost(kept, state, role, past_actions)
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


def fit_job_cuts(
    actions: Sequence[PolicyAction], state: WorldState
) -> tuple[list[PolicyAction], list[str]]:
    """Hold cuts to wage subsidies to the schemes actually running: a cut with no scheme to
    cut is dropped, and a bigger or longer cut than the cover in force is trimmed to it."""
    share, turns = job_cover_left(state)
    kept, notes = [], []
    for action in actions:
        if not cuts_jobs_cover(action):
            kept.append(action)
            continue
        size = min(abs(action.magnitude), share / JOB_PROTECTION)
        if size <= 1e-9 or not turns:
            notes.append(
                f"Blocked: {action.kind} {action.target} (no wage subsidy scheme is running to cut)"
            )
            continue
        share -= size * JOB_PROTECTION
        trimmed = action.model_copy(
            update={
                "magnitude": size if action.magnitude > 0 else -size,
                "duration_turns": min(action.duration_turns, turns),
            }
        )
        if trimmed != action:
            notes.append(
                f"Wage subsidies: only the scheme running can be cut, so the cut was held to "
                f"magnitude {size:.2f} for {trimmed.duration_turns} turn(s)"
            )
        kept.append(trimmed)
    return kept, notes


def fit_deficit(
    actions: list[PolicyAction], state: WorldState, limit: float = DEFICIT_LIMIT
) -> tuple[list[PolicyAction], str | None]:
    """Scale the package's borrowing so the projected deficit stays within ``limit``, or,
    when it is already past the limit, so the package adds nothing to it.

    The deficit is projected for every turn the package's costs run, not only this one, so a
    tax rise pays for borrowing only while it lasts (E5): a one-turn rise cannot fund a cut
    that runs for two years."""
    deficit = state.indicators.get(FISCAL_NODE)
    if deficit is None:
        return actions, None
    step = scale(state, FISCAL_NODE)
    costs = [fiscal_cost(a, state) * step for a in actions]
    paid = [fiscal_turns(a, state) for a in actions]
    if not any(c > 0 for c in costs):
        return actions, None
    base = deficit.value + in_flight(state, FISCAL_NODE)
    ceiling = max(limit, base)
    share, worst = 1.0, base
    for t in range(max(paid)):
        loosening = sum(c for c, n in zip(costs, paid, strict=True) if c > 0 and t < n)
        saving = sum(c for c, n in zip(costs, paid, strict=True) if c < 0 and t < n)
        worst = max(worst, base + loosening + saving)
        if loosening > 0:
            share = min(share, max(0.0, ceiling - base - saving) / loosening)
    if share >= 1 - 1e-9:
        return actions, None
    fitted = [
        a.model_copy(update={"magnitude": a.magnitude * share}) if c > 0 else a
        for a, c in zip(actions, costs, strict=True)
    ]
    fitted = [a for a, c in zip(fitted, costs, strict=True) if c <= 0 or share > 0]
    if ceiling > limit:
        return fitted, (
            f"Deficit limit: the deficit is already {base:.1f}% of GDP, past "
            f"the {limit:g}% limit, so the package's borrowing was scaled to {share:.0%}, as "
            "far as its tax rises and cuts pay for while they last"
        )
    return fitted, (
        f"Deficit limit: the package would take the deficit to {worst:.1f}% of GDP, past "
        f"the {limit:g}% limit, so its borrowing was scaled to {share:.0%}"
    )
