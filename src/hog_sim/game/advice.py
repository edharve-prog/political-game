"""Advisers with numbers (review of 13 October, D8).

Before the advisers' options are shown, each one is read the way a response would be
(``Game.propose``: the interpreter, then this turn's limits) and played to the election
with the decision preview (D2). The player sees what each would cost in political capital,
what it would add to the deficit over its life, and what it would do to the vote against
doing nothing. An option the limits would block outright (something was asked for and
nothing survives) is not offered; one that can't be read is shown without numbers.

Nothing here changes the game: proposals are only made, never committed.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from hog_sim.core.models import Model, PolicyAction
from hog_sim.game.interfaces import NeedsClarification
from hog_sim.game.loop import Game, Proposal
from hog_sim.game.preview import preview
from hog_sim.policy.limits import capital_cost, lifetime_cost


class Costing(Model):
    """What an option does if chosen as it stands."""

    actions: list[PolicyAction]
    capital: float = Field(description="Political capital it uses this turn, over its life")
    deficit: float = Field(
        description="Points of GDP it adds to the deficit, summed over the turns paid"
    )
    vote: float = Field(description="Vote share at the election against doing nothing")
    seats: int = Field(description="Seats at the election against doing nothing")
    notes: list[str] = Field(default_factory=list, description="What the limits changed")


class CostedOption(Model):
    option: str
    trade_off: str = ""
    costing: Costing | None = None
    blocked: str | None = Field(None, description="Why the limits would block it outright")


def _costing(game: Game, proposal: Proposal) -> Costing:
    state, config = game.state, game.config
    past = [r.actions for r in game.history]
    result = preview(
        game.start,
        state,
        game.scenario,
        proposal.actions,
        config,
        proposal.pledges,
        proposal.delivery,
        proposal.sacked,
        config.role,
    )
    return Costing(
        actions=proposal.actions,
        capital=capital_cost(proposal.actions, state, config.role, past),
        deficit=sum(
            lifetime_cost(a, state).deficit_turn_pts
            for a in proposal.actions
            if a.kind != "do_nothing"
        ),
        vote=result.package.vote - result.nothing.vote,
        seats=result.package.seats - result.nothing.seats,
        notes=proposal.notes,
    )


def cost_option(game: Game, option: str, trade_off: str = "") -> CostedOption:
    """``option`` read, limited and previewed for this turn."""
    try:
        proposal = game.propose(option)
    except NeedsClarification:
        return CostedOption(option=option, trade_off=trade_off)
    real = [a for a in proposal.requested if a.kind != "do_nothing"]
    if real and not proposal.actions:
        blockers = [n for n in proposal.notes if n.startswith("Blocked")]
        reason = "; ".join(blockers) or "this turn's limits leave nothing of it"
        return CostedOption(option=option, trade_off=trade_off, blocked=reason)
    if not real:
        return CostedOption(option=option, trade_off=trade_off)
    return CostedOption(option=option, trade_off=trade_off, costing=_costing(game, proposal))


def cost_options(game: Game, options: Sequence[tuple[str, str]]) -> list[CostedOption]:
    """Each (option, trade-off) costed for this turn, in order."""
    return [cost_option(game, option, trade_off) for option, trade_off in options]


def costing_line(costing: Costing) -> str:
    """One line of numbers for the advice screen."""
    deficit = (
        f"{costing.deficit:+.1f}% of GDP on the deficit over its life"
        if abs(costing.deficit) >= 0.05
        else "no deficit cost"
    )
    return (
        f"Numbers: {costing.capital:.2f} political capital, {deficit}, "
        f"{costing.vote * 100:+.2f} pts at the election ({costing.seats:+d} seats) "
        "against doing nothing"
    )
