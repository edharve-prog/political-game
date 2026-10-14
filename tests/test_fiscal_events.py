"""Fiscal events (review of 13 October, O6): the Budget is the turn for taxes and spending,
and an independent scorecard costs each measure."""

import pytest

from hog_sim.core.models import PolicyAction
from hog_sim.game.stubs import KeywordInterpreter
from hog_sim.policy.fiscal_events import (
    OFF_BUDGET_SURCHARGE,
    is_budget,
    next_budget,
    scorecard,
    scorecard_lines,
)
from hog_sim.policy.limits import capital_cost, constrain, lifetime_cost
from hog_sim.ui import cli
from hog_sim.world.seed.toy import toy_world

BUDGET = next_budget(0)


def act(kind, target, magnitude=0.5, turns=1):
    return PolicyAction(kind=kind, target=target, magnitude=magnitude, duration_turns=turns)


def on(turn: int):
    world = toy_world()
    world.turn = turn
    return world


def test_the_budget_falls_on_calendar_turns() -> None:
    assert BUDGET == 3 and is_budget(3) and is_budget(15)
    assert not is_budget(0) and not is_budget(4)
    assert next_budget(3) == 15


def test_taxes_and_spending_cost_more_outside_the_budget() -> None:
    spend = act("spend", "sector:public", 1.0)
    assert capital_cost([spend], on(BUDGET)) == pytest.approx(1.0)
    assert capital_cost([spend], on(BUDGET + 1)) == pytest.approx(1.0 + OFF_BUDGET_SURCHARGE)
    # Other levers are not fiscal events.
    law = act("legislate", "group:pensioners", 1.0)
    assert capital_cost([law], on(BUDGET)) == capital_cost([law], on(BUDGET + 1))


def test_the_surcharge_scales_with_lifetime() -> None:
    long = act("tax", "sector:finance", 1.0, turns=12)
    gap = capital_cost([long], on(1)) - capital_cost([long], on(BUDGET))
    assert gap == pytest.approx(OFF_BUDGET_SURCHARGE * (1 + 0.1 * 11))


def test_the_limits_say_when_the_next_budget_is() -> None:
    limited = constrain([act("spend", "sector:public")], on(1), [])
    assert any("Outside the Budget" in n and "turn 3" in n for n in limited.notes)
    budget = constrain([act("spend", "sector:public")], on(BUDGET), [])
    assert not any("Outside the Budget" in n for n in budget.notes)


def test_a_big_package_goes_further_on_budget_day() -> None:
    package = [act("spend", "sector:public", 1.0), act("tax", "sector:finance", 0.5)]
    off = constrain(package, on(1), [])
    on_budget = constrain(package, on(BUDGET), [])
    assert sum(a.magnitude for a in off.actions) < sum(a.magnitude for a in on_budget.actions)


def test_the_scorecard_costs_each_fiscal_measure() -> None:
    world = on(BUDGET)
    package = [
        act("spend", "sector:public", 0.5, turns=6),
        act("tax", "sector:finance", 0.5, turns=6),
        act("legislate", "group:pensioners", 0.5),
    ]
    card = scorecard(package, world)
    assert [line.action.kind for line in card] == ["spend", "tax"]
    for line in card:
        cost = lifetime_cost(line.action, world)
        assert line.per_turn == pytest.approx(cost.deficit)
        assert line.turns == cost.deficit_turns
    assert card[0].per_turn > 0 > card[1].per_turn
    lines = scorecard_lines(package, world)
    assert lines[0].startswith("Independent scorecard")
    assert lines[-1].startswith("  Package:")


def test_the_scorecard_is_empty_without_fiscal_measures() -> None:
    assert scorecard_lines([act("legislate", "group:pensioners")], toy_world()) == []


def test_the_review_screen_shows_the_scorecard() -> None:
    from hog_sim.game.loop import Proposal

    world = on(BUDGET)
    actions = KeywordInterpreter().interpret("Raise taxes on the banks", world, None)
    proposal = Proposal(response="", requested=actions, actions=actions)
    assert "Independent scorecard" in cli._proposal_text(proposal, world)
