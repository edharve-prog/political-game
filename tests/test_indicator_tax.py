"""A tax or spending aimed at an indicator: the budget follows the instrument (raise or cut),
not the way the player wants the indicator to move."""

import pytest

from hog_sim.core.models import Pledge, PolicyAction
from hog_sim.llm.interpreter import Interpretation, check_interpretation
from hog_sim.llm.summary import summarise_state
from hog_sim.policy.pledges import breaks
from hog_sim.world.propagation import actions_to_shocks, fiscal_cost, unfunded_tax_cut
from hog_sim.world.seed.toy import toy_world

HOUSES = "indicator:house_prices"


def tax(magnitude, instrument=None, target=HOUSES):
    return PolicyAction(kind="tax", target=target, magnitude=magnitude, instrument=instrument)


def test_cutting_stamp_duty_to_lift_prices_costs_money() -> None:
    world = toy_world()
    cut = tax(0.5, "cut")  # prices up, tax down
    rise = tax(-0.5, "raise")  # prices down, tax up
    assert fiscal_cost(cut, world) > 0
    assert fiscal_cost(rise, world) < 0
    assert fiscal_cost(cut, world) == pytest.approx(-fiscal_cost(rise, world))
    # The engine still pushes the indicator the way the player wants it.
    push = {s.node: s.delta for s in actions_to_shocks([cut], world)}
    assert push[HOUSES] > 0 and push["indicator:deficit"] > 0


def test_unset_instrument_reads_as_a_rise() -> None:
    """Saved games from before the field existed replay as they were played."""
    world = toy_world()
    assert fiscal_cost(tax(0.5), world) == fiscal_cost(tax(0.5, "raise"), world) < 0
    subsidy = PolicyAction(kind="spend", target="indicator:energy_prices", magnitude=-0.5)
    assert fiscal_cost(subsidy, world) > 0


def test_a_big_indicator_tax_cut_counts_as_unfunded() -> None:
    world = toy_world()
    assert unfunded_tax_cut([tax(1.0, "cut")], world) > 0
    assert unfunded_tax_cut([tax(-1.0, "raise")], world) == 0


def test_pledges_follow_the_instrument() -> None:
    no_rises = Pledge(text="no tax rises", kind="tax", direction="up")
    assert breaks(no_rises, tax(-0.5, "raise"))  # a rise that cools prices
    assert not breaks(no_rises, tax(0.5, "cut"))  # a cut that lifts prices
    assert breaks(no_rises, tax(0.5, target="sector:finance"))


def test_interpreter_must_say_which_way_an_indicator_tax_goes() -> None:
    summary = summarise_state(toy_world())
    problems = check_interpretation(Interpretation(actions=[tax(0.5)]), summary)
    assert any("instrument" in p for p in problems)
    assert check_interpretation(Interpretation(actions=[tax(0.5, "cut")]), summary) == []
