"""EB-9: the debt stock and its interest bill."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, Scenario, Shock
from hog_sim.game.loop import resolve
from hog_sim.world.debt import (
    REPRICE_HALF_LIFE,
    debt_rate,
    interest_bill,
    interest_shocks,
    step_debt,
)
from hog_sim.world.seed.toy import toy_world

QUIET = Scenario(title="Quiet", briefing="Nothing happens.", affected_nodes=[], urgency=0)
NOTHING = Outcome(narrative="Nothing happened.", probability=1.0)
UK = "country:uk"


def play(start, shocks, turns):
    state = start
    for turn in range(turns):
        scenario = QUIET.model_copy(update={"shocks": shocks if turn == 0 else []})
        state = resolve(start, state, scenario, [], NOTHING, GameConfig())
    return state


def test_starting_world_pays_no_extra_interest() -> None:
    world = toy_world()
    assert interest_bill(world) == pytest.approx(4.0)
    assert interest_shocks(world, world) == []
    after = step_debt(world)
    # Deficit 4.5% against nominal growth 4.5% holds debt at 100% of GDP.
    assert after.countries[UK].debt_pct == pytest.approx(100.0)


def test_borrowing_compounds() -> None:
    world = toy_world()
    world.indicators["indicator:deficit"].value = 10.0
    after = step_debt(world)
    assert after.countries[UK].debt_pct == pytest.approx(100 + (10 - 4.5) / 12)


def test_rate_rise_reprices_debt_with_a_half_life() -> None:
    state = toy_world()
    state.indicators["indicator:interest_rate"].value = 5.0
    for _ in range(REPRICE_HALF_LIFE):
        state = step_debt(state)
    assert debt_rate(state) == pytest.approx(4.5, abs=0.02)


def test_higher_rates_raise_the_deficit() -> None:
    start = toy_world()
    hike = [Shock(node="indicator:interest_rate", delta=2.0, duration_turns=24, hold=True)]
    with_debt = play(start, hike, 24)
    no_debt_start = toy_world()
    no_debt_start.countries[UK].debt_pct = 0.0
    without = play(no_debt_start, hike, 24)
    assert with_debt.countries[UK].debt_rate_pct > 4.5
    gap = (
        with_debt.indicators["indicator:deficit"].value
        - without.indicators["indicator:deficit"].value
    )
    assert 0.2 < gap < 2.0


def test_old_saves_without_debt_are_unchanged() -> None:
    world = toy_world()
    world.countries[UK].debt_pct = 0.0
    assert step_debt(world) == world
    assert interest_shocks(world, world) == []
