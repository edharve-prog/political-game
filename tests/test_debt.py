"""EB-9: the debt stock and its interest bill."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, Scenario, Shock
from hog_sim.game.loop import resolve
from hog_sim.world.debt import (
    REPRICE_HALF_LIFE,
    RESERVES_SHARE,
    debt_line,
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
    state.indicators["indicator:gilt_yield"].value = 5.0
    state.indicators["indicator:interest_rate"].value = 5.0
    for _ in range(REPRICE_HALF_LIFE):
        state = step_debt(state)
    assert debt_rate(state) == pytest.approx(4.5, abs=0.02)


def test_bank_rate_alone_reprices_only_the_reserves() -> None:
    state = toy_world()
    state.indicators["indicator:interest_rate"].value = 5.0
    for _ in range(REPRICE_HALF_LIFE):
        state = step_debt(state)
    assert debt_rate(state) == pytest.approx(4.0 + 0.5 * RESERVES_SHARE, abs=0.02)


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
    # Up to 2.5: since Bank Rate works through the output gap (FR-27), a held hike opens a
    # slump that grows the debt, and inflation falls less, so the Bank keeps rates nearer 9%
    # than 7% and the bill on a larger debt is bigger (2.2 points, against 1.6 before).
    assert 0.2 < gap < 2.5


def test_old_saves_without_debt_are_unchanged() -> None:
    world = toy_world()
    world.countries[UK].debt_pct = 0.0
    assert step_debt(world) == world
    assert interest_shocks(world, world) == []


def test_tracked_debt_grows_again_from_zero() -> None:
    world = toy_world()
    world.countries[UK].debt_pct = 0.0
    world.countries[UK].debt_tracked = True
    world.indicators["indicator:deficit"].value = 6.0
    after = step_debt(world)
    assert after.countries[UK].debt_pct == pytest.approx(0.5)
    assert debt_line(after).startswith("Public debt: 0% of GDP")


def test_debt_paid_down_to_zero_stays_tracked() -> None:
    world = toy_world()
    world.countries[UK].debt_pct = 0.05
    world.indicators["indicator:deficit"].value = -6.0  # a surplus pays it off
    paid = step_debt(world)
    assert paid.countries[UK].debt_pct == 0.0
    assert paid.countries[UK].tracks_debt
    paid.indicators["indicator:deficit"].value = 6.0
    assert step_debt(paid).countries[UK].debt_pct == pytest.approx(0.5)
