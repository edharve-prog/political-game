"""EB-8: the Commons follows the polls, and resistance makes rebellions likelier."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction, Scenario
from hog_sim.forecasting.scoring import BASE_RATES, base_rate
from hog_sim.game.loop import COMMONS_POLLS, INSTITUTION_SETTLE, resolve
from hog_sim.policy.feasibility import LEGISLATURE_ID, check_feasibility
from hog_sim.population.popularity import national_approval
from hog_sim.world.seed.toy import toy_world

QUIET = Scenario(title="Quiet", briefing="Nothing happens.", affected_nodes=[], urgency=0)
NOTHING = Outcome(narrative="Nothing happened.", probability=1.0)


def test_commons_support_follows_the_polls() -> None:
    start = toy_world()
    slump = start.snapshot()
    for group in slump.groups.values():
        group.approval -= 0.1
    drop = national_approval(start) - national_approval(slump)
    new = resolve(start, slump, QUIET, [], NOTHING, GameConfig())
    moved = new.institutions[LEGISLATURE_ID].support - start.institutions[LEGISLATURE_ID].support
    assert moved == pytest.approx(-COMMONS_POLLS * drop * INSTITUTION_SETTLE, abs=1e-3)
    # Other institutions are not swayed by polls.
    for iid, inst in new.institutions.items():
        if iid != LEGISLATURE_ID:
            assert inst.support == pytest.approx(start.institutions[iid].support, abs=1e-9)


def test_quiet_start_leaves_the_commons_alone() -> None:
    start = toy_world()
    new = resolve(start, start, QUIET, [], NOTHING, GameConfig())
    assert new.institutions[LEGISLATURE_ID].support == pytest.approx(
        start.institutions[LEGISLATURE_ID].support
    )


def test_resistance_raises_rebellion_odds() -> None:
    state = toy_world()
    spend = PolicyAction(kind="spend", target="sector:public", magnitude=1.0)
    resistance = check_feasibility([spend], state).resistance
    assert resistance > 0
    assert (
        check_feasibility(
            [PolicyAction(kind="do_nothing", target="sector:public", magnitude=0.0)], state
        ).resistance
        == 0
    )
    rebellion = base_rate(["backbench_rebellion"], resistance=resistance)
    assert rebellion == pytest.approx(BASE_RATES["backbench_rebellion"] * (1 + 2 * resistance))
    assert base_rate(["strike"], resistance=resistance) == BASE_RATES["strike"]
    assert base_rate(["backbench_rebellion"], resistance=1.0) <= 1.0
