"""EB-14: whipping rebuilds the Commons, and a measure short of a majority can be forced
through at a cost instead of being blocked."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction
from hog_sim.game.loop import resolve
from hog_sim.game.stubs import CannedScenarios, KeywordInterpreter
from hog_sim.policy.feasibility import (
    FORCE_CAPITAL,
    FORCE_FLOOR,
    LEGISLATURE_ID,
    check_feasibility,
    forcing_shocks,
)
from hog_sim.policy.limits import capital_cost, constrain
from hog_sim.world.seed.toy import toy_world

QUIET = Outcome(narrative="A quiet month.", probability=1.0)
SPEND = PolicyAction(kind="spend", target="sector:public", magnitude=0.5)
WHIP = PolicyAction(kind="communicate", target=LEGISLATURE_ID, magnitude=1.0)


def world_with_support(support: float):
    world = toy_world()
    world.institutions[LEGISLATURE_ID].support = support
    return world


def support(state) -> float:
    return state.institutions[LEGISLATURE_ID].support


def test_short_of_a_majority_is_forced_through() -> None:
    [check] = check_feasibility([SPEND], world_with_support(0.45)).checks
    assert check.feasible and check.forced
    assert check.resistance == 1.0
    assert "forced through" in check.warnings[0]


def test_far_short_of_a_majority_is_still_blocked() -> None:
    [check] = check_feasibility([SPEND], world_with_support(FORCE_FLOOR - 0.01)).checks
    assert not check.feasible and not check.forced


def test_a_majority_needs_no_forcing() -> None:
    world = world_with_support(0.6)
    [check] = check_feasibility([SPEND], world).checks
    assert check.feasible and not check.forced
    assert forcing_shocks([SPEND], world) == []


def test_forcing_costs_capital_and_support() -> None:
    world = world_with_support(0.45)
    assert capital_cost([SPEND], world) == pytest.approx(FORCE_CAPITAL * capital_cost([SPEND]))
    limited = constrain([SPEND], world, [])
    assert limited.actions and any(n.startswith("Forced through") for n in limited.notes)
    after = resolve(
        world,
        world,
        CannedScenarios().next_scenario(world, []),
        [SPEND],
        QUIET,
        GameConfig(k_draws=10),
    )
    without = resolve(
        world, world, CannedScenarios().next_scenario(world, []), [], QUIET, GameConfig(k_draws=10)
    )
    assert support(after) < support(without) - 0.03


def test_whipping_rebuilds_support() -> None:
    world = world_with_support(0.45)
    scenario = CannedScenarios().next_scenario(world, [])
    whipped = resolve(world, world, scenario, [WHIP], QUIET, GameConfig(k_draws=10))
    idle = resolve(world, world, scenario, [], QUIET, GameConfig(k_draws=10))
    assert support(whipped) > support(idle) + 0.05
    # A whip is real work, not a speech: it costs full political capital.
    assert capital_cost([WHIP]) == 1.0


def test_keyword_interpreter_whips_the_backbenchers() -> None:
    world = toy_world()
    scenario = CannedScenarios().next_scenario(world, [])
    [action] = KeywordInterpreter().interpret("Whip the backbenchers into line", world, scenario)
    assert (action.kind, action.target) == ("communicate", LEGISLATURE_ID)


def test_whipping_does_not_wear_out() -> None:
    limited = constrain([WHIP], world_with_support(0.45), [[WHIP], [WHIP], [WHIP]])
    assert limited.actions[0].magnitude == WHIP.magnitude
