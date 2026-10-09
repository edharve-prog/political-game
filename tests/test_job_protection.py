"""Wage subsidies keep people in work through a slump (Project 11, CA-4): the 2020 furlough."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction, Scenario, Shock
from hog_sim.core.state import WorldState
from hog_sim.game.loop import resolve
from hog_sim.game.stubs import KeywordInterpreter
from hog_sim.policy.feasibility import compatibility_problem
from hog_sim.world.propagation import (
    FISCAL_NODE,
    JOB_NODE,
    JOB_PROTECTION,
    actions_to_shocks,
    job_protection,
    simulate,
)
from hog_sim.world.seed.toy import toy_world

NOTHING = Outcome(narrative="Nothing else happens", probability=1)
SLUMP = [Shock(node="country:uk", delta=-5)]


def furlough(magnitude: float = -1.0, turns: int = 6) -> PolicyAction:
    return PolicyAction(kind="spend", target=JOB_NODE, magnitude=magnitude, duration_turns=turns)


def test_it_costs_money_but_does_not_push_unemployment_down() -> None:
    shocks = actions_to_shocks([furlough()], toy_world())
    assert [s.node for s in shocks] == [FISCAL_NODE]
    assert shocks[0].delta > 0


def test_it_covers_its_share_for_as_long_as_it_runs() -> None:
    state = toy_world()
    cover = job_protection(state, [furlough(-0.5, turns=3)])
    assert cover == {0: pytest.approx(0.5 * JOB_PROTECTION), 1: cover[0], 2: cover[0]}


def test_it_holds_back_job_losses_in_a_slump() -> None:
    state = toy_world()
    bare = simulate(state, SLUMP, 8)[JOB_NODE]
    cover = job_protection(state, [furlough(turns=12)])
    held = simulate(state, SLUMP, 8, protection=cover)[JOB_NODE]
    assert max(bare) > 0
    assert max(held) == pytest.approx((1 - JOB_PROTECTION) * max(bare), rel=0.2)


def test_it_does_nothing_when_unemployment_is_falling() -> None:
    state = toy_world()
    boom = [Shock(node="country:uk", delta=3)]
    cover = job_protection(state, [furlough()])
    assert simulate(state, boom, 8, protection=cover) == simulate(state, boom, 8)


def test_a_scheme_carries_into_later_turns() -> None:
    start = toy_world()
    quiet = Scenario(title="Quiet", briefing="", affected_nodes=[], urgency=0)
    state = resolve(start, start, quiet, [furlough(turns=6)], NOTHING, GameConfig())
    assert set(state.job_protection) == {1, 2, 3, 4, 5}
    crash = quiet.model_copy(update={"shocks": SLUMP})
    unprotected = start.model_copy(update={"turn": 1})
    protected = resolve(start, state, crash, [], NOTHING, GameConfig())
    bare = resolve(start, unprotected, crash, [], NOTHING, GameConfig())

    def rise(s: WorldState) -> float:
        return sum(s.pending[t].get(JOB_NODE, 0.0) for t in range(2, 6))

    assert 0 < rise(protected) < rise(bare)
    assert WorldState.from_json(protected.to_json()).job_protection == protected.job_protection


def test_it_is_a_feasible_direct_policy() -> None:
    state = toy_world()
    assert compatibility_problem(furlough(), state) is None


def test_the_keyword_interpreter_knows_furlough() -> None:
    state = toy_world()
    quiet = Scenario(title="Quiet", briefing="", affected_nodes=[], urgency=0)
    [action] = KeywordInterpreter().interpret("Launch a furlough scheme", state, quiet)
    assert (action.kind, action.target) == ("spend", JOB_NODE)
    assert action.magnitude < 0
