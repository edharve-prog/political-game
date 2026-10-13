"""Wage subsidies keep people in work through a slump (Project 11, CA-4): the 2020 furlough."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction, Scenario, Shock
from hog_sim.core.state import WorldState
from hog_sim.game.loop import resolve
from hog_sim.game.stubs import KeywordInterpreter
from hog_sim.policy.feasibility import compatibility_problem
from hog_sim.policy.limits import constrain
from hog_sim.world.propagation import (
    FISCAL_NODE,
    JOB_NODE,
    JOB_PROTECTION,
    actions_to_shocks,
    fiscal_cost,
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
        before = s.indicators[JOB_NODE].value
        for _ in range(4):
            s = resolve(start, s, quiet, [], NOTHING, GameConfig())
        return s.indicators[JOB_NODE].value - before

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


QUIET = Scenario(title="Quiet", briefing="", affected_nodes=[], urgency=0)


def cut(magnitude: float = -1.0, turns: int = 6) -> PolicyAction:
    return furlough(magnitude, turns).model_copy(update={"instrument": "cut"})


def running(turns: int = 6) -> WorldState:
    """A world one turn into a full-size scheme of ``turns`` turns."""
    start = toy_world()
    return resolve(start, start, QUIET, [furlough(turns=turns)], NOTHING, GameConfig())


@pytest.mark.parametrize("magnitude", [1.0, -1.0])
def test_either_sign_of_magnitude_follows_the_instrument(magnitude) -> None:
    """Review finding 1: the spending's direction is the instrument's, not the magnitude's."""
    state = toy_world()
    raise_ = furlough(magnitude)
    assert job_protection(state, [raise_])[0] == pytest.approx(JOB_PROTECTION)
    assert fiscal_cost(raise_, state) > 0
    assert job_protection(state, [cut(magnitude)]) == {}
    assert fiscal_cost(cut(magnitude), state) == 0


def test_a_cut_with_no_scheme_running_is_blocked() -> None:
    state = toy_world()
    constrained = constrain([cut()], state, [])
    assert constrained.actions == []
    assert "no wage subsidy scheme is running" in constrained.notes[0]
    assert [s for s in actions_to_shocks([cut()], state) if s.node == FISCAL_NODE] == []


def test_a_cut_removes_the_cover_it_saves_on() -> None:
    state = running(turns=6)  # cover left on turns 1..5
    constrained = constrain([cut(turns=12)], state, [])
    [trimmed] = constrained.actions
    assert trimmed.duration_turns == 5
    assert job_protection(state, [trimmed]) == {}
    [saving] = [s for s in actions_to_shocks([trimmed], state) if s.node == FISCAL_NODE]
    [cost] = [s for s in actions_to_shocks([furlough(turns=6)], state) if s.node == FISCAL_NODE]
    assert saving.delta == pytest.approx(-cost.delta)
    assert saving.duration_turns == 5


def test_a_partial_cut_leaves_the_rest_of_the_scheme() -> None:
    state = running()
    half = cut(-0.5, turns=2)
    cover = job_protection(state, [half])
    assert cover[1] == pytest.approx(0.5 * JOB_PROTECTION)
    assert cover[3] == pytest.approx(JOB_PROTECTION)
    assert fiscal_cost(half, state) == pytest.approx(-0.5 * fiscal_cost(furlough(), state))


def test_a_cut_cannot_save_more_than_the_scheme_costs() -> None:
    state = running()
    [trimmed] = constrain([cut()], state, []).actions
    huge = cut(-1.0).model_copy(update={"duration_turns": 24})
    assert fiscal_cost(huge, state) == pytest.approx(fiscal_cost(trimmed, state))
