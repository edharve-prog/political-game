"""Costless levers (review of 13 October: E1, E2, E4, E12, O1, O2, D4).

Duration is priced, U-turns are not repeats, every lever that costs no money has a loser or a
risk, and no package that costs nothing in deficit buys the election.
"""

from typing import get_args

import pytest

from hog_sim.core.models import ActionKind, PolicyAction
from hog_sim.forecasting.scoring import base_rate
from hog_sim.game.balance import LIMITS, exploits, quiet_game, search
from hog_sim.policy.feasibility import compatibility_problem
from hog_sim.policy.limits import (
    MAX_TURNS,
    capital_cost,
    constrain,
    lifetime_cost,
)
from hog_sim.population.popularity import policy_events
from hog_sim.world.propagation import actions_to_shocks
from hog_sim.world.seed.toy import toy_world

MARGIN = LIMITS["max_free_package_gain"]


def act(kind, target, magnitude=1.0, turns=1, **extra):
    return PolicyAction(
        kind=kind, target=target, magnitude=magnitude, duration_turns=turns, **extra
    )


@pytest.fixture(scope="module")
def idle() -> float:
    return quiet_game({}).vote_share


def gain(idle: float, *actions: PolicyAction, schedule=None) -> float:
    return quiet_game(schedule or {0: list(actions)}).vote_share - idle


# Property 1: lifetime capital or deficit cost rises with duration for every action kind.


def _pairs():
    state = toy_world()
    for kind in get_args(ActionKind):
        if kind == "do_nothing":
            continue
        for target in sorted(state.node_ids()):
            if compatibility_problem(act(kind, target), state) is None:
                yield kind, target


@pytest.mark.parametrize("kind", [k for k in get_args(ActionKind) if k != "do_nothing"])
def test_lifetime_cost_rises_with_duration_for_every_kind(kind) -> None:
    state = toy_world()
    targets = [t for k, t in _pairs() if k == kind]
    assert targets, f"no target accepts {kind}"
    for target in targets:
        for size in (1.0, -1.0):
            costs = [
                lifetime_cost(act(kind, target, size, turns), state)
                for turns in range(1, MAX_TURNS.get(kind, 24) + 1)
            ]
            for shorter, longer in zip(costs, costs[1:], strict=False):
                assert longer.capital > shorter.capital, (kind, target, size)
                assert abs(longer.deficit_turn_pts) >= abs(shorter.deficit_turn_pts)


def test_a_long_measure_is_scaled_to_fit_the_capital_it_uses_over_its_life() -> None:
    state = toy_world()
    one = constrain([act("deregulate", "sector:energy", 1.0, 1)], state, []).actions[0]
    year = constrain([act("deregulate", "sector:energy", 1.0, 24)], state, []).actions[0]
    assert one.magnitude == pytest.approx(1.0)
    assert year.magnitude < 0.5  # 24 turns cost 3.3 turns' capital: 1.5 / 3.3
    assert capital_cost([year], state) == pytest.approx(1.5)


def test_speeches_run_for_a_few_turns_at_most() -> None:
    limited = constrain([act("communicate", "group:pensioners", 1.0, 24)], toy_world(), [])
    assert limited.actions[0].duration_turns == MAX_TURNS["communicate"]
    assert any(n.startswith("Duration:") for n in limited.notes)


# E12: reversing a measure is a U-turn, not a repeat.


def test_reversing_a_running_measure_is_a_u_turn_not_a_repeat() -> None:
    state = toy_world()
    rise = act("tax", "sector:finance", 0.6, 6)
    cut = act("tax", "sector:finance", -0.6, 1)
    limited = constrain([cut], state, [[rise]])
    assert limited.actions[0].magnitude == pytest.approx(-0.6)  # not halved to -0.3
    assert any(n.startswith("U-turn:") for n in limited.notes)
    assert capital_cost([cut], state, past_actions=[[rise]]) > capital_cost([cut], state)


def test_reversing_a_finished_measure_is_neither() -> None:
    state = toy_world()
    cut = act("tax", "sector:finance", -0.6, 1)
    limited = constrain([cut], state, [[act("tax", "sector:finance", 0.6, 1)]])
    assert limited.actions[0].magnitude == pytest.approx(-0.6)
    assert not any(n.startswith(("U-turn:", "Diminishing")) for n in limited.notes)


def test_repeating_the_same_direction_still_wears_out() -> None:
    state = toy_world()
    rise = act("tax", "sector:finance", 0.6, 1)
    limited = constrain([rise], state, [[rise]])
    assert limited.actions[0].magnitude == pytest.approx(0.3)


# O2 and E2: laws aimed at a group have a losing side and fade unless money follows.


def test_a_law_for_a_group_costs_its_counterparty() -> None:
    (event,) = policy_events([act("legislate", "group:pensioners", 1.0, 24)], toy_world())
    assert event.group_effects["group:pensioners"] > 0
    assert event.group_effects["group:young_renters"] == pytest.approx(
        -event.group_effects["group:pensioners"]
    )


def test_a_law_with_nothing_behind_it_wears_thin() -> None:
    state = toy_world()
    law = act("legislate", "group:pensioners", 1.0, 24)
    (alone,) = policy_events([law], state)
    funded = policy_events([law, act("spend", "group:pensioners", 0.5, 24)], state)
    assert alone.hold_turns == 2
    assert funded[0].hold_turns == 23


@pytest.mark.parametrize("kind,size", [("regulate", -1.0), ("deregulate", 1.0)])
def test_a_rule_aimed_at_a_group_is_a_law(kind, size) -> None:
    (event,) = policy_events([act(kind, "group:pensioners", size, 24)], toy_world())
    assert event.group_effects["group:young_renters"] < 0
    assert event.hold_turns == 2


# E4: speeches lift confidence once; deregulation has a loser and, for some sectors, a risk.


def test_a_speech_to_a_sector_does_not_hold_its_output_up() -> None:
    (shock,) = actions_to_shocks([act("communicate", "sector:public", 1.0, 3)], toy_world())
    assert not shock.hold and shock.duration_turns == 1


def test_deregulation_costs_workers_and_moves_its_side_indicator() -> None:
    state = toy_world()
    energy = act("deregulate", "sector:energy", 1.0, 6)
    shocks = {s.node: s for s in actions_to_shocks([energy], state)}
    assert shocks["indicator:energy_prices"].delta > 0 and shocks["indicator:energy_prices"].hold
    (event,) = policy_events([energy], state)
    assert all(v < 0 for v in event.group_effects.values())
    (public,) = policy_events([act("deregulate", "sector:public", 1.0)], state)
    assert public.group_effects["group:public_workers"] < 0


def test_deregulating_finance_makes_a_selloff_likelier() -> None:
    finance = [act("deregulate", "sector:finance", 1.0)]
    assert base_rate(["market_selloff"], actions=finance) == pytest.approx(
        2 * base_rate(["market_selloff"])
    )
    assert base_rate(["strike"], actions=finance) == pytest.approx(base_rate(["strike"]))


# Property 2: no package that costs nothing in deficit, within one turn's capital, beats doing
# nothing by more than 0.5 vote points at the election.

REVIEW_PACKAGES = {
    "law for pensioners": [act("legislate", "group:pensioners", 1.0, 24)],
    "speech to pensioners": [act("communicate", "group:pensioners", 1.0, 24)],
    "deregulate the public sector": [act("deregulate", "sector:public", 1.0, 24)],
    "speech to the public sector": [act("communicate", "sector:public", 1.0, 24)],
    "regulate on behalf of pensioners": [act("regulate", "group:pensioners", -1.0, 24)],
}


@pytest.mark.parametrize("name", REVIEW_PACKAGES)
def test_the_review_s_free_packages_no_longer_pay(idle, name) -> None:
    assert gain(idle, *REVIEW_PACKAGES[name]) <= MARGIN


def test_the_four_turn_law_bot_no_longer_pays(idle) -> None:
    groups = ["group:pensioners", "group:young_renters", "group:public_workers", "group:business"]
    schedule = {
        turn: [act("legislate", g, 1.0, 24), act("communicate", g, 1.0, 24)]
        for turn, g in enumerate(groups)
    }
    assert gain(idle, schedule=schedule) <= MARGIN


@pytest.mark.xfail(
    reason="E6: taxing or paying a group has no economic effect yet (Fiscal incidence thread)",
    strict=False,
)
def test_a_deficit_neutral_transfer_between_groups_does_not_buy_the_election(idle) -> None:
    package = [act("tax", "group:business", 0.75, 24), act("spend", "group:pensioners", 0.75, 24)]
    assert gain(idle, *package) <= MARGIN


def test_the_exploit_search_finds_no_free_package_that_pays() -> None:
    probes = search(budget=60)
    assert len(probes) == 60
    assert exploits(probes) == []
