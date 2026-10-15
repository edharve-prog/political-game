"""Taxing or paying a group moves the economy and other groups too (review of 13 October 2026,
E6).

A group-targeted tax or spend used to reach only the deficit and that group's approval:
taxing business owners to raise pensions moved no sector and won votes for free.
"""

import pytest

from hog_sim.core.models import PolicyAction
from hog_sim.population.popularity import FAIRNESS, policy_events
from hog_sim.world.propagation import (
    DEMAND_SHARE,
    FISCAL_NODE,
    HOME_SHARE,
    actions_to_shocks,
    demand_shocks,
    fiscal_cost,
    scale,
)
from hog_sim.world.seed.toy import toy_world


def act(kind: str, target: str, magnitude: float, turns: int = 12) -> PolicyAction:
    return PolicyAction(kind=kind, target=target, magnitude=magnitude, duration_turns=turns)


def spent_bn(shocks, world) -> float:
    return sum(s.delta * scale(world, s.node) for s in shocks if s.node in world.sectors)


def test_a_pension_rise_is_spent_across_the_private_sectors() -> None:
    world = toy_world()
    shocks = demand_shocks(act("spend", "group:pensioners", 1.0), world)
    assert {s.node for s in shocks} == {s for s in world.sectors if s != "sector:public"}
    assert all(s.delta > 0 and s.hold and s.duration_turns == 12 for s in shocks)
    money = fiscal_cost(act("spend", "group:pensioners", 1.0), world)
    money_bn = money * scale(world, FISCAL_NODE) / 100 * world.countries["country:uk"].gdp_bn
    at_home = money_bn * DEMAND_SHARE["group:pensioners"] * HOME_SHARE
    assert spent_bn(shocks, world) == pytest.approx(at_home)


def test_taxing_business_owners_cuts_investment() -> None:
    world = toy_world()
    shocks = demand_shocks(act("tax", "group:business", 0.75), world)
    assert {s.node for s in shocks} == {"sector:manufacturing", "sector:finance"}
    assert all(s.delta < 0 for s in shocks)


def test_a_transfer_lifts_output_by_less_than_its_cost() -> None:
    """Part of a transfer is saved, so it never adds more output than it costs."""
    world = toy_world()
    gdp = world.countries["country:uk"].gdp_bn
    for group in world.groups:
        action = act("spend", group, 1.0)
        cost_bn = fiscal_cost(action, world) * scale(world, FISCAL_NODE) / 100 * gdp
        assert 0 < spent_bn(demand_shocks(action, world), world) < cost_bn


def test_only_tax_and_spending_move_demand() -> None:
    world = toy_world()
    for kind in ("communicate", "legislate", "regulate"):
        assert demand_shocks(act(kind, "group:pensioners", 1.0), world) == []
    shocks = actions_to_shocks([act("spend", "group:pensioners", 1.0)], world)
    assert any(s.node == FISCAL_NODE for s in shocks)
    assert any(s.node in world.sectors for s in shocks)


def test_money_for_one_group_costs_the_others_a_little() -> None:
    world = toy_world()
    [event] = policy_events([act("spend", "group:pensioners", 0.75)], world)
    gain = event.group_effects["group:pensioners"]
    assert gain > 0
    for group in world.groups:
        if group != "group:pensioners":
            assert event.group_effects[group] == pytest.approx(-FAIRNESS * gain)


def test_a_tax_rise_or_a_speech_draws_no_fairness_reaction() -> None:
    world = toy_world()
    for action in (act("tax", "group:business", 0.75), act("communicate", "group:pensioners", 1)):
        [event] = policy_events([action], world)
        assert set(event.group_effects) == {action.target}
