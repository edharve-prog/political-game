"""Markets turn on unfunded tax cuts (Project 11, CA-4): the 2022 mini-budget."""

import pytest

from hog_sim.core.models import PolicyAction
from hog_sim.population.popularity import CONFIDENCE_APPROVAL, policy_events
from hog_sim.world.propagation import (
    CONFIDENCE_SHOCKS,
    CONFIDENCE_THRESHOLD,
    FISCAL_STEPS,
    actions_to_shocks,
    unfunded_tax_cut,
)
from hog_sim.world.seed.toy import toy_world


def act(kind: str, target: str, magnitude: float) -> PolicyAction:
    return PolicyAction(kind=kind, target=target, magnitude=magnitude, duration_turns=1)


BIG_CUT = act("tax", "country:uk", -0.8)


def test_a_big_unfunded_tax_cut_counts() -> None:
    assert unfunded_tax_cut([BIG_CUT]) == pytest.approx(0.8 * FISCAL_STEPS)


def test_small_cuts_and_spending_do_not() -> None:
    small = act("tax", "country:uk", -0.5)
    assert 0.5 * FISCAL_STEPS < CONFIDENCE_THRESHOLD
    assert unfunded_tax_cut([small]) == 0
    assert unfunded_tax_cut([act("spend", "sector:finance", 1.0)]) == 0


def test_a_cut_paid_for_the_same_turn_does_not() -> None:
    paid = [BIG_CUT, act("spend", "sector:public", -0.8)]
    assert unfunded_tax_cut(paid) == 0


def test_markets_push_rates_up_and_the_pound_down() -> None:
    world = toy_world()
    shocks = {s.node: s.delta for s in actions_to_shocks([BIG_CUT], world) if not s.hold}
    loss = unfunded_tax_cut([BIG_CUT])
    for node, size in CONFIDENCE_SHOCKS.items():
        assert shocks[node] == pytest.approx(size * loss)


def test_every_group_loses_approval() -> None:
    world = toy_world()
    events = policy_events([BIG_CUT], world)
    assert len(events) == 1
    loss = -CONFIDENCE_APPROVAL * unfunded_tax_cut([BIG_CUT])
    assert events[0].group_effects == {g: pytest.approx(loss) for g in world.groups}
    assert policy_events([act("tax", "country:uk", -0.5)], world) == []
