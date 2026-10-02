"""Each kind of action moves what it is meant to move, the way the player meant (EB-3, EB-4).

These are the review's effect probes as tests: one action on a quiet world, resolved turn by
turn with nothing else happening.
"""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction, Scenario
from hog_sim.game.loop import resolve
from hog_sim.world.seed.toy import toy_world

QUIET = Scenario(title="Quiet month", briefing="", affected_nodes=[], urgency=0)
NOTHING = Outcome(narrative="Nothing happens", probability=1)


def act(kind, target, magnitude, turns=1):
    return PolicyAction(kind=kind, target=target, magnitude=magnitude, duration_turns=turns)


def run(actions, turns=8):
    """States after each turn, starting from the toy world, with the actions on turn 0."""
    start = state = toy_world()
    states = []
    for t in range(turns):
        state = resolve(start, state, QUIET, actions if t == 0 else [], NOTHING, GameConfig())
        states.append(state)
    return states


def value(state, node):
    return state.indicators[node].value


BASE = toy_world()


@pytest.mark.parametrize(
    ("action", "indicator", "direction"),
    [
        (act("spend", "sector:energy", 0.5), "indicator:energy_prices", -1),
        (act("deregulate", "sector:energy", 0.5), "indicator:energy_prices", -1),
        (act("regulate", "sector:energy", 0.5), "indicator:energy_prices", 1),
        (act("tax", "sector:energy", 0.5), "indicator:energy_prices", 1),
        (act("regulate", "indicator:energy_prices", -0.5), "indicator:energy_prices", -1),
        (act("spend", "indicator:energy_prices", -0.5), "indicator:energy_prices", -1),
        (act("tax", "indicator:house_prices", -0.5), "indicator:house_prices", -1),
        (act("spend", "sector:manufacturing", 0.5), "indicator:unemployment", -1),
        (act("spend", "sector:public", 0.5), "indicator:deficit", 1),
        (act("tax", "sector:finance", 0.5), "indicator:deficit", -1),
        (act("spend", "indicator:energy_prices", -0.5), "indicator:deficit", 1),
        (act("tax", "indicator:house_prices", -0.5), "indicator:deficit", -1),
    ],
)
def test_action_moves_its_indicator_the_intended_way(action, indicator, direction) -> None:
    states = run([action], turns=4)
    change = max((value(s, indicator) - value(BASE, indicator) for s in states), key=abs)
    assert change * direction > 0


def test_capping_bills_trims_energy_output_but_still_lowers_bills() -> None:
    after = run([act("regulate", "indicator:energy_prices", -0.5)], turns=1)[0]
    assert after.sectors["sector:energy"].output_bn < BASE.sectors["sector:energy"].output_bn
    assert value(after, "indicator:energy_prices") < 100


def test_talking_to_the_bank_of_england_does_not_move_rates() -> None:
    for kind in ("communicate", "appoint"):
        states = run([act(kind, "institution:central_bank", 0.5)], turns=3)
        assert value(states[-1], "indicator:interest_rate") == pytest.approx(4.0)


def test_one_off_shocks_fade() -> None:
    states = run([act("deregulate", "sector:energy", 1.0)], turns=24)
    first = value(states[0], "indicator:energy_prices") - 100
    last = value(states[-1], "indicator:energy_prices") - 100
    assert first < 0
    assert abs(last) < abs(first) / 5


def test_a_running_programme_holds_its_effect_and_its_cost() -> None:
    one_off = run([act("spend", "sector:public", 0.5)], turns=12)
    year = run([act("spend", "sector:public", 0.5, turns=12)], turns=12)
    deficit = BASE.indicators["indicator:deficit"].value
    # Both cost the same at first; a year-long programme is still costing at month 10
    assert value(one_off[0], "indicator:deficit") == pytest.approx(
        value(year[0], "indicator:deficit")
    )
    assert value(year[9], "indicator:deficit") > deficit + 0.4
    assert value(one_off[9], "indicator:deficit") < deficit + 0.1
    public = BASE.sectors["sector:public"].output_bn
    assert (
        year[9].sectors["sector:public"].output_bn > one_off[9].sectors["sector:public"].output_bn
    )
    assert year[9].sectors["sector:public"].output_bn > public


def test_a_group_policy_keeps_its_voters_while_it_runs() -> None:
    states = run([act("spend", "group:pensioners", 0.5, turns=12)], turns=16)
    approval = [s.groups["group:pensioners"].approval for s in states]
    baseline = run([], turns=16)
    base = [s.groups["group:pensioners"].approval for s in baseline]
    # Still well above where it would be in month 10, and fading once the policy ends
    assert approval[9] > base[9] + 0.03
    assert approval[15] - base[15] < approval[11] - base[11]


def test_foreign_policy_moves_relationships() -> None:
    warm = run([act("diplomatic", "country:eu", 0.5)], turns=1)[0].countries["country:eu"]
    assert warm.relationship == pytest.approx(BASE.countries["country:eu"].relationship + 0.05)
    hostile = run([act("military", "country:china", 0.5)], turns=1)[0].countries["country:china"]
    assert hostile.relationship < BASE.countries["country:china"].relationship
    assert hostile.stability < BASE.countries["country:china"].stability


def test_indicators_stay_inside_their_bounds() -> None:
    states = run([act("deregulate", "sector:housing", 1.0, turns=24)] * 3, turns=24)
    for state in states:
        for ind in state.indicators.values():
            assert ind.low <= ind.value <= ind.high


def test_outcome_events_move_the_economy() -> None:
    """A tagged market sell-off raises Bank Rate and hits finance (EB-5)."""
    start = toy_world()
    selloff = Outcome(narrative="Markets sold off", events=["market_selloff"], probability=1)
    after = resolve(start, start, QUIET, [], selloff, GameConfig())
    assert value(after, "indicator:interest_rate") > value(start, "indicator:interest_rate")
    assert after.sectors["sector:finance"].output_bn < start.sectors["sector:finance"].output_bn
    quiet = resolve(start, start, QUIET, [], NOTHING, GameConfig())
    assert value(quiet, "indicator:interest_rate") == value(start, "indicator:interest_rate")


def test_unknown_event_nodes_are_skipped() -> None:
    from hog_sim.world.events import event_shocks

    world = toy_world()
    del world.sectors["sector:finance"]
    world.edges = [e for e in world.edges if "sector:finance" not in (e.source, e.target)]
    nodes = {s.node for s in event_shocks(["market_selloff", "media_praise"], world)}
    assert nodes == {"indicator:interest_rate"}
