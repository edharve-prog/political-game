"""Political capital as a stock (O5) and one credibility score (D9), review of 13 October."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, Pledge, PolicyAction, Scenario
from hog_sim.game.loop import Game, replay, resolve
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.policy import capital, credibility
from hog_sim.policy.capital import available, cycle_factor, next_capital, refill
from hog_sim.policy.credibility import CredibilityChange, speech_weight, step_credibility
from hog_sim.policy.limits import constrain, still_running
from hog_sim.population.popularity import target_approval
from hog_sim.world.seed.toy import toy_world

QUIET = Scenario(title="A quiet month", briefing="", affected_nodes=[], urgency=0.0)
CRISIS = Scenario(title="Banks wobble", briefing="", affected_nodes=["sector:finance"], urgency=0.9)
NOTHING = Outcome(narrative="Nothing else happens.", probability=1.0)
CONFIG = GameConfig(seed=0, election_turn=24)


def act(kind, target, magnitude=1.0, turns=1):
    return PolicyAction(kind=kind, target=target, magnitude=magnitude, duration_turns=turns)


def play(state, actions, scenario=QUIET, pledges=()):
    return resolve(toy_world(), state, scenario, actions, NOTHING, CONFIG, pledges)


# O5: capital as a stock


def test_the_first_turn_has_the_base_and_later_turns_hold_a_stock() -> None:
    start = toy_world()
    assert start.capital is None
    assert available(start, CONFIG) == CONFIG.capital_per_turn
    after = play(start, [])
    assert after.capital is not None
    assert available(after, CONFIG) == after.capital


def test_the_cycle_gives_a_honeymoon_and_a_midterm_slump() -> None:
    assert cycle_factor(0, 24) == pytest.approx(1 + capital.HONEYMOON)
    assert cycle_factor(12, 24) < 0.85
    assert cycle_factor(24, 24) == pytest.approx(1.0, abs=0.01)
    assert cycle_factor(1, 24) > 1 > cycle_factor(12, 24) < cycle_factor(22, 24)


def test_unspent_capital_is_partly_banked() -> None:
    idle = play(toy_world(), [])
    busy = play(
        toy_world(), [act("spend", "sector:public", 0.75), act("spend", "sector:housing", 0.75)]
    )
    # Doing nothing banks half the base; spending it all banks nothing.
    banked = capital.BANK_SHARE * CONFIG.capital_per_turn
    assert idle.capital == pytest.approx(refill(idle, CONFIG) + banked)
    assert busy.capital == pytest.approx(refill(busy, CONFIG))


def test_the_stock_is_capped() -> None:
    state = toy_world()
    for _ in range(8):
        state = play(state, [])
    assert state.capital <= capital.MAX_STOCK * CONFIG.capital_per_turn + 1e-9


def test_the_refill_rises_with_approval_and_majority() -> None:
    low, high = toy_world(), toy_world()
    for group in high.groups.values():
        group.approval += 0.05
    high.institutions["institution:legislature"].support = 0.75
    low.institutions["institution:legislature"].support = 0.45
    assert refill(high, CONFIG) > refill(toy_world(), CONFIG) > refill(low, CONFIG)


def test_the_stock_caps_what_a_package_can_do() -> None:
    state = toy_world().model_copy(update={"capital": 0.5})
    limited = constrain([act("spend", "sector:public", 1.0)], state, [], capital=0.5)
    assert limited.actions[0].magnitude == pytest.approx(0.5)
    assert any("Political capital" in n for n in limited.notes)


def test_next_capital_never_banks_an_overspend() -> None:
    state = toy_world()
    assert next_capital(state, -3.0, CONFIG) == pytest.approx(refill(state, CONFIG))


# D9: credibility


def test_measures_remember_how_long_they_have_left() -> None:
    running = still_running([], [act("tax", "sector:energy", 0.5, 3), act("do_nothing", "x")])
    assert [a.duration_turns for a in running] == [2]
    assert still_running(still_running(running, []), []) == []


def test_a_u_turn_costs_credibility() -> None:
    state = play(toy_world(), [act("tax", "sector:energy", 0.5, 6)])
    steady = play(state, [])
    turned = play(state, [act("tax", "sector:energy", -0.5, 1)])
    assert turned.credibility == pytest.approx(
        steady.credibility - credibility.UTURN_LOSS * (1 - credibility.SETTLE)
    )


def test_a_broken_pledge_costs_credibility() -> None:
    pledge = Pledge(text="No new taxes", kind="tax", direction="up")
    state = play(toy_world(), [], pledges=[pledge])
    kept = play(state, [])
    broken = play(state, [act("tax", "sector:energy", 0.5)])
    assert broken.credibility < kept.credibility


def test_a_crisis_met_with_action_builds_credibility_and_words_cost_it() -> None:
    acted = play(toy_world(), [act("spend", "sector:finance", 0.5)], CRISIS)
    talked = play(toy_world(), [act("communicate", "group:business", 0.5)], CRISIS)
    quiet = play(toy_world(), [act("communicate", "group:business", 0.5)])
    elsewhere = play(toy_world(), [act("spend", "sector:housing", 0.5)], CRISIS)
    assert acted.credibility > quiet.credibility > talked.credibility
    assert elsewhere.credibility == pytest.approx(talked.credibility)


def test_credibility_drifts_back_toward_neutral() -> None:
    low = step_credibility(0.2, CredibilityChange())
    assert 0.2 < low < credibility.NEUTRAL


def test_credibility_scales_speeches() -> None:
    speech = act("communicate", "group:pensioners", 0.4)
    doubted = toy_world().model_copy(update={"credibility": 0.2})
    trusted = toy_world().model_copy(update={"credibility": 0.5})
    assert constrain([speech], trusted, []).actions[0].magnitude == pytest.approx(0.4)
    weak = constrain([speech], doubted, [])
    assert weak.actions[0].magnitude == pytest.approx(0.4 * speech_weight(0.2))
    assert any(n.startswith("Credibility:") for n in weak.notes)


def test_credibility_shifts_every_group_and_the_refill() -> None:
    start = toy_world()
    doubted = start.model_copy(update={"credibility": 0.2})
    base = target_approval(start, start)
    low = target_approval(doubted, start)
    for gid in base:
        assert low[gid] == pytest.approx(base[gid] - 0.03)
    assert refill(doubted, CONFIG) < refill(start, CONFIG)


def test_a_game_replays_its_capital_and_credibility() -> None:
    config = GameConfig(seed=2, election_turn=6, k_draws=20)
    plugins = (CannedScenarios(2), KeywordInterpreter(), EngineForecaster())
    game = Game(config, toy_world(), *plugins)
    for text in ("Raise taxes on energy", "Cut taxes on energy", "Do nothing"):
        game.play_turn(text)
    final = replay(game.start, config, game.history)
    assert final.capital == game.state.capital
    assert final.credibility == game.state.credibility
    assert final.measures == game.state.measures
