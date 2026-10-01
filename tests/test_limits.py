"""Feasibility, diminishing returns and political capital, applied by the game loop (EB-2, EB-3)."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import PolicyAction
from hog_sim.game.loop import Game, replay
from hog_sim.game.stubs import CannedScenarios, EngineForecaster
from hog_sim.policy.limits import REPEAT_DECAY, constrain
from hog_sim.world.seed.toy import toy_world


def act(kind, target, magnitude=0.5):
    return PolicyAction(kind=kind, target=target, magnitude=magnitude)


class Scripted:
    def __init__(self, *actions):
        self.actions = list(actions)

    def interpret(self, text, state, scenario):
        return [a.model_copy(deep=True) for a in self.actions]


def game_with(*actions, capital=1.5):
    config = GameConfig(election_turn=6, k_draws=10, capital_per_turn=capital)
    return Game(config, toy_world(), CannedScenarios(0), Scripted(*actions), EngineForecaster())


def test_blocked_actions_are_dropped_in_every_mode() -> None:
    game = game_with(act("appoint", "institution:central_bank"), act("spend", "sector:public"))
    record = game.play_turn("sack the governor and fund the NHS")
    assert [a.target for a in record.actions] == ["sector:public"]
    assert len(record.requested_actions) == 2
    assert any("Bank of England is independent" in n for n in record.notes)


def test_spending_is_blocked_above_the_deficit_limit() -> None:
    world = toy_world()
    world.indicators["indicator:deficit"].value = 12
    limited = constrain([act("spend", "sector:public")], world, [])
    assert limited.actions == []
    assert "above the 10% limit" in limited.notes[0]


def test_repeating_a_lever_brings_diminishing_returns() -> None:
    game = game_with(act("deregulate", "sector:energy", 0.8))
    sizes = [game.play_turn("deregulate energy").actions[0].magnitude for _ in range(3)]
    assert sizes == pytest.approx([0.8, 0.8 * REPEAT_DECAY, 0.8 * REPEAT_DECAY**2])
    assert "Diminishing returns" in game.history[-1].notes[0]


def test_a_different_lever_is_not_penalised() -> None:
    world = toy_world()
    past = [[act("spend", "sector:public")]]
    limited = constrain([act("spend", "sector:housing")], world, past)
    assert limited.actions[0].magnitude == 0.5 and limited.notes == []


def test_political_capital_scales_an_oversized_package() -> None:
    world = toy_world()
    package = [act("spend", "sector:public", 1.0), act("deregulate", "sector:housing", 1.0)]
    limited = constrain(package, world, [], capital=1.5)
    assert [a.magnitude for a in limited.actions] == pytest.approx([0.75, 0.75])
    assert "Political capital" in limited.notes[0]
    # Speeches are cheap
    talk = constrain([act("communicate", "group:pensioners", 1.0)] * 4, world, [], capital=1.5)
    assert talk.notes == []


def test_replay_matches_with_limits_applied() -> None:
    game = game_with(act("deregulate", "sector:energy", 1.0), act("spend", "sector:public", 1.0))
    while not game.over:
        game.play_turn("same again")
    assert replay(game.start, game.config, game.history) == game.state
