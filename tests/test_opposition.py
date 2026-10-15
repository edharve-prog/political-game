"""An opposition that moves (review 2026-10-13, O9)."""

import pytest

import hog_sim.world.opposition as opposition
from hog_sim.core.config import GameConfig
from hog_sim.game.loop import Game, replay
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.llm.summary import summarise_state
from hog_sim.world.opposition import (
    LEGISLATURE,
    QUARTER,
    STANCES,
    choose_stance,
    hold_chance,
    step_opposition,
)
from hog_sim.world.seed.toy import toy_world


def test_nothing_happens_between_quarters() -> None:
    world = toy_world()
    for turn in range(QUARTER - 1):
        assert step_opposition(world, turn, seed=0) is world


def test_the_opposition_goes_after_the_most_disaffected_voters() -> None:
    world = toy_world()
    world.groups["group:young_renters"].approval -= 0.2
    assert choose_stance(world) == "housing and youth"
    world = toy_world()
    world.groups["group:business"].approval -= 0.2
    assert choose_stance(world) == "sound money"


def test_angering_a_small_group_does_not_lure_the_opposition_away() -> None:
    """The exploit search found that a one-turn tax on business drew the opposition onto
    business and homeowners and away from bigger groups, buying the government 0.6 points.
    The opposition counts votes first, so a small grievance does not move it."""
    world = toy_world()
    start = choose_stance(world)
    world.groups["group:business"].approval -= 0.05
    assert choose_stance(world) == start


def test_a_stance_pulls_its_groups_and_a_new_one_takes_the_old_pull_back() -> None:
    world = toy_world()
    leans = {gid: g.lean for gid, g in world.groups.items()}
    world.groups["group:business"].approval -= 0.2
    moved = step_opposition(world, QUARTER - 1, seed=0)
    assert moved.opposition.stance == "sound money"
    pull = STANCES["sound money"].pull["group:business"]
    assert pull < 0
    assert moved.groups["group:business"].lean == pytest.approx(leans["group:business"] + pull)
    moved.groups["group:business"].approval += 0.2
    moved.groups["group:young_renters"].approval -= 0.3
    again = step_opposition(moved, 2 * QUARTER - 1, seed=0)
    assert again.opposition.stance == "housing and youth"
    assert again.groups["group:business"].lean == pytest.approx(leans["group:business"])
    assert again.groups["group:young_renters"].lean == pytest.approx(
        leans["group:young_renters"] + STANCES["housing and youth"].pull["group:young_renters"]
    )


def test_a_stance_measures_disaffection_without_its_own_pull() -> None:
    """A stance's pull lowers leans; it must not count that as new disaffection to chase."""
    world = toy_world()
    world.groups["group:pensioners"].approval -= 0.1
    first = step_opposition(world, QUARTER - 1, seed=0)
    assert first.opposition.stance == choose_stance(first)


def test_byelections_move_the_commons(monkeypatch) -> None:
    monkeypatch.setattr(opposition, "BYELECTION_CHANCE", 1.0)
    world = toy_world()
    support = world.institutions[LEGISLATURE].support
    after = step_opposition(world, QUARTER - 1, seed=0)
    [vote] = after.opposition.byelections
    change = after.institutions[LEGISLATURE].support - support
    assert change == pytest.approx(
        opposition.BYELECTION_HOLD if vote.held else -opposition.BYELECTION_LOSS
    )
    assert hold_chance(0.30) == 0.05 and hold_chance(0.60) == 0.95
    assert hold_chance(0.45) == 0.5


def test_games_replay_with_the_opposition_and_prompts_mention_it() -> None:
    config = GameConfig(seed=2, k_draws=10)
    start = toy_world()
    game = Game(config, start, CannedScenarios(2), KeywordInterpreter(), EngineForecaster())
    for _ in range(2 * QUARTER):
        game.play_turn("Raise pensions")
    assert game.state.opposition.stance in STANCES
    assert replay(start, config, game.history) == game.state
    assert "Opposition: campaigning on" in summarise_state(game.state).to_prompt()
