"""Saving and resuming: a finished turn is saved before the next scenario is fetched, and a
resumed game shows the scenario that was waiting rather than a new one."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Scenario
from hog_sim.game.loop import Game
from hog_sim.game.persistence import SaveStore
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.world.seed.toy import toy_world


class FlakySource:
    """Serves canned scenarios until told to fail, and counts the calls."""

    def __init__(self) -> None:
        self.inner = CannedScenarios(0)
        self.calls = 0
        self.fail = False

    def next_scenario(self, state, history) -> Scenario:
        self.calls += 1
        if self.fail:
            raise ConnectionError("provider down")
        return self.inner.next_scenario(state, history)


class FreshSource:
    """A different scenario every call, as a live generator would give."""

    def __init__(self) -> None:
        self.calls = 0

    def next_scenario(self, state, history) -> Scenario:
        self.calls += 1
        return Scenario(
            title=f"Story {self.calls}",
            briefing="Something happened.",
            affected_nodes=[],
            urgency=0.5,
        )


def config() -> GameConfig:
    return GameConfig(election_turn=6, k_draws=10)


def test_commit_returns_before_the_next_scenario_is_fetched(tmp_path) -> None:
    store = SaveStore(tmp_path / "game.db")
    source = FlakySource()
    game = Game(config(), toy_world(), source, KeywordInterpreter(), EngineForecaster())
    game_id = store.new_game(game.config, game.start)
    proposal = game.propose("Invest in housebuilding")
    source.fail = True  # the provider goes down after the turn is played
    record = game.commit(proposal)
    store.save_turn(game_id, record)  # reached: commit made no scenario request
    with pytest.raises(ConnectionError):
        _ = game.scenario
    _, _, records = store.load(game_id)
    assert [r.turn for r in records] == [0]


def test_resume_shows_the_scenario_that_was_waiting(tmp_path) -> None:
    store = SaveStore(tmp_path / "game.db")
    source = FreshSource()
    game_id = store.new_game(config(), toy_world())
    game = Game(
        config(),
        toy_world(),
        source,
        KeywordInterpreter(),
        EngineForecaster(),
        on_scenario=lambda turn, s: store.save_scenario(game_id, turn, s),
    )
    store.save_turn(game_id, game.play_turn("Do nothing"))
    waiting = game.scenario
    game.add_options(["Ask the advisers"])

    cfg, start, records = store.load(game_id)
    turn = records[-1].state_after.turn
    calls = source.calls
    resumed = Game.resume(
        cfg,
        start,
        records,
        source,
        KeywordInterpreter(),
        EngineForecaster(),
        scenario=store.pending_scenario(game_id, turn),
    )
    assert resumed.scenario.title == waiting.title
    assert resumed.scenario.suggested_options[-1] == "Ask the advisers"
    assert source.calls == calls  # no new request on resume


def test_a_scenario_saved_for_an_earlier_turn_is_not_reused(tmp_path) -> None:
    store = SaveStore(tmp_path / "game.db")
    game_id = store.new_game(config(), toy_world())
    store.save_scenario(game_id, 2, FreshSource().next_scenario(None, []))
    assert store.pending_scenario(game_id, 3) is None
    assert store.pending_scenario(game_id, 2).title == "Story 1"


def test_construction_and_resume_make_no_scenario_requests() -> None:
    source = FreshSource()
    game = Game(config(), toy_world(), source, KeywordInterpreter(), EngineForecaster())
    record = game.play_turn("Do nothing")
    assert source.calls == 1
    Game.resume(config(), toy_world(), [record], source, KeywordInterpreter(), EngineForecaster())
    assert source.calls == 1
