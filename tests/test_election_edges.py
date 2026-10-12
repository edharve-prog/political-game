"""Election edge cases: nobody voting, a tied house, a world that starts past the election,
and resuming a game whose election has been held."""

import builtins

import pytest
from pydantic import ValidationError

from hog_sim.core.config import GameConfig
from hog_sim.core.state import WorldState
from hog_sim.game.loop import Game
from hog_sim.game.persistence import SaveStore
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.population.popularity import ElectionResult, run_election
from hog_sim.ui import cli
from hog_sim.world.seed.toy import toy_world


def plugins():
    return CannedScenarios(0), KeywordInterpreter(), EngineForecaster()


def test_a_world_where_nobody_votes_is_rejected() -> None:
    data = toy_world().model_dump()
    for group in data["groups"].values():
        group["turnout"] = 0.0
    with pytest.raises(ValidationError, match="someone must vote"):
        WorldState.model_validate(data)


def test_an_election_needs_groups() -> None:
    world = toy_world().model_copy(update={"groups": {}})
    with pytest.raises(ValueError, match="population groups"):
        run_election(world)


def test_a_tied_house_is_hung_not_a_defeat() -> None:
    tied = ElectionResult(vote_share=0.5, seats=325, total_seats=650)
    assert not tied.majority and tied.hung
    assert "Hung parliament" in tied.verdict
    lost = ElectionResult(vote_share=0.4, seats=200, total_seats=650)
    assert not lost.hung and "opposition wins" in lost.verdict
    assert ElectionResult(vote_share=0.6, seats=500, total_seats=650).verdict.startswith("You win")


def test_a_world_starting_at_or_past_the_election_is_rejected() -> None:
    world = toy_world()
    world.turn = 3
    with pytest.raises(ValueError, match="at or past the election"):
        Game(GameConfig(election_turn=3), world, *plugins())


def test_resuming_a_finished_game_shows_the_result(tmp_path, monkeypatch, capsys) -> None:
    save = tmp_path / "g.db"
    store = SaveStore(save)
    config = GameConfig(election_turn=2, k_draws=5)
    game = Game(config, toy_world(), *plugins())
    game_id = store.new_game(config, game.start)
    for response in ("Do nothing", "Do nothing"):
        store.save_turn(game_id, game.play_turn(response))
    store.close()
    election = game.history[-1].election
    assert election is not None

    def no_input(prompt=""):
        raise AssertionError("a finished game asked for a turn")

    monkeypatch.setattr(builtins, "input", no_input)
    cli.main(["--offline", "--resume", "--save", str(save)])
    out = capsys.readouterr().out
    assert "This game is over." in out
    assert f"{election.seats}/{election.total_seats} seats" in out
    assert f"Start a new game with: hog-sim --save {save}" in out
