"""Ask advisers for new options (backlog story RB-3)."""

import builtins

from hog_sim.content.library import ScenarioLibrary
from hog_sim.core.config import GameConfig
from hog_sim.game.loop import Game
from hog_sim.game.stubs import EngineForecaster, KeywordInterpreter
from hog_sim.llm.advisers import Advisers
from hog_sim.llm.client import FakeClient
from hog_sim.ui import cli
from hog_sim.world.seed.toy import toy_world


def advice(*options: str) -> dict:
    return {"options": [{"option": o, "trade_off": f"{o} costs something."} for o in options]}


def offline_game() -> Game:
    world = toy_world()
    return Game(
        GameConfig(), world, ScenarioLibrary.load(seed=1), KeywordInterpreter(), EngineForecaster()
    )


def test_advisers_answer_the_question_and_retry_repeats() -> None:
    game = offline_game()
    existing = game.scenario.suggested_options[0]
    client = FakeClient([advice(existing, "Means-test it"), advice("Means-test it", "Delay it")])
    options = Advisers(client).ask("something cheaper", game.state, game.scenario)
    assert [o.option for o in options] == ["Means-test it", "Delay it"]
    assert len(client.requests) == 2
    prompt = client.requests[0].messages[0].content
    assert "The leader asks: something cheaper" in prompt and game.scenario.title in prompt
    assert "must be new" in client.requests[1].messages[-1].content


def test_add_options_keeps_the_turn_and_appends() -> None:
    game = offline_game()
    before = list(game.scenario.suggested_options)
    game.add_options(["Means-test it", before[0]])
    assert game.scenario.suggested_options == [*before, "Means-test it"]
    assert game.state.turn == 0 and game.history == []


def play_cli(monkeypatch, tmp_path, answers, advisers=None):
    answers = iter(answers)

    def fake_input(prompt=""):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", fake_input)
    game = offline_game()
    while True:
        try:
            proposal = cli._ask_for_turn(game, advisers)
        except EOFError:
            return game, None
        if proposal is not None:
            return game, game.commit(proposal)


def test_cli_advise_adds_numbered_options_that_can_be_picked(monkeypatch, tmp_path, capsys):
    client = FakeClient([advice("Freeze rail fares for a year", "Cut VAT on bills")])
    n = len(offline_game().scenario.suggested_options)
    game, record = play_cli(
        monkeypatch, tmp_path, ["advise something cheaper", f"{n + 1}", ""], Advisers(client)
    )
    out = capsys.readouterr().out
    assert f"  {n + 1}. Freeze rail fares for a year" in out and "Trade-off:" in out
    assert record is not None and "Option: Freeze rail fares for a year" in record.response
    assert record.turn == 0 and len(game.history) == 1


def test_cli_advise_offline_says_it_needs_claude(monkeypatch, tmp_path, capsys):
    play_cli(monkeypatch, tmp_path, ["advise what would the unions want"])
    assert "Advisers need Claude" in capsys.readouterr().out
