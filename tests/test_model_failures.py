"""Review of 13 Oct 2026, E22: a provider error crashed the turn with a traceback, and a
failure while writing the outcome lost the response the player had just confirmed."""

import builtins

from hog_sim.game.persistence import SaveStore
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.llm.client import FakeClient, LLMUnavailable
from hog_sim.ui import cli


class FailingForecaster(EngineForecaster):
    def forecast(self, *args, **kwargs):
        raise LLMUnavailable("the API had a server error (529)")


class FailingInterpreter(KeywordInterpreter):
    def interpret(self, *args, **kwargs):
        raise LLMUnavailable("rate limit")


class FailingScenarios(CannedScenarios):
    def next_scenario(self, *args, **kwargs):
        raise LLMUnavailable("outage")


def play(tmp_path, monkeypatch, plugins, answers, resume=False):
    monkeypatch.setattr(
        "hog_sim.llm.providers.make_client", lambda provider: (FakeClient([]), "api", None)
    )
    monkeypatch.setattr("hog_sim.game.llm_plugins.llm_plugins", lambda client, **kw: plugins)
    feed = iter(answers)

    def answer(prompt=""):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", answer)
    save = str(tmp_path / "g.db")
    cli.main(["--save", save, "--llm-log", "off", *(["--resume"] if resume else [])])
    store = SaveStore(save)
    game_id = store.latest_game()
    records = store.load(game_id)[2]
    return store, game_id, records


def test_a_failed_outcome_offers_an_offline_reading_and_keeps_the_response(
    tmp_path, monkeypatch, capsys
) -> None:
    plugins = (CannedScenarios(0, calendar=False), KeywordInterpreter(), FailingForecaster())
    _, _, records = play(tmp_path, monkeypatch, plugins, ["Do nothing", "", "r", "o"])
    out = capsys.readouterr().out
    assert out.count("The model failed while writing what happened") == 2
    assert "Traceback" not in out
    assert len(records) == 1 and records[0].response == "Do nothing"


def test_failed_reading_and_scenario_fall_back_offline(tmp_path, monkeypatch, capsys) -> None:
    plugins = (FailingScenarios(0, calendar=False), FailingInterpreter(), EngineForecaster())
    _, _, records = play(tmp_path, monkeypatch, plugins, ["o", "Do nothing", "o", ""])
    out = capsys.readouterr().out
    assert "The model failed while writing this turn's scenario" in out
    assert "The model failed while reading your response" in out
    assert len(records) == 1


def test_a_confirmed_response_survives_quitting_and_is_offered_on_resume(
    tmp_path, monkeypatch, capsys
) -> None:
    failing = (CannedScenarios(0, calendar=False), KeywordInterpreter(), FailingForecaster())
    store, game_id, records = play(tmp_path, monkeypatch, failing, ["Cut fuel duty", ""])
    assert records == []
    assert store.pending_proposal(game_id, 0).response == "Cut fuel duty"
    store.close()

    working = (CannedScenarios(0, calendar=False), KeywordInterpreter(), EngineForecaster())
    _, _, records = play(tmp_path, monkeypatch, working, [""], resume=True)
    assert "Your confirmed response from last time is waiting" in capsys.readouterr().out
    assert len(records) == 1 and records[0].response == "Cut fuel duty"
