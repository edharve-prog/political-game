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


# O10: a per-turn call budget caps what one turn can spend on the model.


def request():
    from hog_sim.llm.client import LLMRequest

    return LLMRequest(
        model="m", system="s", messages=[], output_schema={}, schema_name="X", prompt_version="1"
    )


def test_the_call_budget_refuses_calls_over_the_cap_and_resets_each_turn() -> None:
    import pytest

    from hog_sim.llm.budget import BudgetedClient, LLMBudgetExceeded

    inner = FakeClient(responder=lambda r: "ok")
    turn = [0]
    client = BudgetedClient(inner, limit=2)
    client.turn_of = lambda: turn[0]
    client.complete(request())
    client.complete(request())
    with pytest.raises(LLMBudgetExceeded, match="used its 2 model calls"):
        client.complete(request())
    assert len(inner.requests) == 2  # the refused call was never sent
    client.extend()
    client.complete(request())
    turn[0] = 1
    client.complete(request())
    client.complete(request())
    assert len(inner.requests) == 5


def test_trying_again_over_budget_allows_more_calls(monkeypatch, capsys) -> None:
    from hog_sim.llm.budget import BudgetedClient

    client = BudgetedClient(FakeClient(responder=lambda r: "ok"), limit=1)
    client.complete(request())
    monkeypatch.setattr(builtins, "input", lambda prompt="": "r")
    reply = cli._guarded(
        None, "forecaster", "writing what happened", lambda: client.complete(request()), object()
    )
    assert reply.text == "ok"
    assert "used its 1 model calls" in " ".join(capsys.readouterr().out.split())


def test_no_cap_with_a_budget_of_zero() -> None:
    from hog_sim.llm.budget import BudgetedClient

    client = BudgetedClient(FakeClient(responder=lambda r: "ok"), limit=0)
    for _ in range(50):
        client.complete(request())
