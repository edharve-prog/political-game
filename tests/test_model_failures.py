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


# O10 gaps from the final review (16 Oct 2026): FR-13 to FR-16.


class FakeAdvisers:
    """Advisers that answer without the model; costing their options is what fails."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def ask(self, question, state, scenario):
        from hog_sim.llm.advisers import AdvisedOption

        return [
            AdvisedOption(option="Freeze fuel duty", trade_off="costs revenue"),
            AdvisedOption(option="Hold a review", trade_off="looks slow"),
        ]


class BudgetedInterpreter(KeywordInterpreter):
    """Reads with the model through a call budget of one call per turn."""

    def __init__(self) -> None:
        from hog_sim.llm.budget import BudgetedClient

        self.client = BudgetedClient(FakeClient(responder=lambda r: "ok"), limit=1)
        self.client.turn_of = lambda: 0

    def interpret(self, *args, **kwargs):
        self.client.complete(request())
        return super().interpret(*args, **kwargs)


def test_costing_advice_when_the_model_fails_offers_the_options_uncosted(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "Advisers", FakeAdvisers)
    plugins = (CannedScenarios(0, calendar=False), FailingInterpreter(), EngineForecaster())
    _, _, records = play(tmp_path, monkeypatch, plugins, ["advise something cheaper", "why"])
    out = " ".join(capsys.readouterr().out.split())
    assert "Traceback" not in out
    assert "The model failed while costing these options: rate limit" in out
    assert "Freeze fuel duty" in out and "Hold a review" in out
    assert out.count("Numbers: not costed, because the model failed") == 2
    # The game carried on at the response prompt: the turn was neither played nor lost.
    assert "Nothing has happened yet" in out.split("costing these options")[1]
    assert records == []


def test_running_out_of_call_budget_while_costing_keeps_the_turn(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "Advisers", FakeAdvisers)
    plugins = (CannedScenarios(0, calendar=False), BudgetedInterpreter(), EngineForecaster())
    _, _, records = play(tmp_path, monkeypatch, plugins, ["advise something cheaper", "why"])
    out = " ".join(capsys.readouterr().out.split())
    assert "Traceback" not in out
    # The first option used the turn's one call; the second ran out of budget.
    assert out.count("Numbers: not costed, because the model failed") == 1
    assert "used its 1 model calls" in out
    assert "Nothing has happened yet" in out
    assert records == []


def test_the_fallback_scenario_keeps_to_the_calendar(tmp_path, monkeypatch) -> None:
    from hog_sim.world.calendar import calendar_event
    from hog_sim.world.seed.toy import toy_world

    event = calendar_event(3)
    assert event is not None
    monkeypatch.setattr(cli, "toy_world", lambda: toy_world().model_copy(update={"turn": 3}))
    plugins = (FailingScenarios(0, calendar=False), KeywordInterpreter(), EngineForecaster())
    _, _, records = play(tmp_path, monkeypatch, plugins, ["o", "Do nothing", ""])
    assert len(records) == 1
    assert records[0].scenario.title == event.title


def test_offline_fallbacks_are_marked_and_not_filed_as_claudes_work(tmp_path, monkeypatch) -> None:
    from hog_sim.knowledge.store import KnowledgeStore

    plugins = (FailingScenarios(0, calendar=False), FailingInterpreter(), FailingForecaster())
    _, _, records = play(tmp_path, monkeypatch, plugins, ["o", "Cut fuel duty", "o", "", "o"])
    assert len(records) == 1
    assert records[0].offline == ["forecaster", "interpreter", "scenarios"]
    knowledge = KnowledgeStore(str(tmp_path / "g.db"))
    stats = knowledge.stats()
    knowledge.close()
    assert stats["scenario"] == stats["interpretation"] == stats["outcome"] == 0
    assert stats.get("graph_change", 0) == 0


def test_claudes_steps_are_still_filed_beside_an_offline_one(tmp_path, monkeypatch) -> None:
    from hog_sim.knowledge.store import KnowledgeStore

    plugins = (CannedScenarios(0, calendar=False), FailingInterpreter(), EngineForecaster())
    _, _, records = play(tmp_path, monkeypatch, plugins, ["Cut fuel duty", "o", ""])
    assert len(records) == 1 and records[0].offline == ["interpreter"]
    knowledge = KnowledgeStore(str(tmp_path / "g.db"))
    stats = knowledge.stats()
    knowledge.close()
    assert stats["interpretation"] == 0 and stats["outcome"] == 1


def test_a_bad_call_budget_in_the_environment_falls_back_to_the_default(
    tmp_path, monkeypatch, capsys
) -> None:
    from hog_sim.llm import budget

    monkeypatch.setenv(budget.BUDGET_ENV, "lots")
    seen = {}

    class Spy(budget.BudgetedClient):
        def __init__(self, inner, limit=budget.DEFAULT_LIMIT) -> None:
            seen["limit"] = limit
            super().__init__(inner, limit)

    monkeypatch.setattr(budget, "BudgetedClient", Spy)
    plugins = (CannedScenarios(0, calendar=False), KeywordInterpreter(), EngineForecaster())
    play(tmp_path, monkeypatch, plugins, [])
    out = capsys.readouterr().out
    assert f"{budget.BUDGET_ENV}='lots' is not a whole number" in out
    assert seen["limit"] == budget.DEFAULT_LIMIT
