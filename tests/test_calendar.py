"""SD-6: the political calendar sets the lead issue on fixed turns."""

from types import SimpleNamespace

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Storyline
from hog_sim.game.loop import Game, replay
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.llm.adapters import LLMScenarioSource
from hog_sim.llm.client import FakeClient
from hog_sim.llm.summary import summarise_state
from hog_sim.ui import cli
from hog_sim.world.calendar import calendar_event, fill, upcoming
from hog_sim.world.seed.toy import toy_world
from hog_sim.world.storylines import must_open_new


def test_events_fall_on_their_turns_and_never_on_turn_zero() -> None:
    kinds = {t: e.kind for t in range(25) if (e := calendar_event(t)) is not None}
    assert kinds == {
        3: "budget",
        6: "summit",
        9: "conference",
        12: "by_election",
        15: "budget",
        18: "summit",
        21: "conference",
        24: "by_election",
    }
    assert [(t, e.kind) for t, e in upcoming(1)] == [(3, "budget"), (6, "summit")]


def test_a_by_election_is_fought_among_the_least_happy_group() -> None:
    state = toy_world()
    state.groups["group:pensioners"].approval = 0.1
    event = fill(calendar_event(12), state)
    assert "pensioners" in event.briefing
    assert "group:pensioners" in event.affected_nodes
    assert any("pensioners" in o for o in event.options)


def test_offline_calendar_turns_lead_with_the_event_and_replay() -> None:
    config = GameConfig(seed=0, election_turn=8, k_draws=20)
    game = Game(config, toy_world(), CannedScenarios(0), KeywordInterpreter(), EngineForecaster())
    for _ in range(7):
        game.play_turn("Invest in housebuilding")
    budget = game.history[3].scenario
    assert budget.title == "The Budget" and budget.source == "scheduled"
    assert "Raise taxes to fund public services" in budget.suggested_options
    assert game.history[6].scenario.title == "International summit"
    assert game.history[2].scenario.source == "generated"
    assert replay(game.start, game.config, game.history) == game.state
    plain = CannedScenarios(0, calendar=False).next_scenario(game.history[3].state_after, [])
    assert plain.source == "generated"


def test_the_dashboard_and_briefing_show_what_is_coming() -> None:
    state = toy_world()
    state.turn = 1
    assert "Turn 3: The Budget\n  Turn 6: International summit" in cli._dashboard(state, state)
    assert "- Turn 3: The Budget (in 2 turns)" in summarise_state(state).to_prompt()


def llm_draft(storyline="new"):
    return {
        "title": "Chancellor's Budget squeezed by gilt yields",
        "category": "housing",
        "storyline": storyline,
        "briefing": "The Budget lands with borrowing costs rising.",
        "affected_nodes": ["indicator:deficit", "sector:public"],
        "urgency": 0.8,
        "suggested_options": ["Something else", "Another thing"],
        "stakeholder_positions": [
            {"node": "group:business", "stance": 0.5, "statement": "Cut our taxes."},
            {"node": "group:public_workers", "stance": 0.8, "statement": "Fund us."},
        ],
        "secondary": [
            {
                "title": "Rents climb again",
                "category": "housing",
                "briefing": "Young renters face another rise.",
                "affected_nodes": ["sector:housing"],
                "urgency": 0.4,
                "storyline": "new",
            }
        ],
    }


def test_on_a_calendar_turn_claude_writes_the_briefing_and_the_event_sets_the_rest() -> None:
    state = toy_world()
    state.turn = 3
    client = FakeClient([llm_draft()])
    scenario = LLMScenarioSource(client).next_scenario(state, [])
    prompt = client.requests[0].messages[0].content
    assert "This turn is on the political calendar: The Budget" in prompt
    assert scenario.source == "scheduled" and scenario.storyline is None
    assert scenario.category == "economy"
    assert scenario.suggested_options == fill(calendar_event(3), state).options
    assert scenario.briefing == "The Budget lands with borrowing costs rising."

    state.turn = 4
    client = FakeClient([llm_draft()])
    ordinary = LLMScenarioSource(client).next_scenario(state, [])
    assert ordinary.source == "generated" and ordinary.storyline is not None
    assert "political calendar: " not in client.requests[0].messages[0].content


def test_calendar_turns_do_not_count_against_the_new_storyline_rule() -> None:
    state = toy_world()

    def record(turn, storyline, source="generated"):
        scenario = SimpleNamespace(storyline=storyline, source=source)
        return SimpleNamespace(turn=turn, scenario=scenario)

    state.storylines["gas-t0"] = Storyline(
        id="gas-t0", title="Gas", opened_turn=0, last_turn=2, last_addressed=2
    )
    history = [
        record(0, "gas-t0"),
        record(1, "gas-t0"),
        record(2, None, "scheduled"),
        record(3, "gas-t0"),
    ]
    assert must_open_new(state, history)


def test_the_offline_library_reaches_the_calendar(monkeypatch) -> None:
    """E19: the CLI and the stability runs wrap the library in the calendar."""
    from hog_sim.content.library import ScenarioLibrary
    from hog_sim.game import stability
    from hog_sim.world.calendar import OnCalendar

    config = GameConfig(seed=1, election_turn=5, k_draws=10)
    source = OnCalendar(ScenarioLibrary.load(seed=1))
    game = Game(config, toy_world(), source, KeywordInterpreter(), EngineForecaster())
    for _ in range(4):
        game.play_turn("do nothing")
    titles = [r.scenario.title for r in game.history]
    assert titles[3] == "The Budget" and "The Budget" not in titles[:3]
    assert source.seed == 1  # anything else is the library's own

    seen: list[str] = []
    real = Game.play_turn

    def spy(self, text):
        seen.append(self.scenario.title)
        return real(self, text)

    monkeypatch.setattr(Game, "play_turn", spy)
    stability.play("do nothing", seed=1, turns=4)
    assert seen[3] == "The Budget"


def test_offline_cli_shows_the_budget_on_turn_3(tmp_path, monkeypatch, capsys) -> None:
    import builtins

    answers = iter(["do nothing", ""] * 3)

    def fake_input(prompt=""):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", fake_input)
    cli.main(["--offline", "--save", str(tmp_path / "g.db")])
    # The Budget's options are listed only when it is the lead issue.
    assert "Raise taxes to fund public services" in capsys.readouterr().out
