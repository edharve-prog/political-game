"""Storylines that run across turns (backlog story SD-1)."""

import re

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.game.loop import Game, replay
from hog_sim.game.persistence import SaveStore
from hog_sim.game.stubs import EngineForecaster
from hog_sim.llm.adapters import LLMInterpreter, LLMScenarioSource
from hog_sim.llm.client import FakeClient, LLMRequest
from hog_sim.world.seed.toy import toy_world
from hog_sim.world.storylines import ESCALATE_AFTER, advance_storylines, storylines_text


def scenario(title: str, storyline: str | None, urgency: float = 0.5) -> Scenario:
    return Scenario(
        title=title,
        briefing="...",
        affected_nodes=["sector:energy"],
        urgency=urgency,
        category="energy",
        storyline=storyline,
    )


def outcome(narrative: str = "It went ahead.", resolves: bool = False) -> Outcome:
    return Outcome(narrative=narrative, probability=1.0, resolves_storyline=resolves)


def test_storyline_opens_continues_escalates_and_closes() -> None:
    state = toy_world()
    state = advance_storylines(state, 0, scenario("Gas strike ballot", "gas"), outcome())
    state = advance_storylines(state, 1, scenario("Rail fares", "rail"), outcome())
    gas = state.storylines["gas"]
    assert (gas.stage, gas.open, gas.opened_turn) == (1, True, 0)

    # Left alone: escalates once it has been idle ESCALATE_AFTER turns, never closes.
    for turn in range(2, 2 + ESCALATE_AFTER):
        state = advance_storylines(state, turn, scenario("Rail fares", "rail"), outcome())
    gas = state.storylines["gas"]
    assert gas.stage == 2 and gas.open and gas.pressure > 0.5
    assert "escalated while left alone" in gas.history[-1]

    # Continued, then closed only by an outcome that resolves it.
    turn = 2 + ESCALATE_AFTER
    state = advance_storylines(state, turn, scenario("Gas strike called", "gas"), outcome())
    assert state.storylines["gas"].stage == 3 and state.storylines["gas"].open
    state = advance_storylines(
        state, turn + 1, scenario("Gas deal signed", "gas"), outcome("Deal.", resolves=True)
    )
    assert not state.storylines["gas"].open
    assert "gas" not in storylines_text(state) and "rail" in storylines_text(state)


def test_scenario_without_storyline_changes_nothing() -> None:
    state = toy_world()
    assert advance_storylines(state, 0, scenario("One-off", None), outcome()) is state


def claude_game(turns: int):
    """A FakeClient game whose scenario writer continues the top open storyline two turns
    in three, and otherwise opens a new one."""

    def responder(request: LLMRequest) -> dict:
        if request.schema_name != "ScenarioDraft":
            action = PolicyAction(kind="tax", target="sector:energy", magnitude=0.3)
            return {"actions": [action.model_dump()]}
        prompt = request.messages[0].content
        turn = int(re.search(r"Turn (\d+)", prompt).group(1)) if "Turn " in prompt else 0
        ids = re.findall(r"^- (\S+-t\d+) \[", prompt, re.M)
        continuing = ids and turn % 3
        return {
            "title": f"Energy row, part {turn}" if continuing else f"New trouble {turn}",
            "category": "energy",
            "storyline": ids[0] if continuing else "new",
            "briefing": "Suppliers are struggling.",
            "affected_nodes": ["sector:energy", "group:pensioners"],
            "urgency": 0.6,
            "suggested_options": ["Tax windfall profits", "Do nothing"],
            "stakeholder_positions": [
                {"node": "group:pensioners", "stance": 0.8, "statement": "Help us."},
                {"node": "sector:energy", "stance": -0.5, "statement": "Hands off."},
            ],
        }

    client = FakeClient(responder=responder)
    config = GameConfig(election_turn=turns + 4, k_draws=10)
    world = toy_world()
    game = Game(
        config, world, LLMScenarioSource(client), LLMInterpreter(client), EngineForecaster()
    )
    for _ in range(turns):
        game.play_turn("Windfall tax on energy firms")
    return game, client, world, config


def test_twenty_turn_game_has_a_storyline_that_spans_turns() -> None:
    game, client, world, config = claude_game(20)
    spans = {}
    for record in game.history:
        if record.scenario.storyline:
            spans.setdefault(record.scenario.storyline, []).append(record.turn)
    longest_id, turns = max(spans.items(), key=lambda kv: len(kv[1]))
    assert len(turns) >= 3

    # The scenario prompt for a later stage carries the earlier stages.
    prompts = [r.messages[0].content for r in client.requests if r.schema_name == "ScenarioDraft"]
    third = prompts[turns[2]]
    story = game.state.storylines[longest_id]
    assert "Open storylines" in third and longest_id in third
    assert f"Turn {turns[0]}, stage 1:" in third and f"Turn {turns[1]}, stage 2:" in third
    assert story.stage >= 3

    # Storylines live in the state, so replay reproduces them.
    assert replay(world, config, game.history) == game.state


def test_storylines_survive_save_and_resume(tmp_path) -> None:
    game, _, world, config = claude_game(4)
    store = SaveStore(tmp_path / "games.db")
    game_id = store.new_game(config, world)
    for record in game.history:
        store.save_turn(game_id, record)
    _, _, records = store.load(game_id)
    assert records[-1].state_after.storylines == game.state.storylines
    assert WorldState.from_json(game.state.to_json()).storylines == game.state.storylines
