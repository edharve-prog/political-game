"""Storylines come to a head (backlog story SD-9)."""

import re
from types import SimpleNamespace

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, Scenario
from hog_sim.game.loop import Game, replay
from hog_sim.game.stubs import EngineForecaster
from hog_sim.llm.adapters import LLMInterpreter, LLMScenarioSource
from hog_sim.llm.client import FakeClient, LLMRequest
from hog_sim.llm.prompts import outcomes as outcomes_prompt
from hog_sim.llm.scenario_gen import ScenarioDraft, check_draft
from hog_sim.llm.summary import summarise_state
from hog_sim.world.seed.toy import toy_world
from hog_sim.world.storylines import (
    FINAL_STAGE,
    MAX_AGE,
    advance_storylines,
    final_storylines,
    must_open_new,
    storylines_text,
)

QUIET = Outcome(narrative="Talks dragged on.", probability=1.0)


def lead(storyline: str, title: str = "Gas strike") -> Scenario:
    return Scenario(
        title=title,
        briefing="...",
        affected_nodes=["sector:energy"],
        urgency=0.6,
        storyline=storyline,
    )


def test_a_storyline_at_its_final_stage_ends_whatever_the_outcome_says() -> None:
    state = toy_world()
    for turn in range(FINAL_STAGE - 1):
        state = advance_storylines(state, turn, lead("gas"), QUIET)
    state.turn = FINAL_STAGE - 1
    assert final_storylines(state) == ["gas"]
    assert "FINAL STAGE" in storylines_text(state)

    state = advance_storylines(state, FINAL_STAGE - 1, lead("gas"), QUIET)
    gas = state.storylines["gas"]
    assert gas.stage == FINAL_STAGE and not gas.open
    assert gas.history[-1] == f"Turn {FINAL_STAGE - 1}: came to a head"


def test_an_old_storyline_is_final_even_at_a_low_stage() -> None:
    state = advance_storylines(toy_world(), 0, lead("gas"), QUIET)
    state.turn = MAX_AGE - 1
    assert final_storylines(state) == []
    state.turn = MAX_AGE
    assert final_storylines(state) == ["gas"]


def test_must_open_new_after_two_continued_leads() -> None:
    state = toy_world()
    state = advance_storylines(state, 0, lead("gas"), QUIET)
    state = advance_storylines(state, 1, lead("gas"), QUIET)
    state = advance_storylines(state, 2, lead("gas"), QUIET)

    def record(turn):
        return SimpleNamespace(turn=turn, scenario=lead("gas"))

    assert not must_open_new(state, [record(0), record(1)])  # turn 0 opened it
    assert must_open_new(state, [record(0), record(1), record(2)])


def draft(lead_story: str, *side_stories: str) -> ScenarioDraft:
    side = [
        {
            "title": f"Side {i}",
            "category": "housing",
            "briefing": "...",
            "affected_nodes": ["sector:housing"],
            "urgency": 0.3,
            "storyline": s,
        }
        for i, s in enumerate(side_stories)
    ]
    return ScenarioDraft(
        title="Energy row",
        category="energy",
        storyline=lead_story,
        briefing="...",
        affected_nodes=["sector:energy"],
        urgency=0.6,
        suggested_options=["A", "B"],
        stakeholder_positions=[
            {"node": "group:pensioners", "stance": 0.5, "statement": "Help."},
            {"node": "sector:energy", "stance": -0.5, "statement": "No."},
        ],
        secondary=side,
    )


def test_check_draft_applies_the_lifespan_rules() -> None:
    state = toy_world()
    for name in ("a", "b", "c"):
        state = advance_storylines(state, 0, lead(name, title=name), QUIET)
    stories = list(state.storylines.values())
    catalogue = summarise_state(state).catalogue

    def problems(d, final=(), open_new=False):
        return " ".join(check_draft(d, catalogue, stories, final, open_new))

    assert problems(draft("a", "b", "new")) == ""
    assert "at most one secondary" in problems(draft("a", "b", "c"))
    assert "final stage" in problems(draft("a", "b"), final=["b"])
    assert problems(draft("b", "new"), final=["b"]) == ""
    assert "open a 'new' one" in problems(draft("a", "new"), open_new=True)
    assert problems(draft("new", "a"), open_new=True) == ""


def test_outcome_prompt_says_when_the_story_ends() -> None:
    text = outcomes_prompt.render("S", "Scenario", "Actions", "Engine", 3, ending=True)
    assert outcomes_prompt.ENDING in text
    assert outcomes_prompt.ENDING not in outcomes_prompt.render("S", "Sc", "A", "E", 3)


def greedy_game(turns: int):
    """A scenario writer that always continues the top open storyline when it may."""

    def responder(request: LLMRequest) -> dict:
        if request.schema_name != "ScenarioDraft":
            return {"actions": [{"kind": "tax", "target": "sector:energy", "magnitude": 0.3}]}
        prompt = request.messages[0].content
        turn = int(re.search(r"Turn (\d+)", prompt).group(1)) if "Turn " in prompt else 0
        ids = re.findall(r"^- (\S+-t\d+) \[", prompt, re.M)
        story = "new" if "must open a new one" in prompt or not ids else ids[0]
        return {
            **draft(story, "new").model_dump(),
            "title": f"Energy row {turn}",
            "secondary": [
                {
                    "title": f"Grumble {turn}",
                    "category": "housing",
                    "briefing": "...",
                    "affected_nodes": ["sector:housing"],
                    "urgency": 0.2,
                    "storyline": "new",
                }
            ],
        }

    client = FakeClient(responder=responder)
    config = GameConfig(election_turn=turns + 2, k_draws=10)
    world = toy_world()
    game = Game(
        config,
        world,
        LLMScenarioSource(client, calendar=False),
        LLMInterpreter(client),
        EngineForecaster(),
    )
    for _ in range(turns):
        game.play_turn("Windfall tax on energy firms")
    return game, world, config


def test_thirty_turns_of_a_greedy_writer_still_turn_over_storylines() -> None:
    game, world, config = greedy_game(30)
    leads = [r.scenario.storyline for r in game.history]
    assert len(set(leads)) >= 10
    assert max(leads.count(s) for s in set(leads)) <= FINAL_STAGE
    assert all(s.stage <= FINAL_STAGE for s in game.state.storylines.values() if not s.open)
    assert replay(world, config, game.history) == game.state
