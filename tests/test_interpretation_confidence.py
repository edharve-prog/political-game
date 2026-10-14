"""D6: each action carries the words it came from and a confidence, and the review screen
flags guesses."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import PolicyAction, Reading, Scenario
from hog_sim.game.loop import Game, Proposal
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.knowledge.entries import InterpretationEntry, Provenance, scenario_id
from hog_sim.knowledge.offline import StoredInterpreter
from hog_sim.knowledge.store import KnowledgeStore
from hog_sim.llm.adapters import LLMInterpreter
from hog_sim.llm.client import FakeClient
from hog_sim.ui import cli
from hog_sim.ui.builder import edit_actions
from hog_sim.world.seed.toy import toy_world

SCENARIO = Scenario(title="Waiting lists", briefing="", affected_nodes=["sector:public"], urgency=0)
SPEND = PolicyAction(kind="spend", target="sector:public", magnitude=0.5)
HOUSING = Scenario(title="Housing", briefing="", affected_nodes=["sector:housing"], urgency=0.5)


class OnlyHousing:
    def next_scenario(self, state, history):
        return HOUSING


def store_with(tmp_path, response: str, actions: list[PolicyAction]) -> KnowledgeStore:
    store = KnowledgeStore(tmp_path / "game.db")
    sid = scenario_id(SCENARIO)
    entry = InterpretationEntry(
        scenario_id=sid,
        scenario_title=SCENARIO.title,
        response=response,
        actions=actions,
        provenance=Provenance(game_id="g1", turn=0),
    )
    store.put("interpretation", f"{sid}|{response}", entry, sid)
    return store


def readings(text: str) -> list[Reading]:
    interpreter = KeywordInterpreter()
    interpreter.interpret(text, toy_world(), HOUSING)
    return interpreter.last_readings()


def test_keyword_readings_quote_each_option_and_flag_guesses() -> None:
    first, second, third = readings(
        "Option: Tax the banks\nOption: Spend more\nAlso: crack down on rents"
    )
    assert (first.span, first.confidence, first.fallback) == ("Tax the banks", 0.8, False)
    # No target named: aimed at the scenario's own sector, and flagged.
    assert (second.target, second.fallback, second.doubtful) == ("sector:housing", True, True)
    assert second.span == "Spend more"
    # A speech nobody asked for is only the fallback kind.
    assert third.kind == "communicate" and third.confidence == 0.5 and third.doubtful


def test_review_screen_shows_spans_and_checks() -> None:
    game = Game(
        GameConfig(election_turn=6, k_draws=10),
        toy_world(),
        OnlyHousing(),
        KeywordInterpreter(),
        EngineForecaster(),
    )
    proposal = game.propose("Option: Tax the banks\nOption: Subsidise something")
    text = cli._proposal_text(proposal, game.state)
    assert 'from "Tax the banks"' in text
    assert "CHECK: no target was named, so it was guessed from the scenario" in text
    # An action the player edits is theirs now: no reading, no flag.
    edited = game.revise(proposal, edit_actions(proposal.actions, "2 size 0.2"))
    assert edited.reading(1) is not None
    retargeted = proposal.actions[:1] + [
        proposal.actions[1].model_copy(update={"target": "sector:public"})
    ]
    assert game.revise(proposal, retargeted).reading(1) is None


def test_llm_readings_follow_their_actions_through_repair() -> None:
    state = toy_world()
    scenario = CannedScenarios(0).next_scenario(state, [])
    client = FakeClient(
        [
            {
                "actions": [
                    {"kind": "do_nothing", "target": "country:uk", "magnitude": 0},
                    {"kind": "spend", "target": "housing", "magnitude": 0.5},
                ],
                "readings": [
                    {"action": 1, "span": "build homes", "confidence": 0.55},
                    {"action": 5, "span": "nonsense", "confidence": 1},
                ],
            }
        ]
    )
    interpreter = LLMInterpreter(client)
    [action] = interpreter.interpret("Wait, no: build homes", state, scenario)
    [reading] = interpreter.last_readings()
    assert reading.fits(action) and reading.span == "build homes" and reading.doubtful


def test_a_stored_reading_reports_its_match_score(tmp_path) -> None:
    store = store_with(tmp_path, "Increase public spending", [SPEND])
    interpreter = StoredInterpreter(store, KeywordInterpreter())
    interpreter.interpret("Increase public spending!", toy_world(), SCENARIO)
    [reading] = interpreter.last_readings()
    assert reading.fits(SPEND) and reading.confidence == pytest.approx(1.0)
    interpreter.interpret("Subsidise something", toy_world(), SCENARIO)
    assert interpreter.last_source == "fallback"
    assert interpreter.last_readings()[0].fallback


def test_reading_matches_repeated_actions_in_order() -> None:
    a = SPEND
    proposal = Proposal(
        response="",
        requested=[a, a],
        actions=[a, a],
        readings=[
            Reading(kind="spend", target="sector:public", span="one"),
            Reading(kind="spend", target="sector:public", span="two"),
        ],
    )
    assert [proposal.reading(i).span for i in (0, 1)] == ["one", "two"]
    assert proposal.reading(2) is None
