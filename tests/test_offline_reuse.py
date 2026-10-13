"""Offline reuse of stored interpretations: only for responses that mean the same thing,
with delivery, pledges and sackings read from the player's own words."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import PolicyAction, Scenario
from hog_sim.game.interfaces import NeedsClarification
from hog_sim.game.loop import Game
from hog_sim.game.stubs import EngineForecaster, KeywordInterpreter
from hog_sim.knowledge.entries import InterpretationEntry, Provenance, scenario_id
from hog_sim.knowledge.offline import StoredInterpreter, intent_key, same_request
from hog_sim.knowledge.store import KnowledgeStore
from hog_sim.world.seed.toy import toy_world

SCENARIO = Scenario(
    title="Hospital waiting lists grow",
    briefing="Waiting lists hit a record.",
    affected_nodes=["sector:public"],
    urgency=0.6,
)
SPEND = PolicyAction(kind="spend", target="sector:public", magnitude=0.5)


class OneScenario:
    def next_scenario(self, state, history):
        return SCENARIO.model_copy(deep=True)


def store_with(tmp_path, response: str, actions: list[PolicyAction]) -> KnowledgeStore:
    store = KnowledgeStore(tmp_path / "game.db")
    sid = scenario_id(SCENARIO)
    store.put(
        "interpretation",
        f"{sid}|{response}",
        InterpretationEntry(
            scenario_id=sid,
            scenario_title=SCENARIO.title,
            response=response,
            actions=actions,
            provenance=Provenance(game_id="g1", turn=0),
        ),
        sid,
    )
    return store


@pytest.mark.parametrize(
    ("stored", "text"),
    [
        ("Increase public spending", "Do not increase public spending"),
        ("Increase public spending", "Cut public spending"),
        ("Raise spending on hospitals by 5%", "Raise spending on hospitals by 10%"),
        ("Freeze NHS pay for two years", "Freeze NHS pay for three years"),
        ("Fund the NHS for a year", "Fund the NHS permanently"),
        ("We will fund the NHS", "We won't fund the NHS"),
    ],
)
def test_responses_that_differ_in_meaning_never_match(stored, text) -> None:
    assert same_request(stored, text) == 0.0


def test_rewordings_with_the_same_meaning_still_match() -> None:
    assert same_request("Increase public spending", "increase public spending!") == 1.0
    score = same_request(
        "Increase spending on hospitals by 5% this year",
        "Increase hospital spending by 5% this year",
    )
    assert score >= 0.6
    assert intent_key("cut taxes") == intent_key("cut the taxes")


def test_negated_response_falls_back_instead_of_reusing_the_opposite(tmp_path) -> None:
    store = store_with(tmp_path, "Increase public spending", [SPEND])
    interpreter = StoredInterpreter(store, KeywordInterpreter())
    # Keyword matching can't read a refusal either, so it asks rather than guessing.
    with pytest.raises(NeedsClarification):
        interpreter.interpret("Do not increase public spending", toy_world(), SCENARIO)
    assert interpreter.last_source == "fallback"
    # The same words still reuse the stored reading.
    assert interpreter.interpret("Increase public spending", toy_world(), SCENARIO) == [SPEND]
    assert interpreter.last_source == "stored"


def play(interpreter, response: str):
    game = Game(
        GameConfig(election_turn=4, k_draws=10),
        toy_world(),
        OneScenario(),
        interpreter,
        EngineForecaster(),
    )
    return game.play_turn(response)


@pytest.mark.parametrize("stored", [True, False])
def test_offline_play_keeps_pledges_and_sackings(tmp_path, stored) -> None:
    response = "Fund the hospitals. We will not raise taxes. Sack the Chancellor."
    store = store_with(tmp_path, response if stored else "something else", [SPEND])
    interpreter = StoredInterpreter(store, KeywordInterpreter())
    record = play(interpreter, response)
    assert interpreter.last_source == ("stored" if stored else "fallback")
    assert [(p.kind, p.direction) for p in record.pledges] == [("tax", "up")]
    assert len(record.sacked) == 1
    # The plain keyword interpreter reads the same response the same way.
    plain = play(KeywordInterpreter(), response)
    assert record.pledges == plain.pledges and record.sacked == plain.sacked
