"""The phrases from the 13 October review's "offline keyword reading" table (E15 to E18)
read as the player meant them."""

import builtins

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Scenario
from hog_sim.game.loop import Game
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.ui import cli
from hog_sim.world.calendar import CALENDAR, calendar_event, fill, scheduled_scenario
from hog_sim.world.seed.toy import toy_world

QUIET = Scenario(title="Quiet", briefing="", affected_nodes=["sector:housing"], urgency=0)


def read(text: str, scenario: Scenario = QUIET, turn: int = 0):
    state = toy_world().model_copy(update={"turn": turn})
    interpreter = KeywordInterpreter()
    actions = interpreter.interpret(text, state, scenario)
    return interpreter, [(a.kind, a.target, round(a.magnitude, 2)) for a in actions]


def budget() -> Scenario:
    return scheduled_scenario(calendar_event(3), toy_world().model_copy(update={"turn": 3}))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Spend big to cut hospital waiting lists", [("spend", "sector:public", 0.9)]),
        ("Cut hospital waiting lists", [("spend", "sector:public", -0.5)]),
        ("Nothing less than a pay rise for nurses", [("spend", "sector:public", 0.5)]),
        ("Do nothing", [("do_nothing", "country:uk", 0)]),
        ("Wait it out", [("do_nothing", "country:uk", 0)]),
    ],
)
def test_waiting_lists_are_not_waiting(text, expected) -> None:
    """E15: "wait" and "nothing" only mean do nothing at the start of a clause."""
    assert read(text)[1] == expected


@pytest.mark.parametrize(
    "option, expected",
    [
        (
            "Raise taxes to fund public services",
            [("tax", "country:uk", 0.5), ("spend", "sector:public", 0.5)],
        ),
        ("Cut spending to bring borrowing down", [("spend", "sector:public", -0.5)]),
        ("Cut taxes for business to boost growth", [("tax", "group:business", -0.5)]),
        ("A steady Budget with no big changes", [("do_nothing", "country:uk", 0)]),
    ],
)
def test_the_budget_options_read_as_their_packages(option, expected) -> None:
    """E16: the Budget's own options map to known actions, typed or picked as an option."""
    assert read(option, budget(), turn=3)[1] == expected
    assert read(f"Option: {option}", budget(), turn=3)[1] == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Raise taxes to fund public services",
            [("tax", "country:uk", 0.5), ("spend", "sector:public", 0.5)],
        ),
        ("Cut taxes for business to boost growth", [("tax", "group:business", -0.5)]),
        ("Invest in growth", [("spend", "country:uk", 0.5)]),
        ("Help households with the economy", [("spend", "country:uk", 0.5)]),
    ],
)
def test_free_text_budget_wording_outside_the_budget(text, expected) -> None:
    """E16: business, growth, the economy and households are targets, and "X to fund Y" is
    a tax and a spend."""
    assert read(text)[1] == expected


def test_every_calendar_option_has_a_package_for_the_toy_world() -> None:
    for event in CALENDAR:
        state = toy_world().model_copy(update={"turn": event.offset or event.every})
        filled = fill(event, state)
        assert len(filled.packages) == len(filled.options)
        assert all(filled.packages), event.kind
        scenario = scheduled_scenario(event, state)
        for option, package in zip(filled.options, filled.packages, strict=True):
            actions = KeywordInterpreter().interpret(option, state, scenario)
            assert [(a.kind, a.target) for a in actions] == [(a.kind, a.target) for a in package]


def test_parliament_is_a_venue_not_a_target() -> None:
    """E17: announcing something in Parliament is about the thing announced."""
    interpreter, actions = read("Announce in Parliament a pay rise for nurses")
    assert actions == [("spend", "sector:public", 0.5)]
    assert interpreter.last_delivery().venue == "Parliament"
    assert read("Make a statement to the Commons on new homes")[1][0][1] == "sector:housing"
    # Whip, backbench and vote language still aims at the legislature.
    assert read("Whip the backbenchers into line")[1][0][1] == "institution:legislature"
    assert read("Win the vote on the bill")[1][0][1] == "institution:legislature"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Tighten fire safety rules in housing", [("regulate", "sector:housing", 0.5)]),
        (
            "Replace housing benefit with a new rent subsidy",
            [("spend", "sector:housing", 0.5)],
        ),
        (
            "Ask the Chancellor to replace the energy price cap with a subsidy",
            [("regulate", "indicator:energy_prices", -0.5)],
        ),
    ],
)
def test_everyday_words_sack_nobody(text, expected) -> None:
    """E18: "fire" and "replace" sack someone only when the minister is their object."""
    interpreter, actions = read(text)
    assert interpreter.last_sacked() == []
    assert actions == expected


@pytest.mark.parametrize(
    "text, who",
    [
        ("Sack the Housing Secretary", "person:housing_secretary"),
        ("Fire Owen Pryce", "person:housing_secretary"),
        ("Replace the Chancellor with someone sound", "person:chancellor"),
        ("Dismiss Calder", "person:chancellor"),
    ],
)
def test_a_real_sacking_still_sacks(text, who) -> None:
    interpreter, actions = read(text)
    assert interpreter.last_sacked() == [who]
    assert actions == [("do_nothing", "country:uk", 0)]


def test_a_sacking_keeps_the_rest_of_its_sentence() -> None:
    interpreter, actions = read("Sack the Housing Secretary and build more homes")
    assert interpreter.last_sacked() == ["person:housing_secretary"]
    assert actions == [("spend", "sector:housing", 0.5)]


def test_the_review_screen_can_drop_a_sacking() -> None:
    config = GameConfig(seed=0, election_turn=6, k_draws=10)
    game = Game(config, toy_world(), CannedScenarios(0), KeywordInterpreter(), EngineForecaster())
    proposal = game.propose("Sack the Chancellor and build more homes")
    assert proposal.sacked == ["person:chancellor"]
    kept = game.revise(proposal, proposal.actions, sacked=[])
    assert kept.sacked == [] and kept.actions == proposal.actions
    game.commit(kept)
    assert game.state.characters["person:chancellor"].active


def test_drop_sacking_at_the_review_step(monkeypatch) -> None:
    config = GameConfig(seed=0, election_turn=6, k_draws=10)
    game = Game(config, toy_world(), CannedScenarios(0), KeywordInterpreter(), EngineForecaster())
    answers = iter(["Sack the Chancellor and build more homes", "drop sacking", ""])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    proposal = cli._ask_for_turn(game)
    assert proposal is not None and proposal.sacked == []
    assert [a.target for a in proposal.actions] == ["sector:housing"]
