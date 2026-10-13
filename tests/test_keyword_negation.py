"""Offline keyword reading never acts on the opposite of what the player said: refusals and
conditions ask for clarification, refused sackings sack nobody, and cutting a programme is a
cut (Ed's 2026-10-12 review, finding 3)."""

import pytest

from hog_sim.content.library import ScenarioLibrary
from hog_sim.core.models import Scenario
from hog_sim.game.interfaces import NeedsClarification
from hog_sim.game.stubs import _SCENARIOS, KeywordInterpreter
from hog_sim.world.calendar import CALENDAR
from hog_sim.world.seed.toy import toy_world

QUIET = Scenario(title="Quiet", briefing="", affected_nodes=[], urgency=0)
JOBS = "indicator:unemployment"


def read(text: str):
    interpreter = KeywordInterpreter()
    return interpreter, interpreter.interpret(text, toy_world(), QUIET)


@pytest.mark.parametrize(
    "text",
    [
        "Do not increase public spending",
        "Don't fund the hospitals",
        "We must never subsidise the banks",
        "Refuse to build more homes",
        "Fund hospitals instead of cutting taxes",
        "Raise taxes on business but don't spend it on the NHS",
        "If inflation rises, cut spending",
        "Unless the unions back down, hold firm",
    ],
)
def test_refusals_and_conditions_ask_instead_of_guessing(text) -> None:
    with pytest.raises(NeedsClarification):
        read(text)


@pytest.mark.parametrize(
    "text", ["Do not sack the Chancellor", "We won't fire the Chancellor", "I won't sack anyone"]
)
def test_a_refused_sacking_sacks_nobody(text) -> None:
    interpreter, actions = read(text)
    assert interpreter.last_sacked() == []
    assert [a.kind for a in actions] == ["do_nothing"]


def test_a_sacking_beside_a_refusal_to_sack_someone_else() -> None:
    interpreter, _ = read("Sack the Chancellor. Do not sack the Home Secretary.")
    assert interpreter.last_sacked() == ["person:chancellor"]


def test_mixed_clauses_keep_the_instruction_and_the_promise() -> None:
    interpreter, actions = read("Fund the hospitals but we will not raise taxes")
    assert [(a.kind, a.target, a.magnitude > 0) for a in actions] == [
        ("spend", "sector:public", True)
    ]
    assert [(p.kind, p.direction) for p in interpreter.last_pledges()] == [("tax", "up")]


def test_a_promise_is_still_a_promise() -> None:
    interpreter, actions = read("No cuts to schools")
    assert [a.kind for a in actions] == ["do_nothing"]
    assert [(p.kind, p.direction) for p in interpreter.last_pledges()] == [("spend", "down")]


@pytest.mark.parametrize(
    "text",
    ["Cut spending on furlough", "End the furlough scheme", "Wind down the wage subsidy"],
)
def test_cutting_a_programme_is_a_spending_cut(text) -> None:
    _, [action] = read(text)
    assert (action.kind, action.target, action.instrument) == ("spend", JOBS, "cut")
    assert action.instrument_sign == -1
    assert action.magnitude > 0  # without it, unemployment goes the other way


@pytest.mark.parametrize("text", ["Increase spending on furlough", "Launch a furlough scheme"])
def test_spending_more_on_a_programme_is_a_rise(text) -> None:
    _, [action] = read(text)
    assert (action.kind, action.target) == ("spend", JOBS)
    assert action.instrument_sign == 1
    assert action.magnitude < 0  # it is there to bring unemployment down


def test_every_suggested_option_still_reads() -> None:
    """The text bots play the scenarios' own options, so none of them may need asking about."""
    state = toy_world()
    options = [o for s in ScenarioLibrary.load().scenarios for o in s.suggested_options]
    options += [o for s in _SCENARIOS for o in s.suggested_options]
    options += [o for e in CALENDAR for o in e.options]
    assert len(options) > 200
    for option in options:
        KeywordInterpreter().interpret(option, state, QUIET)
