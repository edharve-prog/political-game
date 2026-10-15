"""Advisers with numbers (review of 13 October, D8): each option is read, limited and
previewed before it is shown, and options the limits would block are not offered."""

import pytest

from hog_sim.content.library import ScenarioLibrary
from hog_sim.core.config import GameConfig
from hog_sim.core.models import PolicyAction
from hog_sim.game.advice import cost_option, cost_options, costing_line
from hog_sim.game.interfaces import NeedsClarification
from hog_sim.game.loop import Game
from hog_sim.game.preview import preview
from hog_sim.game.stubs import EngineForecaster
from hog_sim.llm.advisers import Advisers
from hog_sim.llm.client import FakeClient
from hog_sim.policy.limits import capital_cost, lifetime_cost
from hog_sim.ui import cli
from hog_sim.world.seed.toy import toy_world

SPEND = PolicyAction(kind="spend", target="sector:public", magnitude=0.5, duration_turns=6)
SACK = PolicyAction(kind="appoint", target="institution:central_bank", magnitude=0.5)


class ByText:
    """Reads each option as the actions listed for it."""

    def __init__(self, readings: dict[str, list[PolicyAction]]) -> None:
        self.readings = readings

    def interpret(self, text, state, scenario):
        if text not in self.readings:
            raise NeedsClarification("What do you mean?")
        return [a.model_copy(deep=True) for a in self.readings[text]]


READINGS = {"Fund the NHS": [SPEND], "Sack the governor": [SACK], "Hold firm": []}


def game() -> Game:
    plugins = (ScenarioLibrary.load(seed=1), ByText(READINGS), EngineForecaster())
    return Game(GameConfig(election_turn=6), toy_world(), *plugins)


def test_an_option_carries_its_capital_deficit_and_vote() -> None:
    g = game()
    costed = cost_option(g, "Fund the NHS", "It borrows.")
    assert costed.blocked is None and costed.costing is not None
    c = costed.costing
    assert c.actions == g.propose("Fund the NHS").actions
    assert c.capital == pytest.approx(capital_cost(c.actions, g.state))
    assert c.deficit == pytest.approx(lifetime_cost(c.actions[0], g.state).deficit_turn_pts)
    assert c.deficit > 0
    seen = preview(g.start, g.state, g.scenario, c.actions, g.config)
    assert c.vote == pytest.approx(seen.package.vote - seen.nothing.vote)
    assert "political capital" in costing_line(c) and "% of GDP" in costing_line(c)


def test_blocked_options_are_not_offered_and_unreadable_ones_have_no_numbers() -> None:
    g = game()
    costed = cost_options(g, [("Sack the governor", ""), ("Hold firm", ""), ("Pray", "")])
    assert costed[0].blocked and "Blocked" in costed[0].blocked
    assert costed[1].blocked is None and costed[1].costing is None
    assert costed[2].blocked is None and costed[2].costing is None


def test_costing_changes_nothing() -> None:
    g = game()
    before = g.state
    cost_options(g, [("Fund the NHS", ""), ("Sack the governor", "")])
    assert g.state is before and g.history == []


def test_cli_shows_numbers_and_drops_blocked_options() -> None:
    g = game()
    n = len(g.scenario.suggested_options)
    client = FakeClient(
        [
            {
                "options": [
                    {"option": "Fund the NHS", "trade_off": "It borrows."},
                    {"option": "Sack the governor", "trade_off": "Markets wobble."},
                ]
            }
        ]
    )
    out = cli._advice(g, Advisers(client), "what else?")
    assert f"{n + 1}. Fund the NHS" in out and "Numbers:" in out
    assert "Not offered: Sack the governor" in out
    assert g.scenario.suggested_options[-1] == "Fund the NHS"
    assert "Sack the governor" not in g.scenario.suggested_options
