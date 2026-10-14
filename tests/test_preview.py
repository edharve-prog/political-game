"""Review of 13 Oct 2026, D2: before confirming, show what the package does by the election
against doing nothing, from the same engine and approval rules."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import PolicyAction
from hog_sim.game.loop import Game
from hog_sim.game.preview import QUIET_OUTCOME, QUIET_SCENARIO, preview, preview_lines
from hog_sim.game.stubs import CannedScenarios
from hog_sim.population.popularity import vote_intention
from hog_sim.ui import cli
from hog_sim.world.seed.toy import toy_world

PACKAGE = [
    PolicyAction(kind="spend", target="sector:public", magnitude=1, duration_turns=4),
    PolicyAction(kind="tax", target="sector:finance", magnitude=0.5),
]


class Quiet:
    """Every outcome is the quiet one."""

    def forecast(self, *args, **kwargs):
        return [QUIET_OUTCOME]


class QuietAfterFirst(CannedScenarios):
    def next_scenario(self, state, history):
        return super().next_scenario(state, history) if not history else QUIET_SCENARIO


class Nothing:
    def interpret(self, text, state, scenario):
        return []


def new_game() -> Game:
    config = GameConfig(seed=0, election_turn=6, k_draws=5)
    return Game(config, toy_world(), QuietAfterFirst(0, calendar=False), Nothing(), Quiet())


def run(game: Game, actions):
    return preview(game.start, game.state, game.scenario, actions, game.config)


def test_the_preview_is_what_playing_the_package_in_a_quiet_world_gives() -> None:
    game = new_game()
    proposal = game.revise(game.propose("plan"), PACKAGE)
    result = run(game, proposal.actions)  # as the limits leave it, as the review screen does
    game.commit(proposal)
    deficits = [game.state.indicators["indicator:deficit"].value]
    while not game.over:
        game.play_turn("nothing")
        deficits.append(game.state.indicators["indicator:deficit"].value)
    assert result.package.vote == pytest.approx(vote_intention(game.state), abs=1e-12)
    assert result.package.deficit == pytest.approx(deficits, abs=1e-12)
    assert result.turns == list(range(1, 7))


def test_doing_nothing_previews_the_same_as_the_baseline() -> None:
    game = new_game()
    result = run(game, [])
    assert result.package == result.nothing
    assert result.measures == []


def test_each_measure_has_a_debt_and_capital_cost() -> None:
    game = new_game()
    result = run(game, PACKAGE)
    assert [m.action for m in result.measures] == PACKAGE
    assert all(m.capital > 0 for m in result.measures)
    lines = preview_lines(result)
    assert any(line.startswith("  Election, this plan:") for line in lines)
    assert any("on the debt by the election" in line for line in lines)


def test_the_review_screen_shows_the_preview() -> None:
    game = new_game()
    proposal = game.revise(game.propose("plan"), PACKAGE)
    text = cli._preview_text(game, proposal)
    assert "Looking ahead to the election" in text and "deficit, doing nothing" in text
