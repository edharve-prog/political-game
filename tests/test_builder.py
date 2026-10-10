import builtins

import pytest

from hog_sim.content.library import ScenarioLibrary
from hog_sim.core.config import GameConfig
from hog_sim.core.models import PolicyAction
from hog_sim.game.loop import Game, replay
from hog_sim.game.stubs import EngineForecaster, KeywordInterpreter
from hog_sim.ui import cli
from hog_sim.ui.builder import compose_response, edit_actions
from hog_sim.world.seed.toy import toy_world

OPTIONS = ["Tax energy profits.", "Spend on home insulation.", "Do nothing."]


def test_compose_picks_and_extra_words() -> None:
    assert compose_response("1 2", OPTIONS) == (
        "Option: Tax energy profits.\nOption: Spend on home insulation."
    )
    assert compose_response("2,2 + also freeze rail fares", OPTIONS) == (
        "Option: Spend on home insulation.\nAlso: also freeze rail fares"
    )
    assert compose_response("cap bills now", OPTIONS) == "cap bills now"
    with pytest.raises(ValueError, match="no option 4"):
        compose_response("4", OPTIONS)


def test_edit_actions() -> None:
    a = PolicyAction(kind="tax", target="sector:energy", magnitude=0.5)
    b = PolicyAction(kind="spend", target="sector:housing", magnitude=0.3)
    assert edit_actions([a, b], "drop 1") == [b]
    assert edit_actions([a, b], "2 size -0.2")[1].magnitude == -0.2
    assert edit_actions([a, b], "1 turns 6")[0].duration_turns == 6
    for bad in ("drop 3", "1 size 2", "1 turns 0.5", "frobnicate"):
        with pytest.raises(ValueError):
            edit_actions([a, b], bad)


def game():
    config = GameConfig(election_turn=6, k_draws=10)
    world = toy_world()
    return Game(config, world, ScenarioLibrary.load(), KeywordInterpreter(), EngineForecaster())


def test_propose_changes_nothing_and_commit_uses_edits() -> None:
    g = game()
    before = g.state
    proposal = g.propose(compose_response("1 2", ["Tax energy profits.", "Spend on housing."]))
    assert g.state == before and g.history == []
    assert {a.kind for a in proposal.actions} == {"tax", "spend"}
    smaller = edit_actions(proposal.actions, "1 size 0.1")
    revised = g.revise(proposal, smaller)
    record = g.commit(revised)
    assert record.actions[0].magnitude == pytest.approx(0.1, abs=0.05)
    assert record.requested_actions == proposal.requested
    assert replay(g.start, g.config, g.history) == g.state


def test_cli_review_flow(tmp_path, monkeypatch, capsys) -> None:
    answers = iter(["9", "1 + spend on housing", "drop 1", "", "redo"])

    def fake_input(prompt=""):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", fake_input)
    cli.main(["--offline", "--save", str(tmp_path / "g.db")])
    out = capsys.readouterr().out
    assert "there is no option 9" in out
    assert "Your advisers read that as" in out and "  1. " in out
