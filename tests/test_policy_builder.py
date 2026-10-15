"""O7: the offline policy builder, and editing a keyword reading's kind and target."""

import builtins

import pytest

from hog_sim.content.library import ScenarioLibrary
from hog_sim.core.config import GameConfig
from hog_sim.core.models import PolicyAction
from hog_sim.game.loop import Game
from hog_sim.game.stubs import EngineForecaster, KeywordInterpreter
from hog_sim.ui import cli
from hog_sim.ui.builder import KINDS, build_actions, edit_actions, find_node, targets_for
from hog_sim.world.seed.toy import toy_world


def scripted(*answers: str):
    queue = iter(answers)
    said: list[str] = []
    return (lambda prompt="": next(queue)), said.append, said


def test_build_two_measures_by_number_and_by_name() -> None:
    state = toy_world()
    spend = KINDS.index("spend") + 1
    housing = targets_for("spend", state).index("sector:housing") + 1
    ask, say, said = scripted(
        str(spend), str(housing), "0.3", "6",  # spend 0.3 on housing for 6 turns
        "tax", "business", "-0.5", "",  # cut tax on business for 1 turn
        "",
    )  # fmt: skip
    actions = build_actions(state, ask, say)
    assert [(a.kind, a.target, a.magnitude, a.duration_turns) for a in actions] == [
        ("spend", "sector:housing", 0.3, 6),
        ("tax", "group:business", -0.5, 1),
    ]
    assert any("Added: tax group:business size -0.50" in line for line in said)


def test_the_menu_offers_only_targets_the_kind_fits() -> None:
    state = toy_world()
    assert set(targets_for("diplomatic", state)) == {"country:eu", "country:china"}
    assert "indicator:energy_prices" in targets_for("regulate", state)
    assert "indicator:inflation" not in targets_for("spend", state)
    ask, say, said = scripted("diplomatic", "housing", "china", "-1", "", "")
    [action] = build_actions(state, ask, say)
    assert (action.target, action.magnitude) == ("country:china", -1)
    assert any("no target matches 'housing'" in line for line in said)


def test_bad_numbers_ask_again_and_enter_abandons_a_measure() -> None:
    ask, say, said = scripted("spend", "", "spend", "public", "3", "x", "0.2", "1.5", "2", "")
    [action] = build_actions(toy_world(), ask, say)
    assert (action.target, action.magnitude, action.duration_turns) == ("sector:public", 0.2, 2)
    assert "  enter a number from -1 to 1" in said
    assert "  'x' is not a number" in said
    assert "  enter a whole number from 1 to 120" in said


def test_edit_kind_and_target() -> None:
    nodes = list(toy_world().node_ids())
    a = PolicyAction(kind="communicate", target="sector:housing", magnitude=0.5)
    assert edit_actions([a], "1 kind spend", nodes)[0].kind == "spend"
    assert edit_actions([a], "1 target public", nodes)[0].target == "sector:public"
    assert edit_actions([a], "1 target young renters", nodes)[0].target == "group:young_renters"
    assert edit_actions([a], "1 target country:eu", nodes)[0].target == "country:eu"
    for bad in ("1 kind frobnicate", "1 target nowhere", "2 kind tax"):
        with pytest.raises(ValueError):
            edit_actions([a], bad, nodes)
    with pytest.raises(ValueError, match="could be"):
        find_node("e", nodes)


def test_cli_build_then_add_then_confirm(monkeypatch, capsys) -> None:
    game = Game(
        GameConfig(election_turn=6, k_draws=10),
        toy_world(),
        ScenarioLibrary.load(),
        KeywordInterpreter(),
        EngineForecaster(),
    )
    answers = iter(
        [
            "build",
            "spend",
            "housing",
            "0.4",
            "3",
            "",  # build one measure
            "add",
            "regulate",
            "energy prices",
            "-0.5",
            "",
            "",  # add a second at review
            "2 size -0.3",
            "",
        ]  # fmt: skip
    )
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    proposal = cli._ask_for_turn(game)
    assert proposal.response.startswith("(built) spend sector:housing")
    assert [(a.kind, a.target) for a in proposal.actions] == [
        ("spend", "sector:housing"),
        ("regulate", "indicator:energy_prices"),
    ]
    assert proposal.actions[1].magnitude == pytest.approx(-0.3, abs=0.05)
    record = game.commit(proposal)
    assert record.actions[0].duration_turns == 3
