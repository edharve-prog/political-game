"""TT-2: why did that happen, and what else could have?"""

from hog_sim.core.config import GameConfig
from hog_sim.core.models import PolicyAction
from hog_sim.game.explain import alternatives, why
from hog_sim.game.loop import Game
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.ui import cli
from hog_sim.world.seed.toy import toy_world


def new_game():
    config = GameConfig(seed=0, election_turn=6, k_draws=20)
    plugins = (CannedScenarios(0, calendar=False), KeywordInterpreter(), EngineForecaster())
    return Game(config, toy_world(), *plugins)


def test_why_names_the_action_behind_the_biggest_move_and_its_path() -> None:
    game = new_game()
    tax = PolicyAction(kind="tax", target="sector:energy", magnitude=0.8)
    before = game.state
    record = game.commit(game.revise(game.propose("Do nothing"), [tax]))
    lines = why(record, before)
    assert lines[0] == "Biggest moves in the indicators this turn:"
    energy = next(line for line in lines if "Household energy prices" in line)
    assert "mostly your tax on Energy (size +0.80)" in energy
    assert "via Energy → Household energy prices" in energy
    assert "Biggest moves in approval:" in lines
    assert len([line for line in lines if "pts:" in line]) == 3


def test_a_spending_rise_moves_the_deficit_directly() -> None:
    game = new_game()
    before = game.state
    record = game.play_turn("Fund a pay rise for nurses and teachers")
    lines = why(record, before)
    deficit = next(line for line in lines if line.strip().startswith("Budget deficit"))
    assert "mostly your spend on Public Sector" in deficit and deficit.endswith("directly")


def test_approval_reasons_name_the_events_in_force() -> None:
    game = new_game()
    game.play_turn("Cap household bills. We will not raise taxes.")
    before = game.state
    record = game.play_turn("Put a windfall tax on energy companies")
    reasons = [line for line in why(record, before) if "pts:" in line]
    assert any('"Broke pledge: We will not raise taxes"' in line for line in reasons)


def test_alternatives_lists_every_candidate_and_marks_the_chosen_one() -> None:
    game = new_game()
    record = game.play_turn("Invest in housebuilding")
    lines = alternatives(record)
    assert len([line for line in lines if "p=" in line]) == len(record.candidates)
    assert sum("[chosen]" in line for line in lines) == 1


def test_the_cli_answers_why_and_alternatives_about_the_last_turn() -> None:
    game = new_game()
    assert "play a turn first" in cli._explain(game, "why")
    game.play_turn("Invest in housebuilding")
    assert cli._explain(game, "why").startswith("Biggest moves in the indicators")
    assert cli._explain(game, "alternatives").startswith("The outcomes that were in play")
