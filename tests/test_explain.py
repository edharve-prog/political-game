"""TT-2: why did that happen, and what else could have?"""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import PolicyAction
from hog_sim.game.explain import alternatives, contributions, why
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
    lines = why(record, before, game.start, game.config)
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
    lines = why(record, before, game.start, game.config)
    deficit = next(line for line in lines if line.strip().startswith("Budget deficit"))
    assert "mostly your spend on Public Sector" in deficit and deficit.endswith("directly")


def test_approval_reasons_name_the_events_in_force() -> None:
    game = new_game()
    game.play_turn("Cap household bills. We will not raise taxes.")
    before = game.state
    record = game.play_turn("Put a windfall tax on energy companies")
    reasons = [line for line in why(record, before, game.start, game.config) if "pts:" in line]
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


def test_a_paid_for_tax_cut_is_not_blamed_for_a_market_penalty_that_never_came() -> None:
    """Review of 361a0ff, finding 8: with a tax rise paying for it, a national tax cut set off
    no market penalty and the pound rose, yet the explanation blamed the cut for a fall of
    about 8 points, re-running it on its own as if unpaid for."""
    game = new_game()
    before = game.state
    package = [
        PolicyAction(kind="tax", target="group:business", magnitude=-1),
        PolicyAction(kind="tax", target="sector:public", magnitude=1),
    ]
    record = game.commit(game.revise(game.propose("Do nothing"), package))
    pound = "indicator:exchange_rate"
    # No market penalty: the pound barely moves, where an unpaid cut knocks about 8 points off.
    assert record.state_after.indicators[pound].value > before.indicators[pound].value - 1
    line = next(x for x in why(record, before, game.start, game.config) if "Sterling" in x)
    assert "mostly your tax on Public Sector" in line
    assert "paying for tax cuts the markets would otherwise have punished" in line

    shares = contributions(record, before, game.start, game.config, pound)
    change = record.state_after.indicators[pound].value - before.indicators[pound].value
    assert sum(shares.values()) == pytest.approx(change, abs=1e-9)
    assert shares["your tax on Public Sector (size +0.75)"] > 0


def test_an_unfunded_tax_cut_is_named_for_the_market_penalty() -> None:
    game = new_game()
    before = game.state
    cut = PolicyAction(kind="tax", target="group:business", magnitude=-1)
    record = game.commit(game.revise(game.propose("Do nothing"), [cut]))
    line = next(x for x in why(record, before, game.start, game.config) if "Sterling" in x)
    assert "mostly your tax on Business owners" in line
    assert "markets' verdict on unfunded tax cuts" in line


def test_approval_reasons_are_the_terms_the_approval_rules_add_up() -> None:
    """Review of 13 Oct 2026, E20: after a public-services spend "why" said public workers rose
    because they care about the deficit, when the cause was employment in their sector, and
    it never named the debt penalty."""
    game = new_game()
    before = game.state
    record = game.play_turn("Fund a pay rise for nurses and teachers")
    lines = why(record, before, game.start, game.config)
    workers = next(line for line in lines if line.strip().startswith("Public sector workers"))
    reasons = workers.split("; ")
    assert "heading for" in reasons[0]
    assert reasons[1].startswith("works in Public Sector +")
    assert "cares about Budget deficit" not in workers
    assert any("deficit above 5% of GDP -" in line for line in lines if "pts:" in line)


def test_the_explained_target_reproduces_the_approval_move() -> None:
    from hog_sim.game.explain import _judged
    from hog_sim.population.popularity import ADJUST, target_approval

    game = new_game()
    for text in ("Fund a pay rise for nurses and teachers", "Put a windfall tax on energy"):
        before = game.state
        record = game.play_turn(text)
        seen = _judged(before, record.state_after)
        targets = target_approval(seen, game.start, seen.baselines)
        for gid, group in before.groups.items():
            moved = group.approval + ADJUST * (targets[gid] - group.approval)
            assert moved == pytest.approx(record.state_after.groups[gid].approval, abs=1e-12)
