"""A proposal belongs to the turn and the world it was checked against (review of 361a0ff,
finding 7): an old proposal used to be committed again, after the Commons had turned against
the government, with its full spending intact."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import PolicyAction
from hog_sim.game.loop import Game, StaleProposal
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.world.seed.toy import toy_world

SPEND = PolicyAction(kind="spend", target="sector:public", magnitude=1)


def new_game() -> Game:
    config = GameConfig(seed=0, election_turn=6, k_draws=5)
    plugins = (CannedScenarios(0, calendar=False), KeywordInterpreter(), EngineForecaster())
    return Game(config, toy_world(), *plugins)


def test_a_proposal_cannot_be_committed_twice() -> None:
    game = new_game()
    proposal = game.revise(game.propose("Do nothing"), [SPEND])
    game.commit(proposal)
    with pytest.raises(StaleProposal, match="made on turn 0"):
        game.commit(proposal)
    assert len(game.history) == 1


def test_a_proposal_is_refused_once_the_world_has_changed() -> None:
    game = new_game()
    proposal = game.revise(game.propose("Do nothing"), [SPEND])
    game.state.institutions["institution:legislature"].support = 0.2
    with pytest.raises(StaleProposal, match="world has changed"):
        game.commit(proposal)
    # Checked afresh, the spending is blocked.
    fresh = game.revise(game.propose("Do nothing"), [SPEND])
    assert fresh.actions == []
    assert any(note.startswith("Blocked: spend") for note in fresh.notes)


def test_actions_edited_after_the_check_are_refused() -> None:
    game = new_game()
    proposal = game.revise(game.propose("Do nothing"), [SPEND.model_copy(update={"magnitude": 2})])
    assert proposal.actions[0].magnitude < 2  # the capital budget scaled it
    proposal.actions = [SPEND.model_copy(update={"magnitude": 2})]
    with pytest.raises(StaleProposal, match="limits allow"):
        game.commit(proposal)


def test_a_fresh_proposal_commits_with_its_limits_applied() -> None:
    game = new_game()
    game.play_turn("Do nothing")
    proposal = game.revise(game.propose("Do nothing"), [SPEND])
    record = game.commit(proposal)
    assert record.actions == proposal.actions
