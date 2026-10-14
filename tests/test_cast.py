"""SD-4: a recurring cast whose loyalty follows how the leader treats them."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Delivery, PolicyAction
from hog_sim.game.loop import Game, replay
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.llm.adapters import LLMInterpreter
from hog_sim.llm.client import FakeClient
from hog_sim.llm.scenario_gen import generate_scenario
from hog_sim.llm.summary import summarise_state
from hog_sim.ui import cli
from hog_sim.world.cast import (
    BACKED,
    CONSULTED,
    CROSSED,
    DEPARTURE_COST,
    FULL_EFFECT,
    IGNORED,
    RESIGN_BELOW,
    apply_cast,
    loyalty_change,
)
from hog_sim.world.seed.toy import toy_world


def scenario_draft(**overrides) -> dict:
    data = {
        "title": "Union ballots for strikes",
        "category": "economy",
        "storyline": "new",
        "secondary": [
            {
                "title": "Rents climb again",
                "category": "housing",
                "briefing": "Young renters face another rise.",
                "affected_nodes": ["sector:housing"],
                "urgency": 0.4,
                "storyline": "new",
            }
        ],
        "briefing": "The union opens a strike ballot over pay.",
        "affected_nodes": ["sector:public", "group:public_workers"],
        "urgency": 0.7,
        "suggested_options": ["Offer a pay rise", "Hold firm"],
        "stakeholder_positions": [
            {"node": "group:public_workers", "stance": 0.9, "statement": "Pay us."},
            {"node": "group:pensioners", "stance": -0.2, "statement": "Keep strikes short."},
        ],
    }
    return data | overrides


def act(kind, target, magnitude):
    return PolicyAction(kind=kind, target=target, magnitude=magnitude)


def loyalty_moves(actions: list[PolicyAction]) -> dict[str, float]:
    state = toy_world()
    new = apply_cast(state, 0, actions)
    return {cid: new.characters[cid].loyalty - c.loyalty for cid, c in state.characters.items()}


def test_the_chancellor_rewards_a_fiscal_consolidation() -> None:
    """E14: a tax rise plus a pension cut left the Chancellor, who cares about the deficit,
    where she was, since no action may aim at the deficit. Loyalty now follows effects (D5)."""
    moves = loyalty_moves(
        [act("tax", "group:business", 1.0), act("spend", "group:pensioners", -1.0)]
    )
    assert moves["person:chancellor"] == pytest.approx(BACKED)
    assert moves["person:business_lobby"] == pytest.approx(CROSSED)
    assert moves["person:editor"] == pytest.approx(CROSSED)


def test_the_governor_rewards_lower_inflation() -> None:
    moves = loyalty_moves([act("regulate", "indicator:energy_prices", -0.5)])
    assert moves["person:governor"] > 0
    assert moves["person:editor"] > 0


def test_loyalty_moves_with_the_size_of_the_effect() -> None:
    small = loyalty_moves([act("spend", "sector:housing", 0.1)])["person:housing_secretary"]
    large = loyalty_moves([act("spend", "sector:housing", 0.3)])["person:housing_secretary"]
    assert 0 < small < large == pytest.approx(BACKED)
    assert loyalty_change(FULL_EFFECT / 2) == pytest.approx(BACKED / 2)
    assert loyalty_change(-FULL_EFFECT * 3) == pytest.approx(CROSSED)
    assert loyalty_change(0.01) == 0.0


def test_an_action_that_names_a_node_but_does_not_move_it_backs_no_one() -> None:
    """A spend on unemployment is a wage subsidy: in good times it moves nothing the union
    or the Housing Secretary cares about, though it costs the Chancellor money."""
    moves = loyalty_moves([act("spend", "indicator:unemployment", 0.5)])
    assert moves["person:housing_secretary"] == 0.0
    assert moves["person:chancellor"] < 0


def test_spending_backs_the_union_and_crosses_the_chancellor() -> None:
    state = toy_world()
    new = apply_cast(state, 2, [act("spend", "sector:public", 0.4)])
    union, chancellor = new.characters["person:union_leader"], new.characters["person:chancellor"]
    assert union.loyalty == pytest.approx(state.characters[union.id].loyalty + BACKED)
    assert chancellor.loyalty == pytest.approx(state.characters[chancellor.id].loyalty + CROSSED)
    assert union.memory == ["backed on turn 2"]
    assert new.characters["person:editor"].loyalty == state.characters["person:editor"].loyalty


def test_consulting_helps_and_ignoring_an_involved_person_hurts() -> None:
    state = toy_world()
    talks = Delivery(consulted=["group:public_workers"])
    new = apply_cast(state, 1, [], talks, involved=["person:union_leader", "person:editor"])
    union, editor = new.characters["person:union_leader"], new.characters["person:editor"]
    assert union.loyalty == pytest.approx(state.characters[union.id].loyalty + CONSULTED)
    assert editor.loyalty == pytest.approx(state.characters[editor.id].loyalty + IGNORED)
    assert editor.memory == ["ignored on turn 1"]


def test_a_sacked_minister_leaves_costs_commons_support_and_is_replaced() -> None:
    state = toy_world()
    new = apply_cast(state, 3, [], sacked=["person:chancellor", "person:union_leader"])
    gone = new.characters["person:chancellor"]
    assert not gone.active and gone.memory[-1] == "sacked on turn 3"
    successor = new.characters["person:chancellor-3"]
    assert successor.active and successor.role == gone.role and successor.name != gone.name
    assert new.characters["person:union_leader"].active  # only ministers can be sacked
    support = new.institutions["institution:legislature"].support
    assert support == pytest.approx(
        state.institutions["institution:legislature"].support - DEPARTURE_COST
    )


def test_a_minister_crossed_too_often_resigns() -> None:
    state = toy_world()
    state.characters["person:chancellor"].loyalty = RESIGN_BELOW + 0.05
    new = apply_cast(state, 6, [act("spend", "sector:public", 0.5)])
    assert new.characters["person:chancellor"].memory[-1] == "resigned on turn 6"
    assert "person:chancellor-6" in new.characters


def test_prompts_list_the_cast_and_scenarios_may_involve_them() -> None:
    state = toy_world()
    state.characters["person:union_leader"].loyalty = 0.2
    summary = summarise_state(state)
    text = summary.to_prompt()
    assert "person:chancellor Ruth Calder, Chancellor of the Exchequer (minister)" in text
    assert "[LOW LOYALTY" in summary.cast["person:union_leader"]

    good = scenario_draft(characters=["union_leader"])  # bare id is repaired
    client = FakeClient([scenario_draft(characters=["person:nobody"]), good])
    scenario = generate_scenario(summary, client)
    assert scenario.characters == ["person:union_leader"]
    assert "unknown people" in client.requests[1].messages[-1].content


def test_the_llm_interpreter_can_sack_a_minister_but_no_one_else() -> None:
    state = toy_world()
    scenario = CannedScenarios(0).next_scenario(state, [])
    client = FakeClient([{"sacked": ["person:editor"]}, {"sacked": ["chancellor"]}])
    interpreter = LLMInterpreter(client)
    assert interpreter.interpret("Sack the Chancellor", state, scenario) == []
    assert interpreter.last_sacked() == ["person:chancellor"]
    assert "sacked must be minister ids" in client.requests[1].messages[-1].content


def test_an_offline_game_sacks_a_minister_and_replays() -> None:
    config = GameConfig(seed=0, election_turn=6, k_draws=20)
    game = Game(config, toy_world(), CannedScenarios(0), KeywordInterpreter(), EngineForecaster())
    proposal = game.propose("Sack the Chancellor.")
    assert proposal.sacked == ["person:chancellor"]
    assert "Sacking: Ruth Calder" in cli._proposal_text(proposal, [], game.state)
    record = game.commit(proposal)
    assert "Ruth Calder is sacked" in record.outcome.narrative
    assert not game.state.characters["person:chancellor"].active
    board = cli._dashboard(game.start, game.state)
    assert "PEOPLE" in board and "Ruth Calder" not in board
    game.play_turn("Fund a pay rise for nurses and teachers")
    assert replay(game.start, game.config, game.history) == game.state
