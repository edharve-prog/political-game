"""SD-5: pledges are remembered, and breaking one costs approval."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Pledge, PolicyAction
from hog_sim.forecasting.candidates import LLMForecaster
from hog_sim.game.loop import Game, replay
from hog_sim.game.persistence import SaveStore
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.llm.adapters import LLMInterpreter
from hog_sim.llm.client import FakeClient
from hog_sim.llm.summary import summarise_state
from hog_sim.policy.pledges import (
    BROKEN_HIT,
    BROKEN_HIT_ALL,
    apply_pledges,
    breaks,
    broken_by,
)
from hog_sim.ui import cli
from hog_sim.world.propagation import actions_to_shocks, propagate
from hog_sim.world.seed.toy import toy_world

NO_NEW_TAXES = Pledge(text="No new taxes", kind="tax", direction="up")
HANDS_OFF_PUBLIC = Pledge(
    text="No cuts to schools and hospitals",
    kind="spend",
    direction="down",
    target="sector:public",
    groups=["group:public_workers"],
)


def act(kind, target, magnitude):
    return PolicyAction(kind=kind, target=target, magnitude=magnitude)


@pytest.mark.parametrize(
    ("pledge", "action", "broken"),
    [
        (NO_NEW_TAXES, act("tax", "sector:energy", 0.3), True),
        (NO_NEW_TAXES, act("tax", "sector:energy", -0.3), False),
        (NO_NEW_TAXES, act("spend", "sector:energy", 0.3), False),
        (HANDS_OFF_PUBLIC, act("spend", "sector:public", -0.2), True),
        (HANDS_OFF_PUBLIC, act("spend", "sector:housing", -0.2), False),
        (HANDS_OFF_PUBLIC, act("spend", "sector:public", 0.2), False),
    ],
)
def test_a_pledge_rules_out_one_kind_direction_and_target(pledge, action, broken) -> None:
    assert breaks(pledge, action) is broken


def test_breaking_a_pledge_marks_it_and_hits_the_groups_who_cared() -> None:
    state = toy_world()
    state.pledges = [NO_NEW_TAXES, HANDS_OFF_PUBLIC]
    new = apply_pledges(state, 4, [act("spend", "sector:public", -0.3)], [])
    assert [p.broken_turn for p in new.pledges] == [None, 4]
    event = new.events[-1]
    assert event.name == "Broke pledge: No cuts to schools and hospitals"
    assert event.group_effects == {"group:public_workers": -BROKEN_HIT}
    assert state.pledges[1].broken_turn is None  # the input is not changed

    again = apply_pledges(new, 5, [act("spend", "sector:public", -0.3)], [])
    assert len(again.events) == len(new.events)  # a pledge breaks once


def test_a_pledge_naming_no_groups_costs_everyone_a_little() -> None:
    state = toy_world()
    state.pledges = [NO_NEW_TAXES]
    new = apply_pledges(state, 2, [act("tax", "sector:finance", 0.4)], [])
    assert new.events[-1].group_effects == {g: -BROKEN_HIT_ALL for g in state.groups}


def test_new_pledges_get_the_turn_and_are_not_repeated() -> None:
    state = toy_world()
    new = apply_pledges(state, 3, [], [NO_NEW_TAXES, NO_NEW_TAXES.model_copy()])
    assert len(new.pledges) == 1 and new.pledges[0].made_turn == 3
    # Made this turn, so this turn's own actions don't break it.
    raised = apply_pledges(state, 3, [act("tax", "sector:energy", 0.3)], [NO_NEW_TAXES])
    assert raised.pledges[0].broken_turn is None


def new_game():
    config = GameConfig(seed=0, election_turn=6, k_draws=20)
    plugins = (CannedScenarios(0), KeywordInterpreter(), EngineForecaster())
    return Game(config, toy_world(), *plugins)


def test_a_pledge_made_in_one_turn_is_broken_in_a_later_one_and_replays(tmp_path) -> None:
    game = new_game()
    first = game.play_turn("Cap household bills. We will not raise taxes.")
    assert [p.text for p in first.pledges] == ["We will not raise taxes"]
    assert all(a.kind != "tax" for a in first.actions)
    assert game.state.pledges[0].made_turn == 0

    proposal = game.propose("Put a windfall tax on energy companies")
    assert [p.text for p in broken_by(game.state, proposal.actions)] == ["We will not raise taxes"]
    assert "Warning: this breaks your pledge" in cli._proposal_text(proposal, [], game.state)
    second = game.commit(proposal)
    assert "breaks the pledge" in second.outcome.narrative
    assert game.state.pledges[0].broken_turn == 1
    assert any(e.name.startswith("Broke pledge") for e in game.state.events)
    assert "BROKEN on turn 1" in cli._dashboard(game.start, game.state)

    assert replay(game.start, game.config, game.history) == game.state
    store = SaveStore(tmp_path / "saves.db")
    game_id = store.new_game(game.config, game.start)
    for record in game.history:
        store.save_turn(game_id, record)
    _, start, records = store.load(game_id)
    assert replay(start, game.config, records) == game.state


def test_the_llm_interpreter_extracts_pledges() -> None:
    state = toy_world()
    reply = {
        "actions": [{"kind": "spend", "target": "sector:housing", "magnitude": 0.4}],
        "pledges": [
            {
                "text": "No cuts to the NHS",
                "kind": "spend",
                "direction": "down",
                "target": "public",
                "groups": ["public_workers"],
            }
        ],
    }
    interpreter = LLMInterpreter(FakeClient([reply]))
    interpreter.interpret(
        "Build homes, and no cuts to the NHS", state, CannedScenarios(0).next_scenario(state, [])
    )
    assert interpreter.last_pledges() == [
        Pledge(
            text="No cuts to the NHS",
            kind="spend",
            direction="down",
            target="sector:public",
            groups=["group:public_workers"],
        )
    ]


def test_a_pledge_naming_a_non_group_is_sent_back() -> None:
    state = toy_world()
    bad = {
        "actions": [],
        "clarifying_question": "What?",
        "pledges": [{"text": "x", "kind": "tax", "direction": "up", "groups": ["sector:energy"]}],
    }
    client = FakeClient([bad, {"actions": [], "clarifying_question": "What?"}])
    LLMInterpreter(client).interpret("Hmm", state, CannedScenarios(0).next_scenario(state, []))
    assert len(client.requests) == 2
    assert "pledges[0]" in client.requests[1].messages[-1].content


def test_prompts_show_pledges_and_a_broken_one_reaches_the_outcome_writer() -> None:
    state = toy_world()
    state.pledges = [NO_NEW_TAXES.model_copy(update={"made_turn": 0})]
    assert '"No new taxes" (made turn 0, kept so far)' in summarise_state(state).to_prompt()

    tax = act("tax", "sector:energy", 0.5)
    engine = propagate(state, actions_to_shocks([tax]), horizon=6, k_draws=20)

    def responder(request):
        raise AssertionError("stop after the first request")

    client = FakeClient(responder=responder)
    scenario = CannedScenarios(0).next_scenario(state, [])
    with pytest.raises(AssertionError):
        LLMForecaster(client).forecast(state, scenario, [tax], engine)
    prompt = client.requests[0].messages[0].content
    assert "This response breaks the leader's pledges:\n- No new taxes" in prompt
