"""Several issues per turn: the in-tray (backlog story SD-2)."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction, Scenario, SideIssue
from hog_sim.game.loop import Game, replay
from hog_sim.game.stubs import EngineForecaster
from hog_sim.llm.adapters import LLMInterpreter, LLMScenarioSource
from hog_sim.llm.client import FakeClient, LLMRequest
from hog_sim.ui.cli import _briefing, _left_alone
from hog_sim.world.seed.toy import toy_world
from hog_sim.world.storylines import ESCALATE_AFTER, RELIEF, advance_storylines

ENERGY_TAX = PolicyAction(kind="tax", target="sector:energy", magnitude=0.3)
HOUSING_SPEND = PolicyAction(kind="spend", target="sector:housing", magnitude=0.3)


def tray(*items: SideIssue) -> Scenario:
    return Scenario(
        title="Energy bills soar",
        briefing="...",
        affected_nodes=["sector:energy"],
        urgency=0.7,
        storyline="energy",
        secondary=list(items),
    )


def item(title: str, storyline: str, urgency: float = 0.5, node: str = "sector:housing"):
    return SideIssue(
        title=title, briefing="...", affected_nodes=[node], urgency=urgency, storyline=storyline
    )


OUTCOME = Outcome(narrative="Bills eased.", probability=1.0)


def test_items_acted_on_advance_and_items_left_alone_carry_over() -> None:
    state = toy_world()
    scenario = tray(item("Rents climb", "rents"), item("Pothole row", "potholes", urgency=0.2))
    state = advance_storylines(state, 0, scenario, OUTCOME, [ENERGY_TAX])
    rents, potholes = state.storylines["rents"], state.storylines["potholes"]
    assert rents.open and "left in the in-tray" in rents.history[-1]
    assert not potholes.open and "faded away" in potholes.history[-1]

    # Seen every turn but never acted on: it still escalates on the idle clock.
    for turn in range(1, 1 + ESCALATE_AFTER):
        state = advance_storylines(state, turn, tray(item("Rents climb", "rents")), OUTCOME)
    assert state.storylines["rents"].stage == 2
    assert "escalated while left alone" in state.storylines["rents"].history[-1]

    turn = 1 + ESCALATE_AFTER
    acted = [ENERGY_TAX, HOUSING_SPEND]
    state = advance_storylines(state, turn, tray(item("Rents climb", "rents")), OUTCOME, acted)
    rents = state.storylines["rents"]
    # Acting eases it rather than escalating it (E11): same stage, less pressure, clock reset.
    assert rents.stage == 2 and rents.last_turn == turn and rents.open
    assert "(acted on, eased)" in rents.history[-1]
    assert rents.pressure == pytest.approx(0.5 + 0.15 - RELIEF)
    assert state.storylines["energy"].stage == 1 + 1 + ESCALATE_AFTER


def side(title: str, storyline: str) -> dict:
    return {
        "title": title,
        "category": "housing",
        "briefing": "Rents are up again.",
        "affected_nodes": ["sector:housing", "group:young_renters"],
        "urgency": 0.5,
        "storyline": storyline,
    }


def test_claude_game_in_tray_combines_actions_and_replays() -> None:
    def responder(request: LLMRequest) -> dict:
        prompt = request.messages[-1].content
        if request.schema_name == "Interpretation":
            both = "housing" in prompt.lower().split("respon")[-1]
            actions = [ENERGY_TAX, HOUSING_SPEND] if both else [ENERGY_TAX]
            return {"actions": [a.model_dump() for a in actions]}
        rents = next((w for w in prompt.split() if w.startswith("rents-climb-t")), "new")
        return {
            "title": "Energy bills soar",
            "category": "energy",
            "storyline": "new",
            "briefing": "Bills are up.",
            "affected_nodes": ["sector:energy", "group:pensioners"],
            "urgency": 0.7,
            "suggested_options": ["Tax windfall profits", "Do nothing"],
            "stakeholder_positions": [
                {"node": "group:pensioners", "stance": 0.8, "statement": "Help us."},
                {"node": "sector:energy", "stance": -0.5, "statement": "Hands off."},
            ],
            "secondary": [side("Rents climb", rents)],
        }

    client = FakeClient(responder=responder)
    config = GameConfig(election_turn=10, k_draws=10)
    world = toy_world()
    game = Game(
        config, world, LLMScenarioSource(client), LLMInterpreter(client), EngineForecaster()
    )
    assert "Also in your in-tray" in _briefing(game) and "Rents climb" in _briefing(game)

    for _ in range(4):
        played = game.state
        record = game.play_turn("Windfall tax on energy firms")
        assert "Rents climb (carries over)" in _left_alone(record, played)
    rents_id = record.scenario.secondary[0].storyline
    assert game.state.storylines[rents_id].stage == 2  # escalated while left alone

    played = game.state
    record = game.play_turn("Windfall tax on energy firms, and fund housing")
    assert {a.target for a in record.actions} == {"sector:energy", "sector:housing"}
    assert _left_alone(record, played) == ""
    assert game.state.storylines[rents_id].stage == 2  # eased, not escalated (E11)

    interpreter_prompts = [
        r.messages[-1].content for r in client.requests if r.schema_name == "Interpretation"
    ]
    assert "Also in the in-tray" in interpreter_prompts[0]
    assert replay(world, config, game.history) == game.state


# E11: acting on an in-tray item used to raise its stage and keep it open, while leaving a
# minor one alone made it vanish. Now acting judges the decisions' effect on the item.
HOUSING_CUT = PolicyAction(kind="spend", target="sector:housing", magnitude=-0.3)


def turns_open(urgency: float, actions: list[PolicyAction], every_turn: bool) -> int:
    """The turn the item closes (14 if it never does), shown in the in-tray each turn while
    open, with ``actions`` taken on the first turn (or every turn)."""
    state = toy_world()
    issue = item("Pothole row", "potholes", urgency=urgency)
    for turn in range(14):
        acts = actions if every_turn or turn == 0 else []
        state = advance_storylines(state, turn, tray(issue), OUTCOME, [ENERGY_TAX, *acts])
        if not state.storylines["potholes"].open:
            return turn
    return 14


def test_acting_on_a_minor_item_deals_with_it() -> None:
    state = advance_storylines(
        toy_world(), 0, tray(item("Pothole row", "potholes", urgency=0.2)), OUTCOME, [HOUSING_SPEND]
    )
    potholes = state.storylines["potholes"]
    assert not potholes.open and "dealt with" in potholes.history[-1]


def test_a_cut_is_not_help_and_makes_the_item_worse() -> None:
    state = advance_storylines(
        toy_world(), 0, tray(item("Pothole row", "potholes", urgency=0.2)), OUTCOME, [HOUSING_CUT]
    )
    potholes = state.storylines["potholes"]
    assert potholes.open and potholes.stage == 2 and "made worse" in potholes.history[-1]
    assert potholes.pressure == pytest.approx(0.2 + 0.15)


def test_an_action_on_another_node_neither_helps_nor_harms() -> None:
    state = advance_storylines(
        toy_world(), 0, tray(item("Rents climb", "rents")), OUTCOME, [ENERGY_TAX]
    )
    assert "left in the in-tray" in state.storylines["rents"].history[-1]


@pytest.mark.parametrize("urgency", [0.1, 0.25, 0.4, 0.6, 0.8, 1.0])
@pytest.mark.parametrize("every_turn", [False, True])
def test_acting_never_keeps_an_item_open_longer_than_ignoring_it(
    urgency: float, every_turn: bool
) -> None:
    ignored = turns_open(urgency, [], every_turn)
    acted = turns_open(urgency, [HOUSING_SPEND], every_turn)
    assert acted <= ignored
