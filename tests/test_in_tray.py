"""Several issues per turn: the in-tray (backlog story SD-2)."""

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction, Scenario, SideIssue
from hog_sim.game.loop import Game, replay
from hog_sim.game.stubs import EngineForecaster
from hog_sim.llm.adapters import LLMInterpreter, LLMScenarioSource
from hog_sim.llm.client import FakeClient, LLMRequest
from hog_sim.ui.cli import _briefing, _left_alone
from hog_sim.world.seed.toy import toy_world
from hog_sim.world.storylines import ESCALATE_AFTER, advance_storylines

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
    assert rents.stage == 3 and rents.last_turn == turn and "(acted on)" in rents.history[-1]
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
        record = game.play_turn("Windfall tax on energy firms")
        assert "Rents climb (carries over)" in _left_alone(record)
    rents_id = record.scenario.secondary[0].storyline
    assert game.state.storylines[rents_id].stage == 2  # escalated while left alone

    record = game.play_turn("Windfall tax on energy firms, and fund housing")
    assert {a.target for a in record.actions} == {"sector:energy", "sector:housing"}
    assert _left_alone(record) == ""
    assert game.state.storylines[rents_id].stage == 3

    interpreter_prompts = [
        r.messages[-1].content for r in client.requests if r.schema_name == "Interpretation"
    ]
    assert "Also in the in-tray" in interpreter_prompts[0]
    assert replay(world, config, game.history) == game.state
