"""Blocked measures reach the outcome (backlog story RB-8)."""

from hog_sim.core.config import GameConfig
from hog_sim.core.models import PolicyAction
from hog_sim.forecasting.candidates import (
    CandidateDraft,
    CandidateSet,
    ForecastConfig,
    LLMForecaster,
)
from hog_sim.game.loop import Game
from hog_sim.game.stubs import CannedScenarios, EngineForecaster
from hog_sim.llm.adapters import LLMInterpreter
from hog_sim.llm.client import FakeClient, LLMRequest
from hog_sim.world.propagation import actions_to_shocks, propagate
from hog_sim.world.seed.toy import toy_world

SPEND = PolicyAction(kind="spend", target="group:pensioners", magnitude=0.4, rationale="fuel help")


def hung_commons():
    world = toy_world()
    world.institutions["institution:legislature"].support = 0.4
    return world


def test_blocked_notes_reach_the_forecaster() -> None:
    seen = []

    class Recording(EngineForecaster):
        def forecast(self, *args, limits=None, **kwargs):
            seen.append(limits)
            return super().forecast(*args, limits=limits, **kwargs)

    def responder(request: LLMRequest) -> dict:
        return {"actions": [SPEND.model_dump()]}

    game = Game(
        GameConfig(k_draws=10),
        hung_commons(),
        CannedScenarios(),
        LLMInterpreter(FakeClient(responder=responder)),
        Recording(),
    )
    record = game.play_turn("Winter fuel help for pensioners")
    assert record.actions == []
    assert len(seen[0]) == 1 and seen[0][0].startswith("Blocked: spend group:pensioners")
    assert seen[0] == record.notes


def test_outcome_prompt_lists_blocked_measures() -> None:
    world = hung_commons()
    engine = propagate(world, actions_to_shocks([]), horizon=6, k_draws=10)
    quiet = CandidateDraft(
        title="Commons defeat",
        narrative="The Commons voted the plan down.",
        indicator_shifts=[],
        group_effects=[],
        new_shocks=[],
        graph_changes=[],
        event_tags=["backbench_rebellion"],
        resolves_storyline=False,
        self_probability=0.5,
    )
    client = FakeClient([CandidateSet(candidates=[quiet, quiet.model_copy(update={"title": "x"})])])
    forecaster = LLMForecaster(client, config=ForecastConfig(n_candidates=2, use_judge=False))
    note = "Blocked: spend group:pensioners (House of Commons support is 0.40, short of a majority)"
    scenario = CannedScenarios().next_scenario(world, [])
    forecaster.forecast(world, scenario, [], engine, limits=[note])
    prompt = client.requests[0].messages[0].content
    assert "Blocked or weakened (not enacted as asked):" in prompt and f"- {note}" in prompt
    assert "Never describe a blocked measure as enacted" in client.requests[0].system
