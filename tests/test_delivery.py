"""How a response is delivered feeds the outcome (backlog story RB-4)."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Delivery, PolicyAction
from hog_sim.forecasting.candidates import (
    CandidateDraft,
    CandidateSet,
    ForecastConfig,
    LLMForecaster,
    Shift,
)
from hog_sim.forecasting.scoring import base_rate
from hog_sim.game.loop import Game, replay
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.llm.adapters import LLMInterpreter
from hog_sim.llm.client import FakeClient, LLMRequest
from hog_sim.llm.interpreter import Interpretation, check_interpretation
from hog_sim.llm.summary import summarise_state
from hog_sim.world.propagation import actions_to_shocks, propagate
from hog_sim.world.seed.toy import toy_world

TAX = PolicyAction(kind="tax", target="sector:energy", magnitude=0.5, rationale="windfall tax")
CONSULTED = Delivery(framing="fairness", consulted=["group:public_workers"], speed="phased")
IMPOSED = Delivery(framing="no time to lose", speed="immediate")


@pytest.fixture
def world():
    return toy_world()


@pytest.fixture
def engine(world):
    return propagate(world, actions_to_shocks([TAX]), horizon=6, k_draws=50)


def candidates(engine) -> CandidateSet:
    expected = engine.nodes["indicator:inflation"].mean[2]

    def draft(title, tags):
        return CandidateDraft(
            title=title,
            narrative=f"{title}.",
            indicator_shifts=[Shift(node="indicator:inflation", change=expected)],
            group_effects=[],
            new_shocks=[],
            graph_changes=[],
            event_tags=tags,
            resolves_storyline=False,
            self_probability=0.5,
        )

    return CandidateSet(candidates=[draft("Quiet passage", ["none"]), draft("Strike", ["strike"])])


def strike_probability(world, engine, delivery: Delivery) -> tuple[float, str]:
    client = FakeClient([candidates(engine)])
    forecaster = LLMForecaster(client, config=ForecastConfig(n_candidates=2, use_judge=False))
    scenario = CannedScenarios().next_scenario(world, [])
    outcomes = forecaster.forecast(world, scenario, [TAX], engine, delivery=delivery)
    return outcomes[1].probability, client.requests[0].messages[0].content


def test_consulting_unions_makes_a_strike_less_likely_than_imposing(world, engine) -> None:
    consulted, consulted_prompt = strike_probability(world, engine, CONSULTED)
    imposed, imposed_prompt = strike_probability(world, engine, IMPOSED)
    assert consulted < imposed
    assert "How it was delivered:" in consulted_prompt
    assert 'framed as "fairness"' in consulted_prompt and "group:public_workers" in consulted_prompt
    assert "imposed immediately" in imposed_prompt


def test_delivery_factors() -> None:
    assert base_rate(["strike"], CONSULTED) < base_rate(["strike"]) < base_rate(["strike"], IMPOSED)
    assert base_rate(["market_selloff"], CONSULTED) < base_rate(["market_selloff"])
    assert base_rate(["none"], IMPOSED) == base_rate(["none"])


def test_interpreter_delivery_reaches_the_record_and_the_forecaster(world) -> None:
    def responder(request: LLMRequest) -> dict:
        return Interpretation(actions=[TAX], delivery=CONSULTED).model_dump()

    seen = []

    class Recording(EngineForecaster):
        def forecast(self, *args, delivery=None, **kwargs):
            seen.append(delivery)
            return super().forecast(*args, delivery=delivery, **kwargs)

    config = GameConfig(k_draws=10)
    game = Game(
        config,
        world,
        CannedScenarios(),
        LLMInterpreter(FakeClient(responder=responder)),
        Recording(),
    )
    record = game.play_turn("Talk to the unions, then phase in a windfall tax for fairness")
    assert record.delivery == CONSULTED and seen == [CONSULTED]
    assert replay(world, config, game.history) == game.state


def test_unknown_consulted_ids_are_sent_back(world) -> None:
    bad = Interpretation(actions=[TAX], delivery=Delivery(consulted=["group:aliens"]))
    assert any("group:aliens" in p for p in check_interpretation(bad, summarise_state(world)))


def test_keyword_delivery_offline(world) -> None:
    interpreter = KeywordInterpreter()
    interpreter.interpret("Consult the unions and phase in a windfall tax", world, None)
    delivery = interpreter.last_delivery()
    assert delivery.consulted == ["group:public_workers"] and delivery.speed == "phased"
    interpreter.interpret("Windfall tax on energy firms, overnight", world, None)
    assert interpreter.last_delivery() == Delivery()


def test_offline_consulting_lowers_the_backlash_chance(world, engine) -> None:
    scenario = CannedScenarios().next_scenario(world, [])
    plain = EngineForecaster().forecast(world, scenario, [TAX], engine)
    consulted = EngineForecaster().forecast(world, scenario, [TAX], engine, delivery=CONSULTED)
    assert consulted[1].probability < plain[1].probability
    assert consulted[2].probability == plain[2].probability
    assert sum(o.probability for o in consulted) == pytest.approx(1.0)
