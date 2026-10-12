"""Stored outcomes reused offline are re-scored against today's engine, keep the numbers
their narratives describe, and are not reused when today's engine contradicts them."""

import pytest

from hog_sim.core.models import Outcome, PolicyAction, Scenario
from hog_sim.game.stubs import EngineForecaster
from hog_sim.knowledge.entries import OutcomeEntry, Provenance, scenario_id
from hog_sim.knowledge.offline import StoredForecaster
from hog_sim.knowledge.store import KnowledgeStore
from hog_sim.world.propagation import actions_to_shocks, propagate
from hog_sim.world.seed.toy import toy_world

TAX = PolicyAction(kind="tax", target="sector:energy", magnitude=0.5, rationale="windfall tax")
SCENARIO = Scenario(
    title="Energy bills soar", briefing="Bills are up.", affected_nodes=[], urgency=0.5
)


def engine_forecast(world):
    return propagate(world, actions_to_shocks([TAX], world), 6, 20, 0)


def store_with(tmp_path, candidates) -> KnowledgeStore:
    store = KnowledgeStore(tmp_path / "k.db")
    entry = OutcomeEntry(
        scenario_id=scenario_id(SCENARIO),
        scenario_title=SCENARIO.title,
        affected_nodes=[],
        response="windfall tax",
        actions=[TAX],
        candidates=candidates,
        chosen=0,
        provenance=Provenance(model="fake"),
    )
    store.put("outcome", "k1", entry, scenario=entry.scenario_id)
    return store


def claims_matching(engine, scale=1.0) -> dict[str, float]:
    return {
        n: round(f.mean[2] * scale, 3)
        for n, f in engine.nodes.items()
        if n.startswith("indicator:") and abs(f.mean[2]) > 0.05
    }


def outcome(narrative, claims, judge) -> Outcome:
    return Outcome(
        narrative=narrative,
        indicator_deltas=claims,
        probability=judge,
        scores={"consistency": 1.0, "judge": judge, "base_rate": 0.5, "self_reported": judge},
    )


def test_reused_outcomes_keep_their_claims_and_are_rescored(tmp_path) -> None:
    world = toy_world()
    engine = engine_forecast(world)
    fits = claims_matching(engine)
    assert fits, "the tax should move some indicator"
    wrong = {n: -10 * v for n, v in fits.items()}
    # The stored judge favoured the candidate today's engine contradicts.
    store = store_with(
        tmp_path, [outcome("Wrong way", wrong, 0.9), outcome("As expected", fits, 0.1)]
    )
    forecaster = StoredForecaster(store, EngineForecaster())
    out = forecaster.forecast(world, SCENARIO, [TAX], engine)
    assert forecaster.last_source == "stored"
    assert [c.indicator_deltas for c in out] == [wrong, fits]  # narratives still match
    assert out[1].probability > out[0].probability
    assert sum(c.probability for c in out) == pytest.approx(1.0)
    assert all(c.scores["reused"] == 1.0 for c in out)


def test_outcomes_today_s_engine_contradicts_are_not_reused(tmp_path) -> None:
    world = toy_world()
    engine = engine_forecast(world)
    wrong = {n: -10 * v for n, v in claims_matching(engine).items()}
    store = store_with(tmp_path, [outcome("Wrong way", wrong, 0.5), outcome("Also", wrong, 0.5)])
    forecaster = StoredForecaster(store, EngineForecaster())
    out = forecaster.forecast(world, SCENARIO, [TAX], engine)
    assert forecaster.last_source == "fallback"
    assert all("Wrong way" not in c.narrative for c in out)
