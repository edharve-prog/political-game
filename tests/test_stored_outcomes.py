"""Stored outcomes reused offline are re-scored against today's engine, keep the numbers
their narratives describe, and are not reused when today's engine contradicts them."""

import pytest

from hog_sim.core.models import Delivery, Outcome, Pledge, PolicyAction, Scenario
from hog_sim.game.stubs import EngineForecaster
from hog_sim.knowledge.entries import OutcomeEntry, Provenance, reuse_context, scenario_id
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
    # The candidate today's engine contradicts is dropped rather than carried by the one that
    # fits (finding 9 of the 2026-10-12 review); the one kept keeps its claims, so its
    # narrative still matches.
    assert [c.indicator_deltas for c in out] == [fits]
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


# Review of 13 Oct 2026, E23: stored outcomes were matched on the actions and sackings only,
# so a narrative written for a consulted, phased package, a broken pledge or a bill the
# Commons blocked could be replayed for a turn where none of that happened.

CONSULTED = Delivery(consulted=["sector:energy"], speed="phased")
BLOCKED = "Blocked: spend sector:public (the Commons would not pass it; support 0.31)"


def forecaster_with(tmp_path, context):
    world = toy_world()
    engine = engine_forecast(world)
    store = KnowledgeStore(tmp_path / "k.db")
    entry = OutcomeEntry(
        scenario_id=scenario_id(SCENARIO),
        scenario_title=SCENARIO.title,
        affected_nodes=[],
        response="windfall tax",
        actions=[TAX],
        candidates=[outcome("Stored", claims_matching(engine), 0.9)],
        chosen=0,
        provenance=Provenance(model="fake"),
        context=context,
    )
    store.put("outcome", "k1", entry, scenario=entry.scenario_id)
    return StoredForecaster(store, EngineForecaster()), world, engine


@pytest.mark.parametrize(
    ("stored", "delivery", "limits"),
    [
        (reuse_context(CONSULTED, [], []), None, []),
        (reuse_context(None, [], [BLOCKED]), None, []),
        (reuse_context(None, [], []), CONSULTED, []),
        (None, CONSULTED, []),
    ],
)
def test_outcomes_from_another_context_are_not_reused(tmp_path, stored, delivery, limits):
    forecaster, world, engine = forecaster_with(tmp_path, stored)
    forecaster.forecast(world, SCENARIO, [TAX], engine, delivery=delivery, limits=limits)
    assert forecaster.last_source == "fallback"


def test_outcomes_from_the_same_context_are_reused(tmp_path) -> None:
    # The reason for a block carries numbers that change; the measure blocked is what counts.
    stored = reuse_context(CONSULTED, [], [BLOCKED])
    forecaster, world, engine = forecaster_with(tmp_path, stored)
    later = BLOCKED.replace("0.31", "0.28")
    forecaster.forecast(world, SCENARIO, [TAX], engine, delivery=CONSULTED, limits=[later])
    assert forecaster.last_source == "stored"


def test_old_entries_are_reused_only_on_a_plain_turn(tmp_path) -> None:
    forecaster, world, engine = forecaster_with(tmp_path, None)
    forecaster.forecast(world, SCENARIO, [TAX], engine)
    assert forecaster.last_source == "stored"


def test_a_broken_pledge_is_part_of_the_context(tmp_path) -> None:
    forecaster, world, engine = forecaster_with(tmp_path, reuse_context(None, [], []))
    world.pledges.append(Pledge(text="No new taxes", kind="tax", direction="up", made_turn=0))
    forecaster.forecast(world, SCENARIO, [TAX], engine)
    assert forecaster.last_source == "fallback"


def test_harvest_files_the_turn_s_context(tmp_path) -> None:
    from hog_sim.core.config import GameConfig
    from hog_sim.game.loop import Game
    from hog_sim.game.stubs import CannedScenarios, KeywordInterpreter

    config = GameConfig(seed=0, election_turn=6, k_draws=5)
    plugins = (CannedScenarios(0, calendar=False), KeywordInterpreter(), EngineForecaster())
    game = Game(config, toy_world(), *plugins)
    game.play_turn("Cap household bills. We will not raise taxes.")
    before = game.state
    record = game.play_turn("Put a windfall tax on energy companies")
    store = KnowledgeStore(tmp_path / "k.db")
    store.harvest_turn("g1", record, before)
    (entry,) = store.outcomes(scenario_id(record.scenario))
    assert entry.context.broken_pledges == ["We will not raise taxes"]
