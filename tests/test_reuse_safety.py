"""Stored work is reused only for what means the same thing (Ed's 2026-10-12 review,
findings 4 and 9): directions stay bound to their own targets, outcome signatures carry
duration and the tax or spending direction, and every reused candidate must fit today."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction, Scenario
from hog_sim.forecasting.candidates import CLAIM_TURN
from hog_sim.game.loop import forecast
from hog_sim.game.stubs import EngineForecaster
from hog_sim.knowledge.entries import OutcomeEntry, Provenance, action_signature, scenario_id
from hog_sim.knowledge.offline import StoredForecaster, same_request
from hog_sim.knowledge.store import KnowledgeStore
from hog_sim.world.seed.toy import toy_world

PACKAGE = "raise taxes on business and cut taxes on pensioners"


@pytest.mark.parametrize(
    "text",
    [
        "raise taxes on pensioners and cut taxes on business",
        "cut taxes on business and raise taxes on pensioners",
        "raise taxes on business by 5% and cut taxes on pensioners by 10%",
    ],
)
def test_swapped_beneficiaries_never_match(text) -> None:
    assert same_request(PACKAGE, text) == 0.0


@pytest.mark.parametrize(
    "text",
    [
        "cut taxes on pensioners and raise taxes on business",
        "Raise taxes on businesses, and cut the taxes on pensioners.",
    ],
)
def test_the_same_package_reordered_or_reworded_still_matches(text) -> None:
    assert same_request(PACKAGE, text) >= 0.5


def test_numbers_stay_with_their_own_clause() -> None:
    a = "raise taxes on business by 5% and cut taxes on pensioners by 10%"
    b = "raise taxes on business by 10% and cut taxes on pensioners by 5%"
    assert same_request(a, b) == 0.0


PENSION = PolicyAction(kind="spend", target="group:pensioners", magnitude=0.5)
BILLS = PolicyAction(kind="spend", target="indicator:energy_prices", magnitude=-0.5)


def test_signature_tells_a_one_turn_measure_from_a_long_policy() -> None:
    long = PENSION.model_copy(update={"duration_turns": 24})
    assert action_signature([PENSION]) != action_signature([long])
    six = PENSION.model_copy(update={"duration_turns": 6})
    five = PENSION.model_copy(update={"duration_turns": 5})
    assert action_signature([six]) == action_signature([five])


def test_signature_tells_a_spending_rise_from_a_cut() -> None:
    cut = BILLS.model_copy(update={"instrument": "cut"})
    raised = BILLS.model_copy(update={"instrument": "raise"})
    assert action_signature([BILLS]) != action_signature([cut])
    assert action_signature([BILLS]) == action_signature([raised])  # unset reads as raise


SCENARIO = Scenario(
    title="Pensioners squeezed", briefing="Bills bite.", affected_nodes=[], urgency=0.5
)
CONFIG = GameConfig(k_draws=20)


def stored_set(tmp_path, candidates: list[Outcome], sacked: list[str] | None = None):
    store = KnowledgeStore(tmp_path / "game.db")
    sid = scenario_id(SCENARIO)
    entry = OutcomeEntry(
        scenario_id=sid,
        scenario_title=SCENARIO.title,
        affected_nodes=[],
        response="raise the state pension",
        actions=[PENSION],
        candidates=candidates,
        chosen=0,
        provenance=Provenance(game_id="g1", turn=0),
        sacked=sacked or [],
    )
    store.put("outcome", f"{sid}|{action_signature([PENSION])}", entry, sid)
    return store


def candidates_for(engine):
    turn = min(CLAIM_TURN, engine.horizon - 1)
    fits = {n: f.mean[turn] for n, f in engine.nodes.items() if n.startswith("indicator:")}
    wild = {n: v + 50.0 for n, v in fits.items()}
    return [
        Outcome(narrative="As the engine expects.", indicator_deltas=fits, probability=0.5),
        Outcome(narrative="Something else entirely.", indicator_deltas=wild, probability=0.5),
    ]


def test_only_candidates_that_fit_today_are_reused(tmp_path) -> None:
    state = toy_world()
    engine = forecast(state, state, SCENARIO, [PENSION], CONFIG)
    store = stored_set(tmp_path, candidates_for(engine))
    forecaster = StoredForecaster(store, EngineForecaster())
    outcomes = forecaster.forecast(state, SCENARIO, [PENSION], engine)
    assert forecaster.last_source == "stored"
    assert [o.narrative for o in outcomes] == ["As the engine expects."]
    assert outcomes[0].probability == pytest.approx(1.0)


def test_outcomes_from_a_turn_with_other_sackings_are_not_reused(tmp_path) -> None:
    state = toy_world()
    engine = forecast(state, state, SCENARIO, [PENSION], CONFIG)
    store = stored_set(tmp_path, candidates_for(engine), sacked=["person:chancellor"])
    forecaster = StoredForecaster(store, EngineForecaster())
    forecaster.forecast(state, SCENARIO, [PENSION], engine)
    assert forecaster.last_source == "fallback"
    forecaster.forecast(state, SCENARIO, [PENSION], engine, sacked=["person:chancellor"])
    assert forecaster.last_source == "stored"
