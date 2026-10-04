import math

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import PolicyAction
from hog_sim.forecasting.candidates import (
    CandidateDraft,
    CandidateSet,
    ForecastConfig,
    GroupEffect,
    Judgement,
    LLMForecaster,
    NewShock,
    Shift,
    Verdict,
    engine_text,
)
from hog_sim.forecasting.scoring import (
    BASE_RATES,
    Z_CAP,
    CandidateScores,
    ScoreWeights,
    base_rate,
    combine,
    consistency,
)
from hog_sim.game.interfaces import NeedsClarification
from hog_sim.game.llm_plugins import llm_plugins
from hog_sim.game.loop import Game, replay
from hog_sim.game.stubs import CannedScenarios
from hog_sim.llm.client import FakeClient, LLMOutputError
from hog_sim.llm.interpreter import Interpretation
from hog_sim.llm.scenario_gen import ScenarioDraft, SideIssueDraft, StakeholderPosition
from hog_sim.world.propagation import actions_to_shocks, propagate
from hog_sim.world.seed.toy import toy_world

TAX = PolicyAction(kind="tax", target="sector:energy", magnitude=0.5, rationale="windfall tax")


@pytest.fixture
def world():
    return toy_world()


@pytest.fixture
def engine(world):
    return propagate(world, actions_to_shocks([TAX]), horizon=6, k_draws=100)


def draft(title, inflation, tags=("none",), p=0.25, effects=(), shocks=(), changes=()):
    return CandidateDraft(
        title=title,
        narrative=f"{title} happened.",
        indicator_shifts=[Shift(node="indicator:inflation", change=inflation)],
        group_effects=[GroupEffect(group=g, change=c) for g, c in effects],
        new_shocks=[NewShock(node=n, steps=s) for n, s in shocks],
        graph_changes=list(changes),
        event_tags=list(tags),
        resolves_storyline=False,
        self_probability=p,
    )


def four_candidates(engine):
    expected = engine.nodes["indicator:inflation"].mean[2]
    return CandidateSet(
        candidates=[
            draft("As expected", expected, p=0.5),
            draft("Pensioner anger", expected, ["protest"], 0.2, [("group:pensioners", -0.04)]),
            draft("Market panic", expected + 3.0, ["market_selloff", "capital_flight"], 0.2),
            draft("Strike wave", expected, ["strike"], 0.1, shocks=[("sector:public", -0.5)]),
        ]
    )


def verdict(*ps):
    return Verdict(
        judgements=[Judgement(index=i, probability=p, reason="r") for i, p in enumerate(ps)]
    )


# --- scoring ---------------------------------------------------------------


def test_consistency_prefers_claims_near_the_engine(engine) -> None:
    expected = {n: f.mean[2] for n, f in engine.nodes.items() if n.startswith("indicator:")}
    near = consistency(expected, engine, 2)
    far = consistency(
        {**expected, "indicator:inflation": expected["indicator:inflation"] + 3}, engine, 2
    )
    assert near == pytest.approx(1.0)
    assert far < near
    assert consistency({**expected, "indicator:not_real": 5}, engine, 2) == pytest.approx(1.0)


def test_leaving_out_material_moves_is_not_free(engine) -> None:
    """Finding 8: a candidate with no claims used to score a perfect 1.0."""
    expected = {n: f.mean[2] for n, f in engine.nodes.items() if n.startswith("indicator:")}
    assert any(abs(v) > 0.5 for v in expected.values())
    assert consistency({}, engine, 2) < consistency(expected, engine, 2)
    quiet = engine.model_copy(deep=True)
    for f in quiet.nodes.values():
        f.mean, f.p10, f.p90 = [0.0] * len(f.mean), [0.0] * len(f.mean), [0.0] * len(f.mean)
    assert consistency({}, quiet, 2) == 1.0


def test_base_rates() -> None:
    assert base_rate([]) == BASE_RATES["none"]
    assert base_rate(["capital_flight"]) < base_rate(["media_backlash"])


def test_adding_tags_never_makes_an_outcome_more_common() -> None:
    """Finding 9: under a geometric mean, adding "none" or a common tag raised the score."""
    rare = base_rate(["market_selloff"])
    assert base_rate(["none", "market_selloff"]) == rare
    assert base_rate(["market_selloff", "media_backlash"]) <= rare


def test_combine_normalises_and_respects_weights() -> None:
    a = CandidateScores(consistency=1.0, judge=0.2, base_rate=0.5, self_reported=0.5)
    b = CandidateScores(consistency=0.1, judge=0.8, base_rate=0.5, self_reported=0.5)
    even = combine([a, b], ScoreWeights())
    assert sum(s.probability for s in even) == pytest.approx(1.0)
    engine_only = combine([a, b], ScoreWeights(consistency=1, judge=0, base_rate=0))
    judge_only = combine([a, b], ScoreWeights(consistency=0, judge=1, base_rate=0))
    assert engine_only[0].probability > engine_only[1].probability
    assert judge_only[1].probability > judge_only[0].probability


# --- forecaster ------------------------------------------------------------


def test_forecaster_scores_and_converts(world, engine) -> None:
    client = FakeClient([four_candidates(engine), verdict(0.4, 0.3, 0.2, 0.1)])
    scenario = CannedScenarios().next_scenario(world, [])
    outcomes = LLMForecaster(client).forecast(world, scenario, [TAX], engine)

    assert [r.schema_name for r in client.requests] == ["CandidateSet", "Verdict"]
    assert "indicator:inflation" in client.requests[0].messages[0].content
    assert sum(o.probability for o in outcomes) == pytest.approx(1.0)
    expected, anger, panic, strike = outcomes
    # The panic contradicts the engine and has rare events, so it scores lowest
    assert panic.probability == min(o.probability for o in outcomes)
    assert expected.probability > anger.probability
    assert anger.approval_events[0].group_effects == {"group:pensioners": -0.04}
    assert strike.shocks[0].node == "sector:public" and strike.shocks[0].delta == -0.5
    assert set(expected.scores) == {"consistency", "judge", "base_rate", "self_reported"}
    assert panic.events == ["market_selloff", "capital_flight"]


def test_forecaster_without_judge_uses_self_reports(world, engine) -> None:
    client = FakeClient([four_candidates(engine)])
    scenario = CannedScenarios().next_scenario(world, [])
    config = ForecastConfig(use_judge=False)
    outcomes = LLMForecaster(client, config=config).forecast(world, scenario, [TAX], engine)
    assert len(client.requests) == 1
    assert outcomes[0].scores["judge"] == 0.5


def test_forecaster_retries_bad_candidates(world, engine) -> None:
    bad = four_candidates(engine)
    bad.candidates[0].indicator_shifts[0].node = "indicator:vibes"
    client = FakeClient([bad, four_candidates(engine), verdict(0.25, 0.25, 0.25, 0.25)])
    scenario = CannedScenarios().next_scenario(world, [])
    LLMForecaster(client).forecast(world, scenario, [TAX], engine)
    assert "indicator:vibes" in client.requests[1].messages[-1].content

    wrong_count = CandidateSet(candidates=four_candidates(engine).candidates[:2])
    with pytest.raises(LLMOutputError):
        LLMForecaster(FakeClient([wrong_count] * 3)).forecast(world, scenario, [TAX], engine)


def test_bare_ids_in_candidates_are_prefixed_not_retried(world, engine) -> None:
    """RB-9: the review's most common retry was a bare group id."""
    sloppy = four_candidates(engine)
    sloppy.candidates[0].indicator_shifts[0].node = "inflation"
    sloppy.candidates[1].group_effects[0].group = "pensioners"
    sloppy.candidates[3].new_shocks[0].node = "public"
    client = FakeClient([sloppy, verdict(0.25, 0.25, 0.25, 0.25)])
    scenario = CannedScenarios().next_scenario(world, [])
    outcomes = LLMForecaster(client).forecast(world, scenario, [TAX], engine)
    assert [r.schema_name for r in client.requests] == ["CandidateSet", "Verdict"]
    assert "indicator:inflation" in outcomes[0].indicator_deltas
    assert "group:pensioners" in outcomes[1].approval_events[0].group_effects
    assert outcomes[3].shocks[0].node == "sector:public"


def test_none_is_the_only_tag_when_used(world, engine) -> None:
    bad = four_candidates(engine)
    bad.candidates[0].event_tags = ["none", "market_selloff"]
    client = FakeClient([bad, four_candidates(engine), verdict(0.25, 0.25, 0.25, 0.25)])
    scenario = CannedScenarios().next_scenario(world, [])
    LLMForecaster(client).forecast(world, scenario, [TAX], engine)
    assert "'none' cannot be combined" in client.requests[1].messages[-1].content


def test_engine_text_lists_moving_indicators(world, engine) -> None:
    text = engine_text(world, engine)
    assert "indicator:energy_prices" in text and "indicator:deficit" not in text


# --- full game on a fake client --------------------------------------------


def scripted_responder(engine):
    """Answers every LLM call the game makes, by schema."""
    scenario = ScenarioDraft(
        title="Energy bills soar",
        category="energy",
        briefing="Bills are up and pensioners are worried.",
        affected_nodes=["indicator:energy_prices", "group:pensioners"],
        urgency=0.7,
        storyline="new",
        secondary=[
            SideIssueDraft(
                title="Rents climb again",
                category="housing",
                briefing="Young renters face another rise.",
                affected_nodes=["sector:housing", "group:young_renters"],
                urgency=0.4,
                storyline="new",
            )
        ],
        suggested_options=["Tax windfall profits", "Do nothing"],
        stakeholder_positions=[
            StakeholderPosition(node="group:pensioners", stance=0.8, statement="Help us."),
            StakeholderPosition(node="sector:energy", stance=-0.6, statement="Hands off."),
        ],
    )

    def respond(request):
        if request.schema_name == "ScenarioDraft":
            return scenario
        if request.schema_name == "Interpretation":
            text = request.messages[0].content
            if "sort it out" in text:
                return Interpretation(clarifying_question="Which bills, and how?")
            return Interpretation(actions=[TAX])
        if request.schema_name == "CandidateSet":
            return four_candidates(engine)
        if request.schema_name == "Verdict":
            return verdict(0.4, 0.3, 0.2, 0.1)
        raise AssertionError(request.schema_name)

    return respond


def test_game_runs_on_llm_plugins_and_replays(world, engine) -> None:
    client = FakeClient(responder=scripted_responder(engine))
    config = GameConfig(election_turn=4, k_draws=20)
    game = Game(config, world, *llm_plugins(client))
    assert game.scenario.title == "Energy bills soar"

    with pytest.raises(NeedsClarification, match="Which bills"):
        game.play_turn("sort it out")
    assert game.state.turn == 0 and game.history == []

    for _ in range(4):
        record = game.play_turn("Put a windfall tax on energy companies")
    assert game.over and record.election is not None
    assert all(len(r.candidates) == 4 for r in game.history)
    calls = [r.schema_name for r in client.requests]
    assert calls.count("CandidateSet") == 4 and calls.count("Verdict") == 4
    assert replay(world, config, game.history) == game.state


def test_one_far_off_claim_is_a_penalty_not_a_veto(engine) -> None:
    """EB-13: an uncapped z made consistency alone pick the outcome in the 30-turn review."""
    expected = {n: f.mean[2] for n, f in engine.nodes.items() if n.startswith("indicator:")}
    wild = {n: v + 1000 for n, v in expected.items()}
    assert consistency(wild, engine, 2) == pytest.approx(math.exp(-(Z_CAP**2) / 2))

    # Turn 0 of the review: the judge preferred a candidate that missed the engine's numbers.
    worst = math.exp(-(Z_CAP**2) / 2)
    turn0 = [
        CandidateScores(consistency=worst, judge=0.45, base_rate=0.5, self_reported=0.4),
        CandidateScores(consistency=worst, judge=0.15, base_rate=0.07, self_reported=0.25),
        CandidateScores(consistency=worst, judge=0.2, base_rate=0.18, self_reported=0.2),
        CandidateScores(consistency=1.0, judge=0.2, base_rate=0.25, self_reported=0.15),
    ]
    probabilities = [s.probability for s in combine(turn0, ScoreWeights())]
    assert max(probabilities) < 0.7  # was 0.98
    assert all(p >= 0.03 for p in probabilities)
