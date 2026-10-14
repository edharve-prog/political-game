"""An outcome's shocks are capped per node, however they are written (Ed's review, 2026-10)."""

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, Scenario, Shock
from hog_sim.forecasting.candidates import CandidateDraft, to_outcome
from hog_sim.forecasting.scoring import CandidateScores
from hog_sim.game.loop import resolve
from hog_sim.world.events import MAX_OUTCOME_STEPS, cap_shocks, event_shocks, outcome_shocks
from hog_sim.world.seed.toy import toy_world

SCENARIO = Scenario(title="t", briefing="b", affected_nodes=[], urgency=0)
RATE = "indicator:interest_rate"


def total(shocks: list[Shock], node: str) -> float:
    return sum(s.delta for s in shocks if s.node == node)


def test_repeated_shocks_do_not_add_up_past_the_cap() -> None:
    world = toy_world()
    outcome = Outcome(
        narrative="x", probability=1, shocks=[Shock(node=RATE, delta=1.0) for _ in range(5)]
    )
    assert total(outcome_shocks(outcome, world), RATE) == MAX_OUTCOME_STEPS

    once = Outcome(narrative="x", probability=1, shocks=[Shock(node=RATE, delta=1.0)])
    after = {}
    for name, o in {"five": outcome, "once": once}.items():
        state = resolve(world, world, SCENARIO, [], o, GameConfig())
        after[name] = state.indicators[RATE].value
    assert after["five"] == after["once"]


def test_a_repeated_tag_counts_once() -> None:
    world = toy_world()
    assert event_shocks(["strike"] * 4, world) == event_shocks(["strike"], world)


def test_tags_and_shocks_on_one_node_share_the_cap() -> None:
    world = toy_world()
    outcome = Outcome(
        narrative="x",
        probability=1,
        events=["market_selloff", "capital_flight"],  # finance -0.5 and -0.8
        shocks=[Shock(node="sector:finance", delta=-1.0)],
    )
    shocks = outcome_shocks(outcome, world)
    assert abs(total(shocks, "sector:finance") + MAX_OUTCOME_STEPS) < 1e-9
    # Gilt yields (+0.5 and +0.4) are within the cap and untouched.
    assert abs(total(shocks, "indicator:gilt_yield") - 0.9) < 1e-9


def test_cap_keeps_opposite_shocks_and_shape() -> None:
    shocks = [
        Shock(node="a", delta=1.0, duration_turns=3),
        Shock(node="a", delta=1.0, start_turn=2),
        Shock(node="b", delta=0.5),
        Shock(node="b", delta=-0.5),
    ]
    capped = cap_shocks(shocks)
    assert [s.delta for s in capped[:2]] == [0.5, 0.5]
    assert [(s.duration_turns, s.start_turn) for s in capped[:2]] == [(3, 0), (1, 2)]
    assert [s.delta for s in capped[2:]] == [0.5, -0.5]


def test_candidates_are_capped_and_tags_deduplicated() -> None:
    draft = CandidateDraft(
        title="Selloff",
        narrative="Markets fall.",
        indicator_shifts=[],
        group_effects=[],
        new_shocks=[{"node": RATE, "steps": 1.0}, {"node": RATE, "steps": 1.0}],
        graph_changes=[],
        event_tags=["market_selloff", "market_selloff"],
        resolves_storyline=False,
        self_probability=0.5,
    )
    outcome = to_outcome(
        draft, CandidateScores(consistency=1, judge=1, base_rate=1, self_reported=1)
    )
    assert outcome.events == ["market_selloff"]
    assert total(outcome.shocks, RATE) == MAX_OUTCOME_STEPS
