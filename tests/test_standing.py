"""EB-8: relationship, stability and sentiment have effects."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import EdgeKind, GraphChange, Outcome, PolicyAction, Scenario
from hog_sim.game.loop import resolve
from hog_sim.world.changes import TRADE_PER_RELATIONSHIP, apply_graph_changes
from hog_sim.world.seed.toy import toy_world
from hog_sim.world.standing import (
    CRISIS_RATE,
    crisis_chance,
    crisis_shocks,
    sentiment_shocks,
    settle_sentiment,
)

QUIET = Scenario(title="Quiet", briefing="Nothing happens.", affected_nodes=[], urgency=0)
NOTHING = Outcome(narrative="Nothing happened.", probability=1.0)


@pytest.fixture
def world():
    return toy_world()


def _trade(state, country):
    return {
        (e.source, e.target): e.weight
        for e in state.edges
        if e.kind == EdgeKind.TRADES_WITH and country in (e.source, e.target)
    }


def test_starting_world_is_untouched(world) -> None:
    assert sentiment_shocks(world) == []
    assert all(crisis_shocks(world, seed) == [] for seed in range(20))
    assert settle_sentiment(world) == world


def test_relationship_scales_trade(world) -> None:
    before = _trade(world, "country:eu")
    assert before
    souring = GraphChange(kind="node_attr", node="country:eu", attr="relationship", delta=-0.2)
    after = _trade(apply_graph_changes(world, [souring]), "country:eu")
    for key, weight in before.items():
        assert after[key] == pytest.approx(weight * (1 - 0.2 * TRADE_PER_RELATIONSHIP))


def test_diplomacy_deepens_trade(world) -> None:
    deal = PolicyAction(kind="diplomatic", target="country:eu", magnitude=1.0)
    new = resolve(world, world, QUIET, [deal], NOTHING, GameConfig())
    assert new.countries["country:eu"].relationship > world.countries["country:eu"].relationship
    before, after = _trade(world, "country:eu"), _trade(new, "country:eu")
    assert all(after[k] > w for k, w in before.items() if w > 0)


def test_sentiment_moves_output_then_fades(world) -> None:
    gloom = GraphChange(kind="node_attr", node="sector:finance", attr="sentiment", delta=-0.2)
    state = apply_graph_changes(world, [gloom])
    calm = world
    for _ in range(3):
        state = resolve(world, state, QUIET, [], NOTHING, GameConfig())
        calm = resolve(world, calm, QUIET, [], NOTHING, GameConfig())
    assert state.sectors["sector:finance"].output_bn < calm.sectors["sector:finance"].output_bn
    assert -0.2 < state.sectors["sector:finance"].sentiment < 0


def test_unstable_countries_fall_into_crisis(world) -> None:
    assert crisis_chance(0.5) == 0 and crisis_chance(0.0) == CRISIS_RATE
    world.countries["country:china"].stability = 0.0
    hits = sum(bool(crisis_shocks(world.model_copy(update={"turn": t}), 1)) for t in range(200))
    assert 0.15 * 200 < hits < 0.45 * 200
    # Replay draws the same crises.
    assert crisis_shocks(world, 7) == crisis_shocks(world, 7)
