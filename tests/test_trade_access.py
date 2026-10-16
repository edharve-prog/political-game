"""Trade deals move access, not the partner's economy (M3, Ed's review of 13 October 2026)."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import EdgeKind, Outcome, PolicyAction, Scenario
from hog_sim.game.loop import resolve
from hog_sim.world.changes import COMMUNICATE_RELATIONSHIP, trade_factor
from hog_sim.world.propagation import (
    DIPLOMATIC_TRADE_STEPS,
    access_links,
    actions_to_shocks,
)
from hog_sim.world.seed.toy import toy_world

NOTHING = Outcome(narrative="Nothing else happens", probability=1)
QUIET = Scenario(title="Quiet", briefing="", affected_nodes=[], urgency=0)
EU, UK, FACTORIES = "country:eu", "country:uk", "sector:manufacturing"


def act(kind: str, target: str, magnitude: float, turns: int = 6) -> PolicyAction:
    return PolicyAction(kind=kind, target=target, magnitude=magnitude, duration_turns=turns)


def turns(actions, n: int = 4):
    start = state = toy_world()
    out = []
    for t in range(n):
        state = resolve(start, state, QUIET, actions if t == 0 else [], NOTHING, GameConfig())
        out.append(state)
    return out


def test_a_deal_pushes_the_players_exposed_nodes_not_the_partner():
    world = toy_world()
    shocks = actions_to_shocks([act("diplomatic", EU, 0.8)], world)
    nodes = {s.node: s.delta for s in shocks}
    assert EU not in nodes
    for edge in access_links(world, EU):
        assert nodes[edge.target] == pytest.approx(DIPLOMATIC_TRADE_STEPS * 0.8 * edge.weight)
    assert {"indicator:output_gap", FACTORIES} <= set(nodes)  # UK output (O3)


def test_a_deal_leaves_the_partners_growth_almost_alone_and_helps_the_player():
    start = toy_world()
    after = turns([act("diplomatic", EU, 0.8)])[-1]
    quiet = turns([])[-1]
    # Only the UK's own (small) pull on EU trade reaches the EU.
    assert abs(after.countries[EU].growth_pct - quiet.countries[EU].growth_pct) < 0.05
    assert after.countries[UK].growth_pct > quiet.countries[UK].growth_pct
    assert after.sectors[FACTORIES].output_bn > quiet.sectors[FACTORIES].output_bn
    assert start.countries[EU].growth_pct == quiet.countries[EU].growth_pct


def test_a_speech_abroad_moves_only_the_relationship():
    world = toy_world()
    speech = act("communicate", "country:china", 1.0, turns=1)
    assert actions_to_shocks([speech], world) == []
    after = turns([speech], 1)[0]
    china = world.countries["country:china"]
    assert after.countries["country:china"].relationship == pytest.approx(
        china.relationship + COMMUNICATE_RELATIONSHIP
    )
    assert after.countries["country:china"].growth_pct == china.growth_pct


def test_the_relationship_scales_market_access_as_well_as_trade():
    world = toy_world()
    before = {(e.source, e.target): e.weight for e in access_links(world, EU)}
    after = turns([act("diplomatic", EU, 1.0, turns=1)], 1)[0]
    old = world.countries[EU].relationship
    new = after.countries[EU].relationship
    ratio = trade_factor(new) / trade_factor(old)
    for e in access_links(after, EU):
        assert e.weight == pytest.approx(before[(e.source, e.target)] * ratio)
    assert any(e.kind == EdgeKind.SUPPLIES for e in access_links(after, EU))
