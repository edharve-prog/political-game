"""A hidden true world per game (D1, Ed's review of 13 October 2026): resolve plays on edge
weights drawn once per game from their uncertainty, while the forecast keeps the prior."""

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.game.loop import HIDDEN_WORLD_NOTE, forecast, resolve
from hog_sim.world.propagation import (
    TRUE_WORLD_SIGMAS,
    _propagating_edges,
    true_draws,
    true_weights,
    world_surprises,
)
from hog_sim.world.seed.toy import toy_world

NOTHING = Outcome(narrative="Nothing else happens", probability=1)
QUIET = Scenario(title="Quiet", briefing="", affected_nodes=[], urgency=0)
SPEND = [PolicyAction(kind="spend", target="sector:public", magnitude=0.8, duration_turns=6)]
INFLATION = "indicator:inflation"


def play(config: GameConfig, turns: int = 8, actions=SPEND) -> list[WorldState]:
    start = state = toy_world()
    states = []
    for t in range(turns):
        state = resolve(start, state, QUIET, list(actions) if t == 0 else [], NOTHING, config)
        states.append(state)
    return states


def values(state: WorldState) -> dict[str, float]:
    return {i.id: i.value for i in state.indicators.values()} | {
        s.id: s.output_bn for s in state.sectors.values()
    }


def test_draws_are_fixed_per_seed_and_within_the_guarded_range():
    world = toy_world()
    assert true_draws(world, 7) == true_draws(world, 7)
    assert true_draws(world, 7) != true_draws(world, 8)
    assert all(abs(z) <= TRUE_WORLD_SIGMAS for z in true_draws(world, 7).values())


def test_true_weights_stay_put_for_edges_without_uncertainty():
    world = toy_world()
    truth = true_weights(world, 3)
    for i, edge in enumerate(_propagating_edges(world)):
        if not edge.uncertainty:
            assert truth[i] == edge.weight
    assert any(truth[i] != e.weight for i, e in enumerate(_propagating_edges(world)))


def test_a_reweighted_edge_keeps_its_draw():
    world = toy_world()
    edges = _propagating_edges(world)
    i, edge = next((i, e) for i, e in enumerate(edges) if e.uncertainty)
    before = true_weights(world, 5)[i] - edge.weight
    changed = world.snapshot()
    next(e for e in changed.edges if e == edge).weight += 0.1
    moved = _propagating_edges(changed)[i]
    assert moved.weight != edge.weight
    assert abs(true_weights(changed, 5)[i] - moved.weight - before) < 1e-12


def test_off_by_default_so_resolve_follows_the_central_weights():
    assert GameConfig().hidden_world is False
    central = play(GameConfig(seed=1))
    assert values(central[-1]) == values(play(GameConfig(seed=2))[-1])


def test_a_hidden_world_plays_differently_by_game_and_reproduces_by_seed():
    one, again, other = (play(GameConfig(seed=s, hidden_world=True)) for s in (1, 1, 2))
    central = play(GameConfig(seed=1))
    assert values(one[-1]) == values(again[-1])
    assert values(one[-1]) != values(other[-1])
    assert values(one[-1]) != values(central[-1])


def test_the_forecast_keeps_the_prior_and_says_so():
    start = toy_world()
    on = GameConfig(seed=1, hidden_world=True, k_draws=20, horizon=4)
    off = on.model_copy(update={"hidden_world": False})
    shown = forecast(start, start, QUIET, SPEND, on)
    assert shown.nodes == forecast(start, start, QUIET, SPEND, off).nodes
    assert HIDDEN_WORLD_NOTE in shown.unprojected
    assert HIDDEN_WORLD_NOTE not in forecast(start, start, QUIET, SPEND, off).unprojected


def test_surprises_name_the_links_that_differ_most():
    world = toy_world()
    found = world_surprises(world, 4, n=3)
    assert len(found) == 3
    gaps = [abs(truth - edge.weight) for edge, truth in found]
    assert gaps == sorted(gaps, reverse=True)
