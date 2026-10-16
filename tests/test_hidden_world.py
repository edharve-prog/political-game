"""A hidden true world per game (D1, Ed's review of 13 October 2026): resolve plays on edge
weights drawn once per game from their uncertainty, while the forecast keeps the prior."""

import json

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.game.loop import HIDDEN_WORLD_NOTE, forecast, resolve
from hog_sim.game.persistence import SaveStore
from hog_sim.world.propagation import (
    TRUE_WORLD_SIGMAS,
    _propagating_edges,
    drawn_uncertainty,
    settles,
    taylor_response,
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
    # The game's opening world picks the redraw (FR-24), so a lasting change later in the game
    # moves one edge's weight and leaves every draw where it was.
    world = toy_world()
    edges = _propagating_edges(world)
    i, edge = next(
        (i, e)
        for i, e in enumerate(edges)
        if e.uncertainty and drawn_uncertainty(e) == e.uncertainty
    )
    before = true_weights(world, 5)[i] - edge.weight
    changed = world.snapshot()
    next(e for e in changed.edges if e == edge).weight += 0.1
    moved = _propagating_edges(changed)[i]
    assert moved.weight != edge.weight
    assert true_draws(changed, 5, start=world) == true_draws(world, 5)
    assert abs(true_weights(changed, 5, start=world)[i] - moved.weight - before) < 1e-12


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


# FR-24 (final review, 16 October 2026): a drawn world used to flip link signs (79% of seeds),
# break the Taylor principle (10%) and, in a few seeds, never settle.
RULE_1_RUNAWAYS = (1, 5, 39, 60, 91)
RULE_1_TAYLOR_BREAKS = (9, 15, 32, 46, 50)


def test_a_drawn_world_keeps_every_sign_the_taylor_principle_and_settles():
    world = toy_world()
    edges = _propagating_edges(world)
    for seed in range(40):
        truth = true_weights(world, seed)
        assert all(truth[i] * e.weight > 0 for i, e in enumerate(edges) if e.weight), seed
        assert taylor_response(world, truth) > 1, seed
        assert settles(world, truth), seed


def test_worlds_that_ran_away_or_broke_the_taylor_principle_are_redrawn():
    world = toy_world()
    for seed in RULE_1_RUNAWAYS:
        assert not settles(world, true_weights(world, seed, draw=1))
        assert settles(world, true_weights(world, seed))
    for seed in RULE_1_TAYLOR_BREAKS:
        assert taylor_response(world, true_weights(world, seed, draw=1)) <= 1
        assert taylor_response(world, true_weights(world, seed)) > 1
    # Redraws come from the seed, so a seed always gives the same world.
    assert true_weights(world, RULE_1_RUNAWAYS[0]) == true_weights(toy_world(), RULE_1_RUNAWAYS[0])


def test_small_links_get_uncertainty_in_proportion_to_their_size():
    for edge in _propagating_edges(toy_world()):
        assert drawn_uncertainty(edge) <= edge.uncertainty
        assert TRUE_WORLD_SIGMAS * drawn_uncertainty(edge) < abs(edge.weight) or not edge.weight
    china = next(e for e in _propagating_edges(toy_world()) if e.source == "country:china")
    assert drawn_uncertainty(china) < china.uncertainty  # 0.03 ± 0.1 could flip; no longer


def test_rule_1_is_the_original_draw_so_old_games_keep_their_weights():
    world = toy_world()
    edges = _propagating_edges(world)
    draws = true_draws(world, 7, draw=1)
    truth = true_weights(world, 7, draw=1)
    for i, (edge, z) in enumerate(zip(edges, draws.values(), strict=True)):
        assert truth[i] == edge.weight + z * edge.uncertainty


def test_saves_from_before_rule_2_load_on_rule_1(tmp_path):
    store = SaveStore(tmp_path / "games.db")
    new = store.new_game(GameConfig(seed=7, hidden_world=True), toy_world())
    assert store.load(new)[0].hidden_world_draw == 2
    old_config = GameConfig(seed=7, hidden_world=True).model_dump()
    del old_config["hidden_world_draw"]
    old = store.new_game(GameConfig(seed=7, hidden_world=True), toy_world())
    store.conn.execute(
        "UPDATE games SET config = ? WHERE game_id = ?", (json.dumps(old_config), old)
    )
    config = store.load(old)[0]
    assert config.hidden_world_draw == 1
    store.close()


def test_resolve_plays_the_games_own_draw_rule():
    old = play(GameConfig(seed=1, hidden_world=True, hidden_world_draw=1))
    new = play(GameConfig(seed=1, hidden_world=True))
    assert values(old[-1]) != values(new[-1])
