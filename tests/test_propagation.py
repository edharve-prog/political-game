import pytest

from hog_sim.core.models import Edge, EdgeKind, GraphChange, PolicyAction
from hog_sim.world.changes import check_graph_change
from hog_sim.world.propagation import (
    MAX_LAG0_GAIN,
    PropagationError,
    Shock,
    actions_to_shocks,
    apply_deltas,
    lag0_gain,
    propagate,
    simulate,
)
from hog_sim.world.seed.toy import toy_world


@pytest.fixture
def world():
    return toy_world()


def energy_tax():
    return actions_to_shocks([PolicyAction(kind="tax", target="sector:energy", magnitude=0.5)])


def test_no_shocks_means_no_change(world) -> None:
    run = simulate(world, [], horizon=120)
    assert all(v == 0 for series in run.values() for v in series)


def test_energy_tax_has_sensible_signs_and_lags(world) -> None:
    run = simulate(world, energy_tax(), horizon=24)
    assert run["sector:energy"][0] < 0
    # Prices respond the same turn, inflation one turn later, Bank Rate after that
    assert run["indicator:energy_prices"][0] > 0
    assert run["indicator:inflation"][0] == 0
    assert run["indicator:inflation"][1] > 0
    assert run["indicator:interest_rate"][1] == 0
    assert run["indicator:interest_rate"][2] > 0
    # Higher rates feed back on inflation after six turns, pulling it down from its peak
    assert run["indicator:inflation"][8] < run["indicator:inflation"][7]
    # Bank Rate moves at turn 2 and reaches house prices three turns later
    assert run["indicator:house_prices"][4] == 0
    assert run["indicator:house_prices"][5] < 0


def test_effects_settle_rather_than_explode(world) -> None:
    run = simulate(world, energy_tax(), horizon=120)
    for series in run.values():
        assert abs(series[-1] - series[-2]) < 1e-6
        assert max(abs(v) for v in series) < 10


def test_shock_duration_and_start(world) -> None:
    run = simulate(world, [Shock(node="sector:public", delta=1, start_turn=2, duration_turns=3)], 8)
    # One step a turn for three turns, each turn keeping 0.9 of the push, then fading
    assert run["sector:public"] == pytest.approx([0, 0, 1, 1.9, 2.71, 2.439, 2.1951, 1.97559])


def test_held_shock_stays_put_then_fades(world) -> None:
    held = Shock(node="sector:public", delta=1, duration_turns=4, hold=True)
    run = simulate(world, [held], 7)
    assert run["sector:public"] == pytest.approx([1, 1, 1, 1, 0.9, 0.81, 0.729])


def test_a_held_driver_keeps_what_it_drives_moved(world) -> None:
    held = Shock(node="sector:energy", delta=1, duration_turns=12, hold=True)
    run = simulate(world, [held], 24)
    prices = run["indicator:energy_prices"]
    assert prices[0] < 0
    assert prices[5] == pytest.approx(prices[0])
    assert abs(prices[-1]) < abs(prices[0]) / 3


def test_monte_carlo_is_reproducible_and_brackets_mean(world) -> None:
    a = propagate(world, energy_tax(), horizon=6, k_draws=50, seed=7)
    b = propagate(world, energy_tax(), horizon=6, k_draws=50, seed=7)
    c = propagate(world, energy_tax(), horizon=6, k_draws=50, seed=8)
    assert a == b
    assert a != c
    infl = a.nodes["indicator:inflation"]
    assert infl.p10[3] <= infl.mean[3] <= infl.p90[3]
    assert infl.p10[3] < infl.p90[3]


def test_native_units(world) -> None:
    dist = propagate(world, [Shock(node="sector:energy", delta=-1)], horizon=1, k_draws=1)
    # One step on Energy is 5% of its 80bn output
    assert dist.at(0)["sector:energy"] == pytest.approx(-4.0)


def test_apply_deltas_returns_new_state(world) -> None:
    dist = propagate(world, energy_tax(), horizon=3, k_draws=20)
    after = apply_deltas(world, dist.at(2))
    assert after.indicators["indicator:energy_prices"].value > 100
    assert world.indicators["indicator:energy_prices"].value == 100
    assert "indicator:inflation" in world.diff(after)


def test_unknown_shock_node_raises(world) -> None:
    with pytest.raises(KeyError):
        simulate(world, [Shock(node="sector:space", delta=1)], 3)


def test_spending_and_tax_move_the_deficit(world) -> None:
    spend = actions_to_shocks(
        [PolicyAction(kind="spend", target="sector:housing", magnitude=0.5)], world
    )
    tax = actions_to_shocks(
        [PolicyAction(kind="tax", target="sector:finance", magnitude=0.5)], world
    )
    talk = actions_to_shocks(
        [PolicyAction(kind="communicate", target="sector:housing", magnitude=0.5)], world
    )
    deficit = {s.node: s.delta for s in spend}["indicator:deficit"]
    assert deficit > 0
    assert {s.node: s.delta for s in tax}["indicator:deficit"] == -deficit
    assert "indicator:deficit" not in {s.node for s in talk}
    # Without a state there is no fiscal shock, so worlds without a deficit still work
    assert (
        len(actions_to_shocks([PolicyAction(kind="spend", target="sector:housing", magnitude=0.5)]))
        == 1
    )


def test_borrowing_raises_rates(world) -> None:
    run = simulate(world, [Shock(node="indicator:deficit", delta=1)], horizon=4)
    assert run["indicator:interest_rate"][0] == 0
    assert run["indicator:interest_rate"][1] > 0


def _loop(world, a, b, weight):
    world.edges += [
        Edge(source=a, target=b, kind=EdgeKind.SUPPLIES, weight=weight),
        Edge(source=b, target=a, kind=EdgeKind.SUPPLIES, weight=weight),
    ]
    return world


def test_a_stable_same_turn_loop_settles(world) -> None:
    world = _loop(world, "sector:finance", "sector:public", 0.5)
    assert lag0_gain(world) < MAX_LAG0_GAIN
    path = simulate(world, [Shock(node="sector:finance", delta=1.0)], horizon=2)
    # x = 1 + 0.45 * 0.45 * x  =>  x = 1 / (1 - 0.2025)
    assert path["sector:finance"][0] == pytest.approx(1 / (1 - 0.45**2))


def test_an_unstable_same_turn_loop_raises(world) -> None:
    """Finding 5 / EB-12: 200 iterations used to end silently on a diverging value."""
    world = _loop(world, "sector:finance", "sector:public", 1.2)
    assert lag0_gain(world) > 1
    with pytest.raises(PropagationError):
        simulate(world, [Shock(node="sector:finance", delta=1.0)], horizon=1)


def test_graph_changes_cannot_destabilise_the_loops(world) -> None:
    world.edges.append(
        Edge(
            source="sector:finance",
            target="sector:public",
            kind=EdgeKind.SUPPLIES,
            weight=1.0,
            uncertainty=0.3,
        )
    )
    closing = GraphChange(
        kind="add_edge",
        source="sector:public",
        target="sector:finance",
        edge_kind=EdgeKind.SUPPLIES,
        delta=0.3,
    )
    assert any("unstable" in p for p in check_graph_change(world, closing))
    # The same link with a lag adds no same-turn loop.
    assert check_graph_change(world, closing.model_copy(update={"lag": 2})) == []
