"""Real turns and forecasts share one engine step, and the engine carries its own state
between turns (Ed's review at 7be7c65: caps, forecasts, frozen and truncated effects)."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction, Scenario, Shock
from hog_sim.core.state import WorldState
from hog_sim.game.loop import forecast, resolve
from hog_sim.world.changes import apply_graph_changes
from hog_sim.world.propagation import JOB_NODE, propagate, step
from hog_sim.world.seed.toy import toy_world

NOTHING = Outcome(narrative="Nothing else happens", probability=1)
QUIET = Scenario(title="Quiet", briefing="", affected_nodes=[], urgency=0)
CONFIG = GameConfig()
UNEMPLOYMENT, INFLATION = "indicator:unemployment", "indicator:inflation"


def scenario(*shocks: Shock) -> Scenario:
    return QUIET.model_copy(update={"shocks": list(shocks)})


def play(start: WorldState, turns: int, first=QUIET, actions=()) -> list[WorldState]:
    states, state = [], start
    for t in range(turns):
        state = resolve(
            start,
            state,
            first if t == 0 else QUIET,
            list(actions) if t == 0 else [],
            NOTHING,
            CONFIG,
        )
        states.append(state)
    return states


def value(state: WorldState, node: str) -> float:
    return state.indicators[node].value


def test_a_capped_indicator_fades_back_from_the_cap_not_past_its_start() -> None:
    """A big push at 24% unemployment (cap 25%) used to read 25, then 24.1, 23.3 ... 21.7:
    the fade was worked out from 30%, so it fell well below where it began."""
    start = toy_world()
    start.indicators[UNEMPLOYMENT].value = 24
    states = play(start, 12, scenario(Shock(node=UNEMPLOYMENT, delta=6)))
    path = [value(s, UNEMPLOYMENT) for s in states]
    assert path[0] == 25
    assert all(v >= 24 - 1e-9 for v in path)
    assert all(a >= b - 1e-9 for a, b in zip(path, path[1:], strict=False))


def test_a_held_push_keeps_a_node_at_its_bound_then_fades() -> None:
    start = toy_world()
    start.indicators[UNEMPLOYMENT].value = 24
    held = Shock(node=UNEMPLOYMENT, delta=6, duration_turns=4, hold=True)
    path = [value(s, UNEMPLOYMENT) for s in play(start, 8, scenario(held))]
    assert path[:4] == pytest.approx([25] * 4)
    assert 24 <= path[5] < path[4] < 25


def test_the_forecast_includes_effects_already_in_flight() -> None:
    """A queued two-point rise in inflation was forecast as no change at all."""
    start = toy_world()
    state = resolve(
        start, start, scenario(Shock(node=INFLATION, delta=2, start_turn=1)), [], NOTHING, CONFIG
    )
    engine = forecast(start, state, QUIET, [], CONFIG.model_copy(update={"horizon": 3}))
    assert engine.at(0)[INFLATION] == pytest.approx(2.0, rel=0.05)
    # ... and it is the part nobody decided this turn.
    assert engine.response_at(0)[INFLATION] == pytest.approx(0.0, abs=1e-9)


def certain(world: WorldState) -> WorldState:
    for edge in world.edges:
        edge.uncertainty = 0.0
    return world


def test_the_forecast_is_what_resolve_does_with_no_outcome() -> None:
    start = certain(toy_world())
    start.sectors["sector:energy"].sentiment = -0.5  # a background push the forecast must see
    state = play(start, 3, scenario(Shock(node="sector:energy", delta=-1.5)))[-1]
    actions = [
        PolicyAction(kind="spend", target="sector:public", magnitude=0.6, duration_turns=4),
        PolicyAction(kind="tax", target="sector:finance", magnitude=0.4, duration_turns=2),
    ]
    turns = 6
    config = CONFIG.model_copy(update={"horizon": turns, "k_draws": 1})
    engine = forecast(start, state, QUIET, actions, config)
    actual, now = [], state
    for t in range(turns):
        now = resolve(start, now, QUIET, actions if t == 0 else [], NOTHING, config)
        actual.append(now)
    for t, after in enumerate(actual):
        for node in state.indicators:
            expected = value(state, node) + engine.at(t)[node]
            assert value(after, node) == pytest.approx(expected, abs=1e-9), (node, t)


def test_the_forecast_reprices_the_debt_as_resolve_does() -> None:
    start = certain(toy_world())
    state = play(start, 3, scenario(Shock(node="indicator:interest_rate", delta=2)))[-1]
    config = CONFIG.model_copy(update={"horizon": 6, "k_draws": 1})
    engine = forecast(start, state, QUIET, [], config)
    now = state
    for t in range(6):
        now = resolve(start, now, QUIET, [], NOTHING, config)
        for node in state.indicators:
            expected = value(state, node) + engine.at(t)[node]
            assert value(now, node) == pytest.approx(expected, abs=1e-9), (node, t)


def test_a_policy_keeps_working_past_the_forecast_horizon() -> None:
    """Effects used to be worked out 24 turns ahead and then frozen at that level for good."""
    start = toy_world()
    spend = PolicyAction(kind="spend", target="sector:public", magnitude=0.5, duration_turns=36)
    states = play(start, 60, actions=[spend])
    output = [s.sectors["sector:public"].output_bn for s in states]
    base = start.sectors["sector:public"].output_bn
    assert output[30] > base + 10  # still running at turn 30
    assert output[59] - base < (output[35] - base) / 5  # and fading once it stops


def test_a_one_off_shock_fades_away_entirely() -> None:
    start = toy_world()
    states = play(start, 120, scenario(Shock(node="sector:energy", delta=-2)))
    assert states[-1].sectors["sector:energy"].output_bn == pytest.approx(
        start.sectors["sector:energy"].output_bn, abs=0.01
    )
    assert states[-1].engine.pushes == []


def test_job_protection_covers_losses_already_in_flight() -> None:
    """A furlough started after a slump has begun now holds back the job losses still to come."""
    start = toy_world()
    hit = play(start, 1, scenario(Shock(node="country:uk", delta=-5)))[0]
    furlough = PolicyAction(kind="spend", target=JOB_NODE, magnitude=-1.0, duration_turns=6)
    bare = play_from(start, hit, [])
    held = play_from(start, hit, [furlough])
    assert value(held, UNEMPLOYMENT) < value(bare, UNEMPLOYMENT)


def play_from(start: WorldState, state: WorldState, actions: list[PolicyAction]) -> WorldState:
    for t in range(4):
        state = resolve(start, state, QUIET, actions if t == 0 else [], NOTHING, CONFIG)
    return state


def test_a_later_graph_change_reshapes_effects_in_flight() -> None:
    from hog_sim.core.models import EdgeKind, GraphChange

    start = toy_world()
    hit = play(start, 1, scenario(Shock(node="sector:energy", delta=-2, duration_turns=3)))[0]
    edge = next(
        e
        for e in hit.edges
        if e.source == "sector:energy" and e.target == "indicator:energy_prices"
    )
    cut = GraphChange(
        kind="edge_weight",
        source=edge.source,
        target=edge.target,
        edge_kind=EdgeKind.DRIVES,
        delta=-0.1 if edge.weight > 0 else 0.1,
        reason="x",
    )
    rewired = apply_graph_changes(hit, [cut])
    a = step(hit, [])[1]["indicator:energy_prices"]
    b = step(rewired, [])[1]["indicator:energy_prices"]
    assert a != b


def test_old_saves_still_drain_their_queue() -> None:
    start = toy_world()
    old = start.model_copy(update={"pending": {0: {INFLATION: 0.5}, 1: {INFLATION: -0.2}}})
    first, second = play(old, 2)
    assert value(first, INFLATION) == pytest.approx(value(start, INFLATION) + 0.5)
    assert value(second, INFLATION) == pytest.approx(value(start, INFLATION) + 0.3)
    assert second.pending == {}
    engine = propagate(old, [], horizon=2, k_draws=1)
    assert engine.at(1)[INFLATION] == pytest.approx(0.3)


def test_the_engine_state_survives_a_save() -> None:
    start = toy_world()
    state = play(start, 2, scenario(Shock(node="sector:energy", delta=-1, duration_turns=5)))[-1]
    assert state.engine.pushes and state.engine.own
    assert WorldState.from_json(state.to_json()) == state
