"""Real turns and forecasts share one engine step, and the engine carries its own state
between turns (Ed's review at 7be7c65: caps, forecasts, frozen and truncated effects)."""

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction, Scenario, Shock
from hog_sim.core.state import WorldState
from hog_sim.game.loop import forecast, resolve
from hog_sim.world.changes import apply_graph_changes
from hog_sim.world.propagation import JOB_NODE, METRICS, propagate, step
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


# Review of 361a0ff, findings 2 and 5: the no-action baseline already held the new subsidy's
# job protection, and the forecast left out what resolve does after the engine step.

LEGISLATURE = "institution:legislature"


def test_the_baseline_leaves_out_a_new_subsidy() -> None:
    """After a slump, a wage subsidy's forecast effect on unemployment was reported as 0.000,
    though running with and without it differed by about half a point."""
    start = certain(toy_world())
    config = CONFIG.model_copy(update={"horizon": 6, "k_draws": 1})
    slump = scenario(Shock(node="sector:manufacturing", delta=-2))
    state = resolve(start, start, slump, [], NOTHING, config)
    subsidy = PolicyAction(kind="spend", target=UNEMPLOYMENT, magnitude=1, duration_turns=6)
    with_it = forecast(start, state, QUIET, [subsidy], config)
    without = forecast(start, state, QUIET, [], config)
    separate = with_it.at(2)[UNEMPLOYMENT] - without.at(2)[UNEMPLOYMENT]
    assert separate < -0.3
    assert with_it.response_at(2)[UNEMPLOYMENT] == pytest.approx(separate, abs=1e-9)


def test_with_no_decisions_the_response_is_nothing_on_every_draw() -> None:
    """The baseline runs on each draw's own edge weights, so uncertainty alone cannot make a
    response out of nothing, even where bounds make the engine nonlinear."""
    start = toy_world()
    start.indicators[UNEMPLOYMENT].value = 24  # near its cap, so the bound bites
    crisis = scenario(Shock(node=UNEMPLOYMENT, delta=3), Shock(node="sector:energy", delta=-2))
    engine = forecast(start, start, crisis, [], CONFIG.model_copy(update={"k_draws": 20}))
    for t in range(engine.horizon):
        assert all(abs(v) < 1e-9 for v in engine.response_at(t).values()), t


def test_the_scenario_is_part_of_the_baseline_not_the_response() -> None:
    start = certain(toy_world())
    crisis = scenario(Shock(node=INFLATION, delta=2))
    engine = forecast(start, start, crisis, [], CONFIG.model_copy(update={"k_draws": 1}))
    assert engine.at(0)[INFLATION] == pytest.approx(2.0, rel=0.05)
    assert engine.response_at(0)[INFLATION] == pytest.approx(0.0, abs=1e-9)


def test_a_whip_is_forecast_as_resolve_settles_it() -> None:
    """A full whip was forecast at +0.10 Commons support; resolve settled it to +0.08."""
    start = certain(toy_world())
    whip = PolicyAction(kind="communicate", target=LEGISLATURE, magnitude=1)
    config = CONFIG.model_copy(update={"k_draws": 1})
    engine = forecast(start, start, QUIET, [whip], config)
    after = resolve(start, start, QUIET, [whip], NOTHING, config)
    moved = after.institutions[LEGISLATURE].support - start.institutions[LEGISLATURE].support
    assert engine.at(0)[LEGISLATURE] == pytest.approx(moved, abs=1e-12)


def test_the_forecast_matches_resolve_over_several_turns(monkeypatch) -> None:
    """Whipping, a trade deal (which rescales trade from the next turn), a sacking and the debt,
    with no outcome and the central weights. The Commons' drift with the polls is held off:
    the forecast does not project approval, and says so."""
    import hog_sim.game.loop as loop
    import hog_sim.world.opposition as opposition

    monkeypatch.setattr(loop, "COMMONS_POLLS", 0.0)
    # By-elections turn on approval too, which the forecast does not project.
    monkeypatch.setattr(opposition, "BYELECTION_CHANCE", 0.0)
    start = certain(toy_world())
    state = play(start, 2, scenario(Shock(node="indicator:interest_rate", delta=1)))[-1]
    actions = [
        PolicyAction(kind="communicate", target=LEGISLATURE, magnitude=0.5),
        PolicyAction(kind="diplomatic", target="country:eu", magnitude=1),
        PolicyAction(kind="spend", target="sector:public", magnitude=0.4, duration_turns=3),
    ]
    sacked = ["person:chancellor"]
    turns = 6
    config = CONFIG.model_copy(update={"horizon": turns, "k_draws": 1})
    engine = forecast(start, state, QUIET, actions, config, sacked=sacked)
    assert any("approval" in note for note in engine.unprojected)
    kept = forecast(start, state, QUIET, actions, config)
    assert engine.at(0)[LEGISLATURE] < kept.at(0)[LEGISLATURE] - 0.03  # the sacking costs
    now = state
    for t in range(turns):
        first = t == 0
        now = resolve(
            start,
            now,
            QUIET,
            actions if first else [],
            NOTHING,
            config,
            sacked=sacked if first else (),
        )
        for node in [*state.indicators, *state.sectors, *state.countries, *state.institutions]:
            before = getattr(state.node(node), METRICS[state.node(node).kind])
            after = getattr(now.node(node), METRICS[now.node(node).kind])
            assert after == pytest.approx(before + engine.at(t)[node], abs=1e-9), (node, t)
