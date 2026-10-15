"""Output as a level (O3, M1, Ed's review of 13 October 2026): sectors add up to an output gap,
growth is its change over the year and unemployment follows it, so holding a policy for two
years gives a step in output for two years rather than two years of faster growth, and no
borrowing pays for itself."""

from functools import cache

import pytest

from hog_sim.core.config import GameConfig
from hog_sim.core.models import EdgeKind, GraphChange, Outcome, PolicyAction, Scenario, Shock
from hog_sim.core.state import WorldState
from hog_sim.game.loop import resolve
from hog_sim.world.changes import check_graph_change
from hog_sim.world.propagation import (
    GAP_NODE,
    NATIONAL_MULTIPLIER,
    actions_to_shocks,
    fiscal_cost,
    national_demand,
    pushes_for,
    simulate,
)
from hog_sim.world.seed.toy import toy_world

UK, JOBS = "country:uk", "indicator:unemployment"
NOTHING = Outcome(narrative="Nothing else happens", probability=1)
QUIET = Scenario(title="Quiet", briefing="", affected_nodes=[], urgency=0, shocks=[])
TURNS = 36


def held(delta: float, turns: int) -> Shock:
    return Shock(node=GAP_NODE, delta=delta, duration_turns=turns, hold=True)


def test_growth_is_the_change_in_output_over_the_year() -> None:
    path = simulate(toy_world(), [held(1.0, 40)], 30)
    gap, growth = path[GAP_NODE], path[UK]
    assert gap[5] > 0.95 and gap[25] > 0.95  # output stays up while the policy runs
    for t in range(30):
        year_ago = gap[t - 12] if t >= 12 else 0.0
        assert growth[t] == pytest.approx(gap[t] - year_ago, abs=1e-9)
    assert growth[5] > 0.95 and abs(growth[25]) < 0.01  # faster growth for a year, then trend


def test_a_programme_that_ends_leaves_no_lasting_output() -> None:
    path = simulate(toy_world(), [held(1.0, 12)], 60)
    assert path[GAP_NODE][6] > 0.9
    assert abs(path[GAP_NODE][-1]) < 0.01
    assert sum(path[UK]) == pytest.approx(sum(path[GAP_NODE][-12:]), abs=1e-9)


def test_unemployment_follows_the_gap_not_growth() -> None:
    path = simulate(toy_world(), [held(-1.0, 40)], 30)
    assert abs(path[UK][25]) < 0.01  # growth back at trend
    assert path[JOBS][25] > 0.3  # but output is still below potential, so jobs are too


def test_a_shock_to_the_players_growth_is_a_fall_in_output() -> None:
    world = toy_world()
    (push,) = pushes_for([Shock(node=UK, delta=-5)], world)
    assert push.node == GAP_NODE and push.delta == -5
    (abroad,) = pushes_for([Shock(node="country:eu", delta=-1)], world)
    assert abroad.node == "country:eu"


@pytest.mark.parametrize("kind", ["spend", "tax"])
def test_national_tax_and_spending_move_demand_by_less_than_they_cost(kind) -> None:
    world = toy_world()
    action = PolicyAction(kind=kind, target=UK, magnitude=-1.0 if kind == "tax" else 1.0)
    (demand,) = national_demand(action, world)
    cost = fiscal_cost(action, world)
    assert demand.node == GAP_NODE and demand.hold
    assert demand.delta == pytest.approx(cost * NATIONAL_MULTIPLIER[kind]) and demand.delta > 0
    assert NATIONAL_MULTIPLIER[kind] < 1
    rise = action.model_copy(update={"magnitude": 1.0 if kind == "tax" else -1.0})
    assert national_demand(rise, world)[0].delta < 0
    assert demand in actions_to_shocks([action], world)


def _play(actions: list[PolicyAction]) -> list[WorldState]:
    start = state = toy_world()
    states = []
    for t in range(TURNS):
        state = resolve(start, state, QUIET, actions if t == 0 else [], NOTHING, GameConfig())
        states.append(state)
    return states


@cache
def _debt(target: str | None = None, kind: str = "", magnitude: float = 0.0) -> list[float]:
    """Debt, % of GDP, each turn after a year of ``kind`` on ``target`` (None: nothing)."""
    actions = []
    if target is not None:
        actions = [PolicyAction(kind=kind, target=target, magnitude=magnitude, duration_turns=12)]
    return [s.countries[UK].debt_pct for s in _play(actions)]


_world = toy_world()
TARGETS = [*_world.sectors, *_world.groups, UK]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(("kind", "magnitude"), [("spend", 1.0), ("tax", -1.0)])
def test_extra_borrowing_never_lowers_debt_to_gdp(target, kind, magnitude) -> None:
    """Coordinator's property test for O3: a year of spending or tax cuts, on anything, leaves
    debt as a share of GDP above doing nothing on every turn, while it runs and after."""
    borrowing, idle = _debt(target, kind, magnitude), _debt()
    assert all(b > i for b, i in zip(borrowing, idle, strict=True))


def test_the_output_accounting_cannot_be_rewired() -> None:
    world = toy_world()
    for source, target, kind in [
        ("sector:public", GAP_NODE, EdgeKind.SUPPLIES),
        (GAP_NODE, UK, EdgeKind.SUPPLIES),
        (GAP_NODE, UK, EdgeKind.DRIVES),
    ]:
        change = GraphChange(
            kind="edge_weight", source=source, target=target, edge_kind=kind, delta=0.05, reason="r"
        )
        assert any("fixed" in p for p in check_graph_change(world, change))


def test_links_with_long_lags_fire() -> None:
    """The engine kept only half the longest lag of history, so 12-turn links never fired."""
    path = simulate(toy_world(), [Shock(node="sector:housing", delta=1.0)], 14)
    assert path["indicator:house_prices"][12] != pytest.approx(path["indicator:house_prices"][11])
