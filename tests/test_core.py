import pytest
from pydantic import ValidationError

from hog_sim.core.config import GameConfig, make_rng
from hog_sim.core.models import (
    Country,
    Edge,
    EdgeKind,
    Group,
    Indicator,
    NodeKind,
    PolicyAction,
    Sector,
)
from hog_sim.core.state import WorldState
from hog_sim.world.seed.toy import toy_world


@pytest.fixture
def world() -> WorldState:
    return WorldState(
        player_country="country:uk",
        countries={"country:uk": Country(id="country:uk", name="UK", gdp_bn=2700, growth_pct=0.8)},
        sectors={
            "sector:energy": Sector(
                id="sector:energy", name="Energy", output_bn=80, employment_k=170
            )
        },
        groups={
            "group:pensioners": Group(
                id="group:pensioners", name="Pensioners", population_share=0.19, turnout=0.8
            )
        },
        indicators={
            "indicator:inflation": Indicator(
                id="indicator:inflation", name="Inflation", value=3.8, unit="%"
            )
        },
        edges=[
            Edge(
                source="sector:energy",
                target="indicator:inflation",
                kind=EdgeKind.DRIVES,
                weight=0.3,
                lag=1,
                uncertainty=0.1,
            )
        ],
    )


def test_json_round_trip(world: WorldState) -> None:
    assert WorldState.from_json(world.to_json()) == world


def test_snapshot_is_independent(world: WorldState) -> None:
    snap = world.snapshot()
    world.indicators["indicator:inflation"].value = 5.0
    assert snap.indicators["indicator:inflation"].value == 3.8


def test_diff_reports_changes(world: WorldState) -> None:
    after = world.snapshot()
    after.turn = 1
    after.indicators["indicator:inflation"].value = 4.1
    assert world.diff(after) == {
        "indicator:inflation": {"value": (3.8, 4.1)},
        "world": {"turn": (0, 1)},
    }
    assert world.diff(world.snapshot()) == {}


def test_node_kind_from_id(world: WorldState) -> None:
    assert world.groups["group:pensioners"].kind is NodeKind.GROUP


def test_policy_action_rejects_out_of_range_magnitude() -> None:
    with pytest.raises(ValidationError):
        PolicyAction(kind="tax", target="sector:energy", magnitude=2.0)


def test_rng_is_deterministic() -> None:
    assert make_rng(1, 3, "news").random() == make_rng(1, 3, "news").random()
    assert make_rng(1, 3, "news").random() != make_rng(1, 4, "news").random()


@pytest.mark.parametrize(
    "field",
    [
        {"horizon": 0},
        {"k_draws": 0},
        {"election_turn": 0},
        {"turn_length_months": 0},
        {"capital_per_turn": -1.0},
        {"capital_per_turn": float("nan")},
    ],
)
def test_game_config_rejects_values_the_engine_cannot_run(field) -> None:
    """Finding 11: these used to fail deep in forecasting or flip action signs."""
    with pytest.raises(ValidationError):
        GameConfig(**field)


def _toy_json() -> dict:
    return toy_world().model_dump()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("indicators", "indicator:inflation", "low"), 40.0, "above high"),
        (("indicators", "indicator:inflation", "value"), 99.0, "above its ceiling"),
        (("indicators", "indicator:inflation", "value"), -9.0, "below its floor"),
        (("indicators", "indicator:interest_rate", "controlled_by"), "institution:mint", "not an"),
        (("sectors", "sector:energy", "output_bn"), -1.0, "greater than or equal"),
        (("countries", "country:uk", "growth_pct"), float("inf"), "finite"),
    ],
)
def test_world_state_rejects_broken_invariants(path, value, message) -> None:
    data = _toy_json()
    collection, node, field = path
    data[collection][node][field] = value
    with pytest.raises(ValidationError, match=message):
        WorldState.model_validate(data)


def test_world_state_needs_some_population() -> None:
    data = _toy_json()
    for group in data["groups"].values():
        group["population_share"] = 0.0
    with pytest.raises(ValidationError, match="population_share"):
        WorldState.model_validate(data)
