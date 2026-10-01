import pytest
from pydantic import ValidationError

from hog_sim.core.models import Edge, EdgeKind, NodeKind
from hog_sim.core.state import WorldState
from hog_sim.world.graph import (
    build_graph,
    downstream,
    drivers_of,
    edges_of,
    exposed_groups,
    groups_caring_about,
    groups_employed_by,
    nodes_of_kind,
)
from hog_sim.world.seed.toy import toy_world


@pytest.fixture
def world() -> WorldState:
    return toy_world()


@pytest.fixture
def graph(world: WorldState):
    return build_graph(world)


def test_toy_world_meets_plan_size(world: WorldState) -> None:
    assert len(world.countries) == 3  # player + 2 foreign
    assert len(world.sectors) == 5
    assert len(world.groups) == 4


def test_graph_has_every_node_and_edge(world: WorldState, graph) -> None:
    assert graph.number_of_nodes() == len(list(world.nodes()))
    assert graph.number_of_edges() == len(world.edges)
    assert nodes_of_kind(graph, NodeKind.SECTOR) == sorted(world.sectors)


def test_edges_of_filters_by_kind_and_direction(graph) -> None:
    out = edges_of(graph, "sector:energy", [EdgeKind.DRIVES])
    assert [e.target for e in out] == ["indicator:energy_prices"]
    incoming = edges_of(graph, "indicator:inflation", direction="in")
    assert {e.kind for e in incoming} == {EdgeKind.DRIVES, EdgeKind.CARES_ABOUT}


def test_query_helpers(graph) -> None:
    assert groups_employed_by(graph, "sector:public") == {"group:public_workers": 0.9}
    assert drivers_of(graph, "indicator:inflation") == {
        "indicator:energy_prices": 0.3,
        "indicator:interest_rate": -0.4,
    }
    assert set(groups_caring_about(graph, "indicator:house_prices")) == {
        "group:pensioners",
        "group:young_renters",
    }


def test_energy_shock_reaches_inflation_and_its_groups(graph) -> None:
    reached = downstream(graph, "sector:energy")
    assert {"indicator:energy_prices", "indicator:inflation"} <= reached
    assert downstream(graph, "sector:energy", max_depth=1) == {
        "indicator:energy_prices",
        "sector:manufacturing",
    }
    exposed = exposed_groups(graph, "sector:energy")
    assert {"group:pensioners", "group:public_workers"} <= exposed


def test_snapshot_diff_across_manual_change(world: WorldState) -> None:
    before = world.snapshot()
    world.indicators["indicator:energy_prices"].value = 115
    world.groups["group:pensioners"].approval = 0.42
    assert before.diff(world) == {
        "indicator:energy_prices": {"value": (100, 115)},
        "group:pensioners": {"approval": (0.55, 0.42)},
    }


def test_toy_world_round_trips(world: WorldState) -> None:
    assert WorldState.from_json(world.to_json()) == world


def test_rejects_edge_to_unknown_node(world: WorldState) -> None:
    data = world.model_dump()
    bad = Edge(source="sector:energy", target="indicator:nope", kind=EdgeKind.DRIVES, weight=1)
    data["edges"].append(bad.model_dump())
    with pytest.raises(ValidationError, match="unknown node"):
        WorldState.model_validate(data)


def test_rejects_wrong_prefix(world: WorldState) -> None:
    data = world.model_dump()
    data["sectors"]["group:energy"] = {**data["sectors"].pop("sector:energy"), "id": "group:energy"}
    data["edges"] = []
    with pytest.raises(ValidationError, match="wrong prefix"):
        WorldState.model_validate(data)


def test_node_lookup(world: WorldState) -> None:
    assert world.node("institution:central_bank").name == "Bank of England"
    with pytest.raises(KeyError):
        world.node("country:atlantis")
