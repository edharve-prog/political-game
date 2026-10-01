"""Build a NetworkX view of a WorldState and answer structural questions about it.

The graph is a read-only projection: the WorldState stays the source of truth, and the
graph is rebuilt whenever the state changes shape. Node attributes are the node's fields;
edges are keyed by their EdgeKind so several kinds can link the same pair of nodes.
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from hog_sim.core.models import Edge, EdgeKind, NodeKind
from hog_sim.core.state import WorldState


def build_graph(state: WorldState) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for node in state.nodes():
        graph.add_node(node.id, kind=node.kind, **node.model_dump())
    for edge in state.edges:
        graph.add_edge(edge.source, edge.target, key=edge.kind, **edge.model_dump())
    return graph


def edges_of(
    graph: nx.MultiDiGraph,
    node_id: str,
    kinds: Iterable[EdgeKind] | None = None,
    direction: str = "out",
) -> list[Edge]:
    """Edges touching ``node_id``, optionally filtered by kind. ``direction`` is out, in or both."""
    wanted = set(kinds) if kinds is not None else None
    found: list[tuple[str, str, dict]] = []
    if direction in ("out", "both"):
        found += graph.out_edges(node_id, data=True)
    if direction in ("in", "both"):
        found += graph.in_edges(node_id, data=True)
    edges = [Edge(**data) for _, _, data in found]
    return [e for e in edges if wanted is None or e.kind in wanted]


def nodes_of_kind(graph: nx.MultiDiGraph, kind: NodeKind) -> list[str]:
    return sorted(n for n, data in graph.nodes(data=True) if data["kind"] == kind)


def groups_employed_by(graph: nx.MultiDiGraph, sector_id: str) -> dict[str, float]:
    """Groups employed by a sector, with the employment weight."""
    return {e.target: e.weight for e in edges_of(graph, sector_id, [EdgeKind.EMPLOYS])}


def drivers_of(graph: nx.MultiDiGraph, indicator_id: str) -> dict[str, float]:
    """Nodes that directly drive an indicator, with the driving weight."""
    return {e.source: e.weight for e in edges_of(graph, indicator_id, [EdgeKind.DRIVES], "in")}


def groups_caring_about(graph: nx.MultiDiGraph, indicator_id: str) -> dict[str, float]:
    """Groups whose approval depends on an indicator, with the weight they put on it."""
    return {e.source: e.weight for e in edges_of(graph, indicator_id, [EdgeKind.CARES_ABOUT], "in")}


def downstream(graph: nx.MultiDiGraph, node_id: str, max_depth: int | None = None) -> set[str]:
    """Every node a change at ``node_id`` can reach by following edges forwards."""
    lengths = nx.single_source_shortest_path_length(graph, node_id, cutoff=max_depth)
    return set(lengths) - {node_id}


def exposed_groups(graph: nx.MultiDiGraph, node_id: str, max_depth: int | None = None) -> set[str]:
    """Groups a change at ``node_id`` can reach, directly or through indicators and sectors.

    Groups influence approval through CARES_ABOUT edges that point *from* the group to an
    indicator, so a group is exposed when it employs-in or cares about anything downstream.
    """
    reached = downstream(graph, node_id, max_depth) | {node_id}
    exposed = {n for n in reached if graph.nodes[n]["kind"] == NodeKind.GROUP}
    for indicator in (n for n in reached if graph.nodes[n]["kind"] == NodeKind.INDICATOR):
        exposed |= set(groups_caring_about(graph, indicator))
    return exposed - {node_id}
