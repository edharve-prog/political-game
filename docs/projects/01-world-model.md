# Project 1 — World Model & State

**Status:** In progress (first pass in review)
**Depends on:** 0   **Provides:** `WorldState`, `snapshot()`/`diff()`, graph query helpers

## Goal
Represent the world as a typed graph with a clean state container.

## Scope
- In: node/edge schemas, WorldState validation, NetworkX projection, structural queries, a hand-built toy world.
- Out: any numbers changing over time (Project 3), approval maths (Project 4), real data (Project 2).

## Design
- `WorldState` (`core/state.py`) is the source of truth. Nodes live in one dict per kind, keyed by namespaced id (`sector:energy`). Validation checks keys match ids, ids carry the right prefix, the player country exists and every edge points at a real node.
- `world/graph.py` builds a `MultiDiGraph` projection keyed by `EdgeKind`, so two kinds can link the same pair. The graph is rebuilt from state; it is never edited directly.
- Indicators are nodes (decision from the plan), so `DRIVES` and `CARES_ABOUT` are explicit, inspectable edges.
- `CARES_ABOUT` points group → indicator with a signed weight (negative: a rise lowers approval).

## Interface
```python
build_graph(state) -> nx.MultiDiGraph
edges_of(graph, node_id, kinds=None, direction="out" | "in" | "both") -> list[Edge]
nodes_of_kind(graph, kind) -> list[str]
groups_employed_by(graph, sector_id) -> {group_id: weight}
drivers_of(graph, indicator_id) -> {node_id: weight}
groups_caring_about(graph, indicator_id) -> {group_id: weight}
downstream(graph, node_id, max_depth=None) -> set[str]
exposed_groups(graph, node_id, max_depth=None) -> set[str]
WorldState.node(id), .nodes(), .node_ids(), .snapshot(), .diff(other), .to_json(), .from_json()
```

## Toy world (`world/seed/toy.py`)
UK plus EU and China; Energy, Finance, Manufacturing, Housing & Construction, Public Sector; Pensioners, Young renters, Public sector workers, Business owners; Commons and Bank of England; six indicators (inflation, unemployment, Bank Rate, energy prices, house prices, deficit). Figures are illustrative, not sourced.

## Tasks
- [x] WorldState validation and node lookup
- [x] Graph projection and query helpers
- [x] Toy world
- [x] Tests: build, query, snapshot/diff across a manual change, validation failures

## Done when
Toy world builds, can be queried, snapshot/diff works across a manual change. (Met by `tests/test_world.py`.)

## Open questions
- Should `downstream` respect edge sign or lag (e.g. "what is affected within 3 turns")? Deferred to Project 3, where lags matter.
- Sector granularity for the MVP: the plan says 8 sectors; the toy has 5. Project 2 sets the real list.
