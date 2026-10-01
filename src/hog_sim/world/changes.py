"""Changes to the world graph proposed by the LLM, checked and applied by the engine.

An outcome may say that the world itself has shifted: a trade row leaves relations with
China worse, a bailout ties finance more tightly to the public sector. Those arrive as
``GraphChange`` records. Nothing here trusts them: ``validate_graph_changes`` checks every
id, field and size against the current state, and ``apply_graph_changes`` (called from
``resolve()``) applies only what passes, clamped to each field's bounds. Because the changes
live in the logged outcome, replay reproduces them exactly.
"""

from __future__ import annotations

from hog_sim.core.models import Edge, EdgeKind, GraphChange, NodeKind
from hog_sim.core.state import WorldState

MAX_CHANGES = 3
MAX_ATTR_DELTA = 0.2
MAX_WEIGHT_STEP = 0.1  # an edge weight moves by at most this, or 25% of its size if larger
MAX_WEIGHT_STEP_FRACTION = 0.25
MAX_WEIGHT = 1.0
MAX_NEW_EDGE_WEIGHT = 0.3
MAX_NEW_EDGE_LAG = 6

# Node fields an outcome may move, with their bounds.
ATTRS: dict[NodeKind, dict[str, tuple[float, float]]] = {
    NodeKind.COUNTRY: {"relationship": (-1.0, 1.0), "stability": (0.0, 1.0)},
    NodeKind.INSTITUTION: {"support": (0.0, 1.0), "independence": (0.0, 1.0)},
    NodeKind.SECTOR: {"sentiment": (-1.0, 1.0)},
}

# Edge kinds an outcome may create. CARES_ABOUT and EMPLOYS describe who people are, which a
# single month's events do not change.
NEW_EDGE_KINDS = {
    EdgeKind.TRADES_WITH,
    EdgeKind.ALLIED_WITH,
    EdgeKind.RIVAL_OF,
    EdgeKind.SUPPLIES,
    EdgeKind.DRIVES,
    EdgeKind.INFLUENCES,
}


def _find_edge(state: WorldState, change: GraphChange) -> int | None:
    for i, edge in enumerate(state.edges):
        if (edge.source, edge.target, edge.kind) == (
            change.source,
            change.target,
            change.edge_kind,
        ):
            return i
    return None


def max_weight_step(weight: float) -> float:
    return max(MAX_WEIGHT_STEP, MAX_WEIGHT_STEP_FRACTION * abs(weight))


def check_graph_change(state: WorldState, change: GraphChange) -> list[str]:
    """Problems with one change against ``state``; empty when it may be applied."""
    ids = set(state.node_ids())
    if change.kind == "node_attr":
        if change.node not in ids:
            return [f"unknown node {change.node!r}"]
        kind = state.node(change.node).kind
        allowed = ATTRS.get(kind, {})
        if change.attr not in allowed:
            names = ", ".join(sorted(allowed)) or "none"
            return [f"{change.node}: attr {change.attr!r} cannot change (allowed: {names})"]
        if abs(change.delta) > MAX_ATTR_DELTA:
            return [f"{change.node}.{change.attr}: |delta| must be <= {MAX_ATTR_DELTA}"]
        return []

    problems = [f"unknown node {n!r}" for n in (change.source, change.target) if n not in ids]
    if change.edge_kind is None:
        problems.append("edge changes need edge_kind")
    if problems:
        return problems
    if change.source == change.target:
        return ["an edge cannot link a node to itself"]
    existing = _find_edge(state, change)
    label = f"{change.source} -{change.edge_kind}-> {change.target}"
    if change.kind == "edge_weight":
        if existing is None:
            return [f"no edge {label}; use add_edge to create one"]
        weight = state.edges[existing].weight
        if abs(change.delta) > max_weight_step(weight):
            return [f"{label}: weight may move by at most {max_weight_step(weight):.2f}"]
        return []
    # add_edge
    if existing is not None:
        return [f"edge {label} already exists; use edge_weight"]
    if change.edge_kind not in NEW_EDGE_KINDS:
        return [f"{change.edge_kind} edges cannot be created by events"]
    if not 0 < abs(change.delta) <= MAX_NEW_EDGE_WEIGHT:
        return [f"{label}: a new edge's weight must be non-zero and within ±{MAX_NEW_EDGE_WEIGHT}"]
    if change.lag > MAX_NEW_EDGE_LAG:
        return [f"{label}: lag must be <= {MAX_NEW_EDGE_LAG}"]
    if change.target.startswith("group:"):
        return [f"{label}: edges into groups are not created by events"]
    return []


def validate_graph_changes(state: WorldState, changes: list[GraphChange]) -> list[str]:
    """Problems with a set of changes, prefixed by index; empty when all may be applied."""
    problems = []
    if len(changes) > MAX_CHANGES:
        problems.append(f"at most {MAX_CHANGES} graph changes per outcome")
    seen = set()
    for i, change in enumerate(changes):
        target = (
            change.kind == "node_attr",
            change.node,
            change.attr,
            change.source,
            change.target,
            change.edge_kind,
        )
        if target in seen:
            problems.append(f"graph_changes[{i}]: duplicates an earlier change")
        seen.add(target)
        problems += [f"graph_changes[{i}]: {p}" for p in check_graph_change(state, change)]
    return problems


def apply_graph_changes(state: WorldState, changes: list[GraphChange]) -> WorldState:
    """Return a new state with every valid change applied; invalid ones are skipped.

    Skipping (rather than raising) keeps an old save resolvable if the rules tighten later.
    """
    if not changes:
        return state
    new = state.snapshot()
    for change in changes[:MAX_CHANGES]:
        if check_graph_change(new, change):
            continue
        if change.kind == "node_attr":
            node = new.node(change.node)
            low, high = ATTRS[node.kind][change.attr]
            setattr(
                node, change.attr, min(high, max(low, getattr(node, change.attr) + change.delta))
            )
        elif change.kind == "edge_weight":
            edge = new.edges[_find_edge(new, change)]
            edge.weight = min(MAX_WEIGHT, max(-MAX_WEIGHT, edge.weight + change.delta))
        else:
            new.edges.append(
                Edge(
                    source=change.source,
                    target=change.target,
                    kind=change.edge_kind,
                    weight=change.delta,
                    lag=change.lag,
                    uncertainty=abs(change.delta) / 2,
                )
            )
    return new
