"""Changes to the world graph proposed by the LLM, checked and applied by the engine.

An outcome may say that the world itself has shifted: a trade row leaves relations with
China worse, a bailout ties finance more tightly to the public sector. Those arrive as
``GraphChange`` records. Nothing here trusts them: ``validate_graph_changes`` checks every
id, field and size against the current state, and ``apply_graph_changes`` (called from
``resolve()``) applies only what passes, clamped to each field's bounds. Because the changes
live in the logged outcome, replay reproduces them exactly.
"""

from __future__ import annotations

from hog_sim.core.models import Edge, EdgeKind, GraphChange, NodeKind, PolicyAction
from hog_sim.core.state import WorldState
from hog_sim.world.propagation import MAX_LAG0_GAIN, PROPAGATING, lag0_gain

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

# Edge kinds that describe who people are: what a group cares about and where it works. A
# single month's events do not change them, so outcomes may neither create nor reweight them
# (EB-10). INFLUENCES edges into a group are protected for the same reason.
IDENTITY_KINDS = {EdgeKind.CARES_ABOUT, EdgeKind.EMPLOYS}

# Edge kinds an outcome may create.
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
        if change.edge_kind in IDENTITY_KINDS or change.target.startswith("group:"):
            return [f"{label}: edges that define a population group cannot change"]
        weight = state.edges[existing].weight
        if abs(change.delta) > max_weight_step(weight):
            return [f"{label}: weight may move by at most {max_weight_step(weight):.2f}"]
        return _stability_problem(state, change, label)
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
    return _stability_problem(state, change, label)


def _changed_edges(state: WorldState, change: GraphChange) -> list[Edge]:
    """The state's edges as they would be after an edge change (ignoring the weight clamp)."""
    edges = [e.model_copy() for e in state.edges]
    existing = _find_edge(state, change)
    if existing is not None:
        edge = edges[existing]
        edge.weight = min(MAX_WEIGHT, max(-MAX_WEIGHT, edge.weight + change.delta))
    else:
        edges.append(_new_edge(change))
    return edges


def _stability_problem(state: WorldState, change: GraphChange, label: str) -> list[str]:
    """An edge change must not make the within-turn feedback loops unstable (EC-6)."""
    if change.lag and _find_edge(state, change) is None:
        return []  # a new lagged edge adds no within-turn loop
    after = lag0_gain(state, _propagating(_changed_edges(state, change)))
    if after >= MAX_LAG0_GAIN and after > lag0_gain(state):
        return [f"{label}: would make same-turn feedback unstable (loop gain {after:.2f})"]
    return []


def _propagating(edges: list[Edge]) -> list[Edge]:
    return [e for e in edges if e.kind in PROPAGATING and not e.target.startswith("group:")]


def _new_edge(change: GraphChange) -> Edge:
    return Edge(
        source=change.source,
        target=change.target,
        kind=change.edge_kind,
        weight=change.delta,
        lag=change.lag,
        uncertainty=abs(change.delta) / 2,
    )


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


# What the player's own foreign policy does to the target country's standing, per unit of
# magnitude: diplomacy moves the relationship its way; military escalation sours it and
# unsettles the country, and de-escalation (magnitude < 0) mends it.
DIPLOMATIC_RELATIONSHIP = 0.1
MILITARY_RELATIONSHIP = -0.15
MILITARY_STABILITY = -0.05


def action_changes(state: WorldState, actions: list[PolicyAction]) -> list[GraphChange]:
    """Node changes implied by diplomatic and military actions towards foreign countries."""
    changes = []
    for a in actions:
        if a.target not in state.countries or a.target == state.player_country:
            continue
        if a.kind == "diplomatic" and a.magnitude:
            changes.append(_attr(a.target, "relationship", DIPLOMATIC_RELATIONSHIP * a.magnitude))
        elif a.kind == "military" and a.magnitude:
            changes.append(_attr(a.target, "relationship", MILITARY_RELATIONSHIP * a.magnitude))
            if a.magnitude > 0:
                changes.append(_attr(a.target, "stability", MILITARY_STABILITY * a.magnitude))
    return changes


def _attr(node: str, attr: str, delta: float) -> GraphChange:
    return GraphChange(kind="node_attr", node=node, attr=attr, delta=delta, reason="player action")


def apply_graph_changes(
    state: WorldState, changes: list[GraphChange], *, trusted: bool = False
) -> WorldState:
    """Return a new state with every valid change applied; invalid ones are skipped.

    Skipping (rather than raising) keeps an old save resolvable if the rules tighten later.
    Outcome (LLM) changes are capped at ``MAX_CHANGES``. ``trusted`` changes come from the
    engine itself (``action_changes``), so every one is applied, whatever their number.
    """
    if not changes:
        return state
    new = state.snapshot()
    for change in changes if trusted else changes[:MAX_CHANGES]:
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
            new.edges.append(_new_edge(change))
    return new
