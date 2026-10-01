"""Shock propagation through the world graph.

The engine is linear difference equations on the graph:

* Every node has one primary metric (see ``METRICS``) and a ``scale``: the size of one
  "standard step" in that metric's native units. Propagation runs in standard steps so
  edge weights are comparable: weight 0.5 means a one-step move at the source moves the
  target half a step.
* A shock is an impulse of ``delta`` standard steps at a node, repeated for
  ``duration_turns`` turns. Impulses are level shifts: they persist until offset.
* An impulse travels along each outgoing propagating edge, multiplied by the edge weight
  and by ``DAMPING`` per hop, and lands ``lag`` turns later. Lag-0 edges land the same turn.
* Population groups are not propagated here: approval is Project 4's job, so ``EMPLOYS``,
  ``CARES_ABOUT`` and any edge into a group are skipped.

Monte Carlo draws perturb each edge weight by its ``uncertainty`` (a normal std dev) once
per draw, seeded from the game seed so results reproduce exactly.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean, quantiles

from hog_sim.core.config import make_rng
from hog_sim.core.models import Edge, EdgeKind, Model, NodeKind, PolicyAction, Shock
from hog_sim.core.state import WorldState

DAMPING = 0.9
MIN_IMPULSE = 1e-4
MAX_HOPS_PER_TURN = 200

PROPAGATING = {EdgeKind.DRIVES, EdgeKind.SUPPLIES, EdgeKind.TRADES_WITH, EdgeKind.INFLUENCES}

# Primary metric per node kind. Groups are listed so deltas can be applied to them later.
METRICS = {
    NodeKind.COUNTRY: "growth_pct",
    NodeKind.SECTOR: "output_bn",
    NodeKind.GROUP: "approval",
    NodeKind.INSTITUTION: "support",
    NodeKind.INDICATOR: "value",
}

# Native size of one standard step for each indicator unit.
INDICATOR_UNIT_SCALE = {"%": 1.0, "% GDP": 1.0, "index": 10.0}


def scale(state: WorldState, node_id: str) -> float:
    """Native units per standard step for a node's primary metric."""
    node = state.node(node_id)
    match node.kind:
        case NodeKind.SECTOR:
            return 0.05 * node.output_bn  # 5% of output
        case NodeKind.COUNTRY:
            return 1.0  # 1pp of growth
        case NodeKind.GROUP:
            return 0.05
        case NodeKind.INSTITUTION:
            return 0.1
        case _:
            return INDICATOR_UNIT_SCALE.get(node.unit, 1.0)


class NodeForecast(Model):
    """Cumulative change in the node's primary metric, native units, one entry per turn."""

    mean: list[float]
    p10: list[float]
    p90: list[float]


class DeltaDistribution(Model):
    horizon: int
    k_draws: int
    nodes: dict[str, NodeForecast]

    def at(self, turn: int) -> dict[str, float]:
        """Expected cumulative native delta for every node at ``turn`` (0-based)."""
        return {n: f.mean[turn] for n, f in self.nodes.items()}


# Sign of the first-order effect of each action kind on its target's primary metric.
_ACTION_SIGN = {
    "tax": -1.0,
    "spend": 1.0,
    "regulate": -0.5,
    "deregulate": 0.5,
    "diplomatic": 1.0,
    "military": 1.0,
    "communicate": 0.2,
    "legislate": 1.0,
    "appoint": 0.5,
    "do_nothing": 0.0,
}
ACTION_STEPS = 2.0  # magnitude 1.0 means a two-step shock


# Spending and tax also move the budget: magnitude 1.0 of spending (or a tax cut) adds this
# many standard steps to the deficit, and the same size of tax rise takes it off.
FISCAL_NODE = "indicator:deficit"
_FISCAL_SIGN = {"spend": 1.0, "tax": -1.0}
FISCAL_STEPS = 0.6


def actions_to_shocks(actions: list[PolicyAction], state: WorldState | None = None) -> list[Shock]:
    """First-pass mapping from interpreted actions to engine shocks.

    The interpreter (Project 5) may later emit shocks directly; this keeps the loop runnable.
    Shocks are spread evenly over the action's duration. When ``state`` is given and has a
    deficit indicator, spending and tax actions also shock the deficit, so nothing is free.
    """
    fiscal = state is not None and FISCAL_NODE in state.indicators
    shocks = []
    for action in actions:
        total = _ACTION_SIGN[action.kind] * action.magnitude * ACTION_STEPS
        if total:
            shocks.append(
                Shock(
                    node=action.target,
                    delta=total / action.duration_turns,
                    duration_turns=action.duration_turns,
                )
            )
        cost = _FISCAL_SIGN.get(action.kind, 0.0) * action.magnitude * FISCAL_STEPS
        if fiscal and cost and action.target != FISCAL_NODE:
            shocks.append(
                Shock(
                    node=FISCAL_NODE,
                    delta=cost / action.duration_turns,
                    duration_turns=action.duration_turns,
                )
            )
    return shocks


def _propagating_edges(state: WorldState) -> list[Edge]:
    return [e for e in state.edges if e.kind in PROPAGATING and not e.target.startswith("group:")]


def simulate(
    state: WorldState,
    shocks: list[Shock],
    horizon: int,
    weights: dict[int, float] | None = None,
) -> dict[str, list[float]]:
    """One deterministic run. Returns cumulative standard-step deltas per node per turn.

    ``weights`` overrides edge weights by index into the propagating edge list (used by
    Monte Carlo draws).
    """
    edges = _propagating_edges(state)
    out: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for i, e in enumerate(edges):
        w = weights[i] if weights is not None else e.weight
        out[e.source].append((w * DAMPING, i))

    arrivals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for shock in shocks:
        state.node(shock.node)  # raises KeyError on an unknown node
        for t in range(shock.start_turn, shock.start_turn + shock.duration_turns):
            if t < horizon:
                arrivals[t][shock.node] += shock.delta

    level = {node_id: 0.0 for node_id in state.node_ids()}
    trajectory: dict[str, list[float]] = {node_id: [] for node_id in level}
    for t in range(horizon):
        work = list(arrivals.pop(t, {}).items())
        hops = 0
        while work and hops < MAX_HOPS_PER_TURN:
            hops += 1
            node_id, impulse = work.pop()
            level[node_id] += impulse
            for factor, i in out[node_id]:
                passed = impulse * factor
                if abs(passed) < MIN_IMPULSE:
                    continue
                lag = edges[i].lag
                if lag == 0:
                    work.append((edges[i].target, passed))
                elif t + lag < horizon:
                    arrivals[t + lag][edges[i].target] += passed
        for node_id, value in level.items():
            trajectory[node_id].append(value)
    return trajectory


def propagate(
    state: WorldState,
    shocks: list[Shock],
    horizon: int = 12,
    k_draws: int = 200,
    seed: int = 0,
) -> DeltaDistribution:
    """Monte Carlo propagation. Returns mean and 10th/90th percentiles in native units."""
    edges = _propagating_edges(state)
    draws: list[dict[str, list[float]]] = []
    for k in range(k_draws):
        rng = make_rng(seed, state.turn, f"mc:{k}")
        weights = {i: rng.gauss(e.weight, e.uncertainty) for i, e in enumerate(edges)}
        draws.append(simulate(state, shocks, horizon, weights))

    scales = {node_id: scale(state, node_id) for node_id in state.node_ids()}
    nodes = {}
    for node_id, s in scales.items():
        mean, p10, p90 = [], [], []
        for t in range(horizon):
            values = [d[node_id][t] * s for d in draws]
            mean.append(fmean(values))
            if len(values) > 1:
                cuts = quantiles(values, n=10, method="inclusive")
                p10.append(cuts[0])
                p90.append(cuts[-1])
            else:
                p10.append(values[0])
                p90.append(values[0])
        nodes[node_id] = NodeForecast(mean=mean, p10=p10, p90=p90)
    return DeltaDistribution(horizon=horizon, k_draws=k_draws, nodes=nodes)


def apply_deltas(state: WorldState, deltas: dict[str, float]) -> WorldState:
    """Return a new state with native-unit deltas added to each node's primary metric."""
    new = state.snapshot()
    for node_id, delta in deltas.items():
        if not delta:
            continue
        node = new.node(node_id)
        field = METRICS[node.kind]
        value = getattr(node, field) + delta
        if field in ("approval", "support"):
            value = min(1.0, max(0.0, value))
        setattr(node, field, value)
    return new
