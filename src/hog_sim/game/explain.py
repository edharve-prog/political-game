"""Why did that happen? Explanations of a played turn (backlog story TT-2).

``why`` traces the turn's biggest indicator moves back to what caused them: each of the
player's actions (and the scenario's and outcome's shocks) is re-run through the engine on
its own, the largest contributor is named, and the graph path it travelled is shown. It
then gives the reasons behind the biggest approval moves: the approval events in force and
the indicators each group cares about. ``alternatives`` lists the outcomes that were not
chosen, with their scores.

Everything here reads the turn record and the states either side of it; nothing changes.
"""

from __future__ import annotations

import networkx as nx

from hog_sim.core.models import EdgeKind, Shock
from hog_sim.core.state import WorldState
from hog_sim.game.records import TurnRecord
from hog_sim.population.popularity import EVENT_FLOOR, _event_weight
from hog_sim.world.events import event_shocks
from hog_sim.world.propagation import PROPAGATING, actions_to_shocks, scale, simulate

TOP = 3
MIN_MOVE = 1e-3


def _name(state: WorldState, node_id: str) -> str:
    try:
        return state.node(node_id).name
    except KeyError:
        return node_id


def _causes(record: TurnRecord, before: WorldState) -> list[tuple[str, list[Shock]]]:
    causes = [
        (f"your {a.kind} on {_name(before, a.target)} (size {a.magnitude:+.2f})", [a])
        for a in record.actions
        if a.kind != "do_nothing"
    ]
    shocks = [(label, actions_to_shocks(acts, before)) for label, acts in causes]
    if record.scenario.shocks:
        shocks.append(("the scenario itself", list(record.scenario.shocks)))
    outcome = list(record.outcome.shocks) + event_shocks(record.outcome.events, before)
    if outcome:
        shocks.append(("what happened next (the chosen outcome)", outcome))
    return shocks


def _contribution(before: WorldState, shocks: list[Shock], node_id: str) -> float:
    trajectory = simulate(before, shocks, 1)
    return trajectory.get(node_id, [0.0])[0] * scale(before, node_id)


def _path(graph: nx.DiGraph, sources: list[str], target: str, before: WorldState) -> str:
    best = None
    for source in sources:
        if source == target or source not in graph or target not in graph:
            continue
        try:
            path = nx.shortest_path(graph, source, target)
        except nx.NetworkXNoPath:
            continue
        if best is None or len(path) < len(best):
            best = path
    if best is None:
        return ""
    return " → ".join(_name(before, n) for n in best)


def _propagation_graph(state: WorldState) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(state.node_ids())
    graph.add_edges_from((e.source, e.target) for e in state.edges if e.kind in PROPAGATING)
    return graph


def why(record: TurnRecord, before: WorldState) -> list[str]:
    """Lines explaining the turn's biggest indicator and approval moves."""
    after = record.state_after
    graph = _propagation_graph(before)
    causes = _causes(record, before)
    lines = ["Biggest moves in the indicators this turn:"]
    moves = sorted(
        (
            (after.indicators[i].value - ind.value, i)
            for i, ind in before.indicators.items()
            if i in after.indicators
        ),
        key=lambda m: -abs(m[0]),
    )
    shown = 0
    for change, ind_id in moves:
        if abs(change) < MIN_MOVE or shown == TOP:
            break
        shown += 1
        unit = before.indicators[ind_id].unit
        line = f"  {_name(before, ind_id)} {change:+.2f} {unit}".rstrip()
        parts = [(label, shocks, _contribution(before, shocks, ind_id)) for label, shocks in causes]
        parts = [p for p in parts if abs(p[2]) >= MIN_MOVE]
        if parts:
            label, shocks, size = max(parts, key=lambda p: abs(p[2]))
            nodes = [s.node for s in shocks]
            path = _path(graph, nodes, ind_id, before)
            via = " directly" if ind_id in nodes else f" via {path}" if path else ""
            line += f": mostly {label}, {size:+.2f} of it{via}"
        else:
            line += ": the tail of earlier turns' policies and shocks"
        lines.append(line)
    if shown == 0:
        lines.append("  none of note")

    lines.append("Biggest moves in approval:")
    groups = sorted(
        (
            (after.groups[g].approval - grp.approval, g)
            for g, grp in before.groups.items()
            if g in after.groups
        ),
        key=lambda m: -abs(m[0]),
    )[:TOP]
    for change, group_id in groups:
        lines.append(
            f"  {_name(before, group_id)} {change * 100:+.1f} pts: "
            + "; ".join(_approval_reasons(before, after, group_id))
        )
    return lines


def _approval_reasons(before: WorldState, after: WorldState, group_id: str) -> list[str]:
    reasons = []
    events = []
    for e in after.events:
        effect = e.group_effects.get(group_id, 0.0) * _event_weight(
            e.age_turns, e.half_life_turns, e.hold_turns
        )
        if abs(effect) >= EVENT_FLOOR:
            events.append((effect, e.name))
    events.sort(key=lambda x: -abs(x[0]))
    reasons += [f'"{name}" {effect * 100:+.1f}' for effect, name in events[:2]]
    cares = []
    for edge in before.edges:
        if edge.kind == EdgeKind.CARES_ABOUT and edge.source == group_id:
            if edge.target in before.indicators and edge.target in after.indicators:
                change = after.indicators[edge.target].value - before.indicators[edge.target].value
                felt = edge.weight * change / scale(before, edge.target)
                if abs(change) >= MIN_MOVE:
                    cares.append((felt, edge.target, change))
    cares.sort(key=lambda x: -abs(x[0]))
    for _, ind_id, change in cares[:2]:
        reasons.append(f"cares about {_name(before, ind_id)} ({change:+.2f})")
    return reasons or ["drifting back towards their usual level"]


def alternatives(record: TurnRecord) -> list[str]:
    """The candidate outcomes, chosen one marked, with probability and score breakdown."""
    lines = ["The outcomes that were in play this turn:"]
    for i, outcome in enumerate(record.candidates):
        mark = "chosen" if i == record.chosen else "      "
        scores = ", ".join(f"{k} {v:.2f}" for k, v in outcome.scores.items() if k != "probability")
        first = outcome.narrative.split(". ")[0].rstrip(".")
        lines.append(f"  [{mark}] p={outcome.probability:.2f}  {first}.")
        if scores:
            lines.append(f"           ({scores})")
    return lines
