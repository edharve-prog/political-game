"""Why did that happen? Explanations of a played turn (backlog story TT-2).

``why`` traces the turn's biggest indicator moves back to what caused them. It replays the
turn exactly as it was resolved, then again without one cause at a time (each of the player's
actions, the scenario's shocks, the chosen outcome's effects) and once without any of them.
A cause's share is what the turn would have lost without it, so effects that only the whole
package makes, such as the markets' verdict on tax cuts it leaves unpaid for, go to the
measures that made or prevented them. What happens with none of them (the tail of earlier
turns and background pressures) is its own share, and whatever is left over is how the causes
combined (shared limits, interactions), reported rather than hidden. The largest share is
named with the graph path it travelled. It then gives the reasons behind the biggest approval
moves: each group's target approval split into the terms the approval rules add up (lean,
what it cares about, where it works, institutions, the debt penalty and events), largest first.
``alternatives`` lists the outcomes that were not chosen, with their scores.

Everything here reads the turn record and the states either side of it; nothing changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx

from hog_sim.core.config import GameConfig
from hog_sim.core.models import Outcome, PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.game.loop import resolve
from hog_sim.game.records import TurnRecord
from hog_sim.population.popularity import (
    DEFICIT_TOLERANCE,
    Term,
    approval_terms,
    target_approval,
)
from hog_sim.world.events import outcome_shocks
from hog_sim.world.propagation import (
    CONFIDENCE_SHOCKS,
    PROPAGATING,
    actions_to_shocks,
    unfunded_tax_cut,
)

TOP = 3
MIN_MOVE = 1e-3
BACKGROUND = "the tail of earlier turns' policies and shocks, and background pressures"


def _name(state: WorldState, node_id: str) -> str:
    try:
        return state.node(node_id).name
    except KeyError:
        return node_id


@dataclass
class _Cause:
    label: str
    nodes: list[str]  # where it pushes first, for the path
    without: WorldState  # the turn resolved without it
    markets: str = ""  # how removing it changes the markets' verdict on the package


def _quiet(outcome: Outcome) -> Outcome:
    """The outcome with none of its effects on the world."""
    return outcome.model_copy(update={"shocks": [], "events": [], "graph_changes": []})


def _causes(
    record: TurnRecord, before: WorldState, start: WorldState, config: GameConfig
) -> tuple[list[_Cause], WorldState]:
    """Each cause with the turn resolved without it, and the turn resolved with none."""

    def rerun(actions: Sequence[PolicyAction], scenario: Scenario, outcome: Outcome) -> WorldState:
        return resolve(
            start,
            before,
            scenario,
            list(actions),
            outcome,
            config,
            record.pledges,
            record.delivery,
            record.sacked,
        )

    actions, scenario, outcome = record.actions, record.scenario, record.outcome
    unfunded = unfunded_tax_cut(actions, before)
    causes = []
    for i, action in enumerate(actions):
        if action.kind == "do_nothing":
            continue
        rest = [a for j, a in enumerate(actions) if j != i]
        markets = ""
        if unfunded_tax_cut(rest, before) > unfunded:
            markets = "by paying for tax cuts the markets would otherwise have punished"
        elif unfunded_tax_cut(rest, before) < unfunded:
            markets = "through the markets' verdict on unfunded tax cuts"
        causes.append(
            _Cause(
                f"your {action.kind} on {_name(before, action.target)} "
                f"(size {action.magnitude:+.2f})",
                [s.node for s in actions_to_shocks([action], before)] or [action.target],
                rerun(rest, scenario, outcome),
                markets,
            )
        )
    calm = scenario.model_copy(update={"shocks": []})
    if scenario.shocks:
        causes.append(
            _Cause(
                "the scenario itself",
                [s.node for s in scenario.shocks],
                rerun(actions, calm, outcome),
            )
        )
    shocks = outcome_shocks(outcome, before)
    if shocks or outcome.graph_changes:
        causes.append(
            _Cause(
                "what happened next (the chosen outcome)",
                [s.node for s in shocks],
                rerun(actions, scenario, _quiet(outcome)),
            )
        )
    return causes, rerun([], calm, _quiet(outcome))


def _value(state: WorldState, node_id: str) -> float:
    return state.indicators[node_id].value


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


COMBINED = "how this turn's causes combined"


def contributions(
    record: TurnRecord, before: WorldState, start: WorldState, config: GameConfig, node_id: str
) -> dict[str, float]:
    """Each cause's share of the turn's change in indicator ``node_id``, plus ``BACKGROUND``
    and ``COMBINED``. The shares add up to the change exactly."""
    causes, nothing = _causes(record, before, start, config)
    return _shares(record, before, causes, nothing, node_id)


def _shares(
    record: TurnRecord,
    before: WorldState,
    causes: list[_Cause],
    nothing: WorldState,
    node_id: str,
) -> dict[str, float]:
    actual = _value(record.state_after, node_id)
    shares = {c.label: actual - _value(c.without, node_id) for c in causes}
    shares[BACKGROUND] = _value(nothing, node_id) - _value(before, node_id)
    shares[COMBINED] = actual - _value(before, node_id) - sum(shares.values())
    return shares


def why(record: TurnRecord, before: WorldState, start: WorldState, config: GameConfig) -> list[str]:
    """Lines explaining the turn's biggest indicator and approval moves. ``start`` and
    ``config`` are the game's, so the turn can be resolved again without each cause."""
    after = record.state_after
    graph = _propagation_graph(before)
    causes, nothing = _causes(record, before, start, config)
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
        shares = _shares(record, before, causes, nothing, ind_id)
        background, residual = shares[BACKGROUND], shares[COMBINED]
        parts = [(c, shares[c.label]) for c in causes if abs(shares[c.label]) >= MIN_MOVE]
        top = max(parts, key=lambda p: abs(p[1]), default=None)
        if top is not None and abs(top[1]) >= abs(background):
            cause, size = top
            if cause.markets and ind_id in CONFIDENCE_SHOCKS:
                via = f" {cause.markets}"
            elif ind_id in cause.nodes:
                via = " directly"
            else:
                path = _path(graph, cause.nodes, ind_id, before)
                via = f" via {path}" if path else ""
            line += f": mostly {cause.label}, {size:+.2f} of it{via}"
            if abs(background) >= MIN_MOVE:
                line += f"; {background:+.2f} from earlier turns and background pressures"
        elif abs(background) >= MIN_MOVE:
            line += f": mostly {BACKGROUND}, {background:+.2f} of it"
        else:
            line += f": {BACKGROUND}"
        if abs(residual) >= MIN_MOVE:
            line += f"; {residual:+.2f} from {COMBINED}"
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
            + "; ".join(_approval_reasons(before, after, start, group_id))
        )
    return lines


def _judged(before: WorldState, after: WorldState) -> WorldState:
    """The state ``step_approval`` judged this turn: ``after`` with the baselines voters
    held coming into the turn and the events as old as they were then."""
    seen = after.snapshot()
    seen.baselines = dict(before.baselines)
    for event in seen.events:
        event.age_turns = max(0, event.age_turns - 1)
    return seen


def _term_label(state: WorldState, term: Term) -> str:
    if term.kind == "cares":
        return f"cares about {_name(state, term.node)}"
    if term.kind == "employs":
        return f"works in {_name(state, term.node)}"
    if term.kind == "influences":
        return f"{_name(state, term.node)}'s standing"
    if term.kind == "debt":
        return f"deficit above {DEFICIT_TOLERANCE:g}% of GDP"
    return "events in force (capped)"


def _approval_reasons(
    before: WorldState, after: WorldState, start: WorldState, group_id: str
) -> list[str]:
    """The terms of the group's target approval as ``target_approval`` computes them (E20):
    where it is heading, its lean, and the largest pushes away from the lean, in points."""
    seen = _judged(before, after)
    terms = approval_terms(seen, start, seen.baselines)[group_id]
    target = target_approval(seen, start, seen.baselines)[group_id]
    lean = next(t.value for t in terms if t.kind == "lean")
    pushes = []
    for term in terms:
        if term.kind == "lean":
            continue
        # The cap tapers rather than cuts off (E8), so the term is capped when it falls short
        # of its events' sum.
        capped = term.kind == "events" and abs(sum(e for _, e in term.events) - term.value) > 1e-9
        if term.kind == "events" and not capped:
            pushes += [(effect, f'"{name}"') for name, effect in term.events]
        else:
            pushes.append((term.value, _term_label(before, term)))
    pushes = [p for p in pushes if abs(p[0]) >= 5e-4]
    pushes.sort(key=lambda p: -abs(p[0]))
    head = f"heading for {target * 100:.0f}% (lean {lean * 100:.0f}%)"
    if not pushes:
        return [f"{head}, drifting back towards their usual level"]
    return [head, *(f"{label} {value * 100:+.1f}" for value, label in pushes[:TOP])]


def alternatives(record: TurnRecord) -> list[str]:
    """The candidate outcomes, chosen one marked, with relative likelihood and score
    breakdown. Likelihoods are relative until the base rates are calibrated (D7)."""
    lines = ["The outcomes that were in play this turn (relative likelihood, not calibrated):"]
    for i, outcome in enumerate(record.candidates):
        mark = "chosen" if i == record.chosen else "      "
        scores = ", ".join(f"{k} {v:.2f}" for k, v in outcome.scores.items() if k != "probability")
        first = outcome.narrative.split(". ")[0].rstrip(".")
        lines.append(f"  [{mark}] likelihood {outcome.probability:.2f}  {first}.")
        if scores:
            lines.append(f"           ({scores})")
    return lines
