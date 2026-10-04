"""Compress a WorldState into the short briefing every prompt starts from.

The LLM never sees raw state. It sees this summary: what is under stress, who is angry,
where abroad is tense, what happened recently, and the catalogue of node ids it may refer
to. Outputs are then checked against that catalogue.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field

from hog_sim.core.models import Model
from hog_sim.core.state import WorldState
from hog_sim.policy.pledges import pledge_line
from hog_sim.world.graph import build_graph, edges_of

ROLE_TITLES = {"prime_minister": "Prime Minister", "president": "President"}


class IndicatorLine(Model):
    id: str
    name: str
    value: float
    unit: str
    change: float | None = Field(None, description="Relative change over the recent window")
    interventions: list[str] = Field(
        default_factory=list, description="Action kinds that may target it directly"
    )


class GroupLine(Model):
    id: str
    name: str
    approval: float
    population_share: float
    change: float | None = Field(None, description="Approval change over the recent window")


class CountryLine(Model):
    id: str
    name: str
    relationship: float
    stability: float


class InstitutionLine(Model):
    id: str
    name: str
    support: float
    independence: float


class StateSummary(Model):
    turn: int
    role: str
    player_country: str
    indicators: list[IndicatorLine]
    stressed_indicators: list[str]
    groups: list[GroupLine]
    angry_groups: list[str]
    turning_groups: list[str] = Field(
        default_factory=list, description="Groups whose approval fell sharply recently"
    )
    foreign: list[CountryLine]
    foreign_tensions: list[str]
    institutions: list[InstitutionLine]
    recent_events: list[str] = Field(default_factory=list)
    links: list[str] = Field(
        default_factory=list, description="Graph edges relevant to this call, one per line"
    )
    precedents: list[str] = Field(
        default_factory=list, description="Similar past situations from the knowledge store"
    )
    pledges: list[str] = Field(
        default_factory=list, description="Promises the leader made, kept or broken (SD-5)"
    )
    catalogue: dict[str, str] = Field(description="Every node id the model may reference -> name")

    def to_prompt(self) -> str:
        lines = [
            f"Turn {self.turn}. You are briefing the {ROLE_TITLES.get(self.role, self.role)} "
            f"of {self.catalogue[self.player_country]}.",
            "",
            "Indicators:",
        ]
        for i in self.indicators:
            change = f" ({i.change:+.0%} recently)" if i.change is not None else ""
            stress = "  [STRESSED]" if i.id in self.stressed_indicators else ""
            unit = i.unit if i.unit.startswith("%") or not i.unit else f" {i.unit}"
            direct = f"  [direct: {', '.join(i.interventions)}]" if i.interventions else ""
            lines.append(f"- {i.id} {i.name}: {i.value:g}{unit}{change}{stress}{direct}")
        lines += ["", "Population groups (approval of government, 0-1):"]
        for g in self.groups:
            change = f" ({g.change:+.2f} recently)" if g.change is not None else ""
            angry = "  [ANGRY]" if g.id in self.angry_groups else ""
            turning = "  [TURNING AGAINST YOU]" if g.id in self.turning_groups else ""
            lines.append(
                f"- {g.id} {g.name}: {g.approval:.2f}{change}, "
                f"{g.population_share:.0%} of voters{angry}{turning}"
            )
        lines += ["", "Foreign countries (relationship -1..1, stability 0..1):"]
        for c in self.foreign:
            tense = "  [TENSE]" if c.id in self.foreign_tensions else ""
            lines.append(f"- {c.id} {c.name}: {c.relationship:+.2f}, {c.stability:.2f}{tense}")
        lines += ["", "Institutions (support for government, independence):"]
        for inst in self.institutions:
            lines.append(f"- {inst.id} {inst.name}: {inst.support:.2f}, {inst.independence:.2f}")
        other = sorted(
            node_id
            for node_id in self.catalogue
            if node_id.startswith("sector:") or node_id == self.player_country
        )
        lines += ["", "Other nodes:"]
        lines += [f"- {node_id} {self.catalogue[node_id]}" for node_id in other]
        if self.recent_events:
            lines += ["", "Recent events (most recent last):"]
            lines += [f"- {e}" for e in self.recent_events]
        if self.pledges:
            lines += ["", "Pledges the leader has made (voters remember them):"]
            lines += [f"- {p}" for p in self.pledges]
        if self.links:
            lines += ["", "Links in the world graph (source -KIND-> target, weight, lag in turns):"]
            lines += [f"- {e}" for e in self.links]
        if self.precedents:
            lines += [
                "",
                "Precedents from earlier turns and games (reference only; the current numbers "
                "rule, and do not repeat them):",
            ]
            lines += [f"- {p}" for p in self.precedents]
        return "\n".join(lines)


def summarise_state(
    state: WorldState,
    role: str = "prime_minister",
    recent_events: list[str] | None = None,
    *,
    precedents: list[str] | None = None,
    window: int = 3,
    stress_threshold: float = 0.05,
    angry_below: float = 0.4,
    turning_drop: float = 0.02,
    max_events: int = 8,
) -> StateSummary:
    """Summarise ``state`` for a prompt.

    An indicator is stressed when it moved by at least ``stress_threshold`` (relative) over
    the last ``window`` turns of history. A group is angry below ``angry_below`` approval and
    turning when its approval fell by at least ``turning_drop`` over the window. A foreign
    country is tense with a negative relationship or stability under 0.4.
    """
    indicators = []
    stressed = []
    for ind in state.indicators.values():
        change = None
        if ind.history:
            past = ind.history[-window] if len(ind.history) >= window else ind.history[0]
            if past:
                change = (ind.value - past) / abs(past)
        indicators.append(
            IndicatorLine(
                id=ind.id,
                name=ind.name,
                value=ind.value,
                unit=ind.unit,
                change=change,
                interventions=list(ind.interventions),
            )
        )
        if change is not None and abs(change) >= stress_threshold:
            stressed.append(ind.id)

    groups = sorted(state.groups.values(), key=lambda g: g.approval)
    group_change = {
        g.id: g.approval - (g.history[-window] if len(g.history) >= window else g.history[0])
        for g in groups
        if g.history
    }
    foreign = [c for c in state.countries.values() if c.id != state.player_country]
    return StateSummary(
        turn=state.turn,
        role=role,
        player_country=state.player_country,
        indicators=indicators,
        stressed_indicators=stressed,
        groups=[
            GroupLine(
                id=g.id,
                name=g.name,
                approval=g.approval,
                population_share=g.population_share,
                change=group_change.get(g.id),
            )
            for g in groups
        ],
        angry_groups=[g.id for g in groups if g.approval < angry_below],
        turning_groups=[g.id for g in groups if group_change.get(g.id, 0.0) <= -turning_drop],
        foreign=[
            CountryLine(id=c.id, name=c.name, relationship=c.relationship, stability=c.stability)
            for c in foreign
        ],
        foreign_tensions=[c.id for c in foreign if c.relationship < 0 or c.stability < 0.4],
        institutions=[
            InstitutionLine(id=i.id, name=i.name, support=i.support, independence=i.independence)
            for i in state.institutions.values()
        ],
        recent_events=(recent_events or [])[-max_events:],
        precedents=list(precedents or []),
        pledges=[pledge_line(p) for p in state.pledges],
        catalogue={node.id: node.name for node in state.nodes()},
    )


def relevant_links(state: WorldState, node_ids: list[str], limit: int = 15) -> list[str]:
    """Edges touching ``node_ids`` (groups' CARES_ABOUT edges left out), for prompts that may
    propose graph changes."""
    graph = build_graph(state)
    seen: set[tuple[str, str, str]] = set()
    lines = []
    for node_id in node_ids:
        if node_id not in graph:
            continue
        for edge in edges_of(graph, node_id, direction="both"):
            key = (edge.source, edge.target, edge.kind)
            if key in seen or edge.kind == "CARES_ABOUT":
                continue
            seen.add(key)
            lines.append(
                f"{edge.source} -{edge.kind}-> {edge.target}, {edge.weight:+.2f}, {edge.lag}"
            )
    return lines[:limit]


def resolve_id(raw: str, ids: Iterable[str]) -> str:
    """Map a bare id (``business``) to its one prefixed id (``group:business``).

    Returns ``raw`` unchanged when it is already an id, or when no id or several ids match,
    so the check still reports it.
    """
    ids = list(ids)
    if raw in ids or ":" in raw:
        return raw
    matches = [i for i in ids if i.split(":", 1)[-1] == raw]
    return matches[0] if len(matches) == 1 else raw
