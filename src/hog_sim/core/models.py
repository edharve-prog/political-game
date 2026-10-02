"""First-pass schemas for the world graph and the turn loop.

Node ids are namespaced strings, e.g. ``"country:uk"``, ``"sector:energy"``,
``"group:pensioners"``, ``"indicator:inflation"``. See section 5 of the plan.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Nodes -----------------------------------------------------------------


class NodeKind(StrEnum):
    COUNTRY = "country"
    SECTOR = "sector"
    GROUP = "group"
    INSTITUTION = "institution"
    INDICATOR = "indicator"


class Node(Model):
    id: str
    name: str

    @property
    def kind(self) -> NodeKind:
        return NodeKind(self.id.split(":", 1)[0])


class Country(Node):
    gdp_bn: float
    growth_pct: float
    relationship: float = Field(0.0, ge=-1, le=1, description="Stance towards the player")
    stability: float = Field(0.5, ge=0, le=1)


class Sector(Node):
    output_bn: float
    employment_k: float
    sentiment: float = Field(0.0, ge=-1, le=1)


class Group(Node):
    population_share: float = Field(ge=0, le=1)
    turnout: float = Field(ge=0, le=1)
    approval: float = Field(0.5, ge=0, le=1)
    lean: float = Field(
        0.5, ge=0, le=1, description="Approval the group drifts to when nothing changes"
    )
    history: list[float] = Field(
        default_factory=list, description="Approval at the start of each past turn, oldest first"
    )


class Institution(Node):
    support: float = Field(0.5, ge=0, le=1, description="Support for the government")
    power: float = Field(0.5, ge=0, le=1)
    independence: float = Field(0.5, ge=0, le=1)


ActionKind = Literal[
    "tax",
    "spend",
    "regulate",
    "deregulate",
    "diplomatic",
    "military",
    "communicate",
    "legislate",
    "appoint",
    "do_nothing",
]


class Indicator(Node):
    value: float
    unit: str = ""
    low: float | None = Field(None, description="Hard floor; the engine never goes below it")
    high: float | None = Field(None, description="Hard ceiling")
    persistence: float | None = Field(
        None,
        ge=0,
        le=1,
        description="Share of a shock's own push kept each turn; None uses the engine default",
    )
    controlled_by: str | None = Field(
        None, description="Institution that sets this indicator, e.g. the central bank"
    )
    interventions: list[ActionKind] = Field(
        default_factory=list,
        description="Action kinds that may aim at this indicator directly (a price cap is "
        "'regulate'); every other policy has to work through a sector, group or institution",
    )
    history: list[float] = Field(
        default_factory=list, description="Value at the start of each past turn, oldest first"
    )


AnyNode = Country | Sector | Group | Institution | Indicator


# --- Edges -----------------------------------------------------------------


class EdgeKind(StrEnum):
    TRADES_WITH = "TRADES_WITH"
    ALLIED_WITH = "ALLIED_WITH"
    RIVAL_OF = "RIVAL_OF"
    SUPPLIES = "SUPPLIES"
    EMPLOYS = "EMPLOYS"
    CARES_ABOUT = "CARES_ABOUT"
    DRIVES = "DRIVES"
    INFLUENCES = "INFLUENCES"


class Edge(Model):
    source: str
    target: str
    kind: EdgeKind
    weight: float
    lag: int = Field(0, ge=0, description="Turns before the effect lands")
    uncertainty: float = Field(0.0, ge=0, description="Std dev used in Monte Carlo draws")


# --- Turn loop -------------------------------------------------------------


class PolicyAction(Model):
    kind: ActionKind
    target: str
    magnitude: float = Field(ge=-1, le=1)
    duration_turns: int = Field(1, ge=1)
    requires: list[str] = Field(default_factory=list)
    rationale: str = ""


class Shock(Model):
    """An impulse to one node, in standard steps (see world/propagation.py)."""

    node: str
    delta: float = Field(description="Impulse in standard steps")
    start_turn: int = Field(0, ge=0, description="Turns from now")
    duration_turns: int = Field(1, ge=1)
    hold: bool = Field(
        False,
        description="Hold the push at delta for duration_turns (a policy that keeps running) "
        "instead of adding delta every turn",
    )


class ApprovalEvent(Model):
    """A one-off hit or boost to some groups' approval that fades over time."""

    name: str
    group_effects: dict[str, float] = Field(description="Approval change per group, 0..1 scale")
    half_life_turns: float = Field(3.0, gt=0)
    hold_turns: int = Field(0, ge=0, description="Turns at full strength before fading starts")
    age_turns: int = Field(0, ge=0)


class GraphChange(Model):
    """A proposed change to the world graph's structure or a node's standing.

    LLM outcomes may carry these; ``world/changes.py`` validates them against the state
    (known ids, whitelisted fields, size caps) and applies them inside ``resolve()``.

    - ``edge_weight``: move the weight of the existing ``source -edge_kind-> target`` edge by
      ``delta``.
    - ``add_edge``: create that edge with weight ``delta`` and the given ``lag``.
    - ``node_attr``: move ``attr`` of ``node`` by ``delta`` (clamped to the field's bounds).
    """

    kind: Literal["edge_weight", "add_edge", "node_attr"]
    source: str | None = None
    target: str | None = None
    edge_kind: EdgeKind | None = None
    node: str | None = None
    attr: str | None = None
    delta: float
    lag: int = Field(0, ge=0)
    reason: str = ""

    def describe(self) -> str:
        if self.kind == "node_attr":
            return f"{self.node}.{self.attr} {self.delta:+.2f}"
        verb = "new edge" if self.kind == "add_edge" else "edge"
        return f"{verb} {self.source} -{self.edge_kind}-> {self.target} {self.delta:+.2f}"


# Issue categories for scenarios (backlog story SD-3).
Category = Literal[
    "economy",
    "energy",
    "housing",
    "health",
    "education",
    "crime_justice",
    "immigration",
    "defence_security",
    "foreign_affairs",
    "environment_disasters",
    "party_scandal",
    "media_technology",
]


class SideIssue(Model):
    """A smaller item in the turn's in-tray, beside the lead scenario (backlog story SD-2)."""

    title: str
    briefing: str
    category: Category | None = None
    affected_nodes: list[str]
    urgency: float = Field(ge=0, le=1)
    storyline: str | None = None


class Scenario(Model):
    title: str
    briefing: str
    affected_nodes: list[str]
    urgency: float = Field(ge=0, le=1)
    source: Literal["news", "generated", "scheduled"] = "generated"
    suggested_options: list[str] = Field(default_factory=list)
    shocks: list[Shock] = Field(default_factory=list, description="Exogenous shocks it brings")
    category: Category | None = None
    storyline: str | None = Field(
        None, description="Id of the storyline this scenario continues or opens (SD-1)"
    )
    secondary: list[SideIssue] = Field(
        default_factory=list, description="Other items in this turn's in-tray (SD-2)"
    )


class Storyline(Model):
    """An issue that runs across turns (backlog story SD-1). Kept in ``WorldState``."""

    id: str
    title: str
    category: Category | None = None
    stage: int = Field(1, ge=1)
    open: bool = True
    nodes: list[str] = Field(default_factory=list)
    opened_turn: int = Field(ge=0)
    last_turn: int = Field(ge=0, description="Last turn a scenario continued or escalated it")
    last_addressed: int = Field(
        ge=0, description="Last turn a scenario put it in front of the player"
    )
    pressure: float = Field(0.5, ge=0, le=1, description="How urgent it has become")
    history: list[str] = Field(default_factory=list, description="One line per stage, oldest first")


class Outcome(Model):
    narrative: str
    indicator_deltas: dict[str, float] = Field(
        default_factory=dict,
        description="Claimed native-unit changes; the engine's are authoritative",
    )
    events: list[str] = Field(default_factory=list)
    approval_events: list[ApprovalEvent] = Field(default_factory=list)
    shocks: list[Shock] = Field(default_factory=list, description="New shocks the outcome triggers")
    graph_changes: list[GraphChange] = Field(
        default_factory=list, description="Validated changes to the world graph, applied on resolve"
    )
    resolves_storyline: bool = Field(
        False, description="This outcome ends the scenario's storyline"
    )
    probability: float = Field(ge=0, le=1)
    scores: dict[str, float] = Field(
        default_factory=dict, description="Score breakdown behind probability, for logs and UI"
    )
