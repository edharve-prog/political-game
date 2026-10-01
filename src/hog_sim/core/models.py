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


class Institution(Node):
    support: float = Field(0.5, ge=0, le=1, description="Support for the government")
    power: float = Field(0.5, ge=0, le=1)
    independence: float = Field(0.5, ge=0, le=1)


class Indicator(Node):
    value: float
    unit: str = ""
    history: list[float] = Field(default_factory=list)


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
    kind: Literal[
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


class ApprovalEvent(Model):
    """A one-off hit or boost to some groups' approval that fades over time."""

    name: str
    group_effects: dict[str, float] = Field(description="Approval change per group, 0..1 scale")
    half_life_turns: float = Field(3.0, gt=0)
    age_turns: int = Field(0, ge=0)


class Scenario(Model):
    title: str
    briefing: str
    affected_nodes: list[str]
    urgency: float = Field(ge=0, le=1)
    source: Literal["news", "generated", "scheduled"] = "generated"
    suggested_options: list[str] = Field(default_factory=list)
    shocks: list[Shock] = Field(default_factory=list, description="Exogenous shocks it brings")


class Outcome(Model):
    narrative: str
    indicator_deltas: dict[str, float] = Field(
        default_factory=dict,
        description="Claimed native-unit changes; the engine's are authoritative",
    )
    events: list[str] = Field(default_factory=list)
    approval_events: list[ApprovalEvent] = Field(default_factory=list)
    shocks: list[Shock] = Field(default_factory=list, description="New shocks the outcome triggers")
    probability: float = Field(ge=0, le=1)
    scores: dict[str, float] = Field(
        default_factory=dict, description="Score breakdown behind probability, for logs and UI"
    )
