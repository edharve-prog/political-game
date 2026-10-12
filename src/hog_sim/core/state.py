"""WorldState: the single source of truth the engine reads and writes."""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import Field, model_validator

from hog_sim.core.models import (
    AnyNode,
    ApprovalEvent,
    Character,
    Country,
    Edge,
    Group,
    Indicator,
    Institution,
    Model,
    NodeKind,
    Pledge,
    Sector,
    Storyline,
)


class Push(Model):
    """A shock still running in the engine, in absolute turns (see world/propagation.py)."""

    node: str
    delta: float = Field(description="Standard steps")
    start: int = Field(description="First absolute turn it pushes")
    end: int = Field(description="Absolute turn it stops pushing (exclusive)")
    hold: bool = False


class EngineState(Model):
    """What the engine carries from one turn to the next (see world/propagation.py).

    Deviations are in standard steps from where each node would be without any shock."""

    own: dict[str, float] = Field(
        default_factory=dict, description="Each node's own push, still fading"
    )
    levels: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Each node's recent deviations, oldest first, as far back as the longest "
        "lag; a missing node has none",
    )
    pushes: list[Push] = Field(default_factory=list, description="Shocks still running")


class WorldState(Model):
    turn: int = 0
    player_country: str
    countries: dict[str, Country] = Field(default_factory=dict)
    sectors: dict[str, Sector] = Field(default_factory=dict)
    groups: dict[str, Group] = Field(default_factory=dict)
    institutions: dict[str, Institution] = Field(default_factory=dict)
    indicators: dict[str, Indicator] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)
    events: list[ApprovalEvent] = Field(default_factory=list)
    engine: EngineState = Field(
        default_factory=EngineState, description="Effects still in flight (the engine's state)"
    )
    pending: dict[int, dict[str, float]] = Field(
        default_factory=dict,
        description="Effects queued by saves made before the engine carried its own state: "
        "absolute turn -> node id -> native-unit change. Still applied; nothing adds to it.",
    )
    job_protection: dict[int, float] = Field(
        default_factory=dict,
        description="Wage subsidies in force: absolute turn -> share of job losses held back",
    )
    baselines: dict[str, float] = Field(
        default_factory=dict,
        description="What voters are used to (EB-11): node id -> the level of its primary "
        "metric they now take as normal. A node missing here is judged against the start.",
    )
    storylines: dict[str, Storyline] = Field(default_factory=dict)
    pledges: list[Pledge] = Field(default_factory=list, description="Promises made (SD-5)")
    characters: dict[str, Character] = Field(
        default_factory=dict, description="The cast, by id (SD-4)"
    )

    @model_validator(mode="after")
    def _check_references(self) -> WorldState:
        for attr in _NODE_COLLECTIONS:
            for key, node in getattr(self, attr).items():
                if key != node.id:
                    raise ValueError(f"{attr} key {key!r} does not match node id {node.id!r}")
                if not node.id.startswith(f"{_NODE_COLLECTIONS[attr]}:"):
                    raise ValueError(f"{attr} node id {node.id!r} has the wrong prefix")
        ids = set(self.node_ids())
        if self.player_country not in self.countries:
            raise ValueError(f"player_country {self.player_country!r} is not a country")
        for edge in self.edges:
            for end in (edge.source, edge.target):
                if end not in ids:
                    raise ValueError(f"edge {edge.kind} references unknown node {end!r}")
        for ind in self.indicators.values():
            if ind.controlled_by is not None and ind.controlled_by not in self.institutions:
                raise ValueError(
                    f"{ind.id} is controlled_by {ind.controlled_by!r}, which is not an institution"
                )
        if self.groups and sum(g.population_share for g in self.groups.values()) <= 0:
            raise ValueError("population groups must have a total population_share above 0")
        return self

    def nodes(self) -> Iterator[AnyNode]:
        for attr in _NODE_COLLECTIONS:
            yield from getattr(self, attr).values()

    def node_ids(self) -> Iterator[str]:
        return (node.id for node in self.nodes())

    def node(self, node_id: str) -> AnyNode:
        for attr in _NODE_COLLECTIONS:
            collection = getattr(self, attr)
            if node_id in collection:
                return collection[node_id]
        raise KeyError(node_id)

    def snapshot(self) -> WorldState:
        return self.model_copy(deep=True)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> WorldState:
        return cls.model_validate_json(data)

    def diff(self, other: WorldState) -> dict[str, dict[str, tuple[object, object]]]:
        """Changed fields per node id, as ``{node_id: {field: (self_value, other_value)}}``."""
        changes: dict[str, dict[str, tuple[object, object]]] = {}
        for attr in _NODE_COLLECTIONS:
            mine, theirs = getattr(self, attr), getattr(other, attr)
            for node_id in mine.keys() | theirs.keys():
                a = mine[node_id].model_dump() if node_id in mine else {}
                b = theirs[node_id].model_dump() if node_id in theirs else {}
                fields = {
                    k: (a.get(k), b.get(k)) for k in a.keys() | b.keys() if a.get(k) != b.get(k)
                }
                if fields:
                    changes[node_id] = fields
        if self.turn != other.turn:
            changes["world"] = {"turn": (self.turn, other.turn)}
        return changes


# Collection name -> node id prefix.
_NODE_COLLECTIONS = {
    "countries": NodeKind.COUNTRY,
    "sectors": NodeKind.SECTOR,
    "groups": NodeKind.GROUP,
    "institutions": NodeKind.INSTITUTION,
    "indicators": NodeKind.INDICATOR,
}
