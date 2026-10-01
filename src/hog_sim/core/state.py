"""WorldState: the single source of truth the engine reads and writes."""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import Field, model_validator

from hog_sim.core.models import (
    AnyNode,
    ApprovalEvent,
    Country,
    Edge,
    Group,
    Indicator,
    Institution,
    Model,
    NodeKind,
    Sector,
)


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
