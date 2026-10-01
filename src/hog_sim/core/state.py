"""WorldState: the single source of truth the engine reads and writes."""

from __future__ import annotations

from pydantic import Field

from hog_sim.core.models import (
    Country,
    Edge,
    Group,
    Indicator,
    Institution,
    Model,
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
        for attr in ("countries", "sectors", "groups", "institutions", "indicators"):
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
