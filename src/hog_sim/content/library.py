"""Scenario library: reusable scenarios for offline play (backlog story SD-3).

Entries come from three places and share one format, ``LibraryScenario``:

* handwritten JSON files in ``content/scenarios/<category>.json``
* scenarios Claude generated in earlier games, harvested by the knowledge store (Project 15),
  which hands them over as ``LibraryScenario`` objects or a JSONL export
* news, later (Project 7)

``ScenarioLibrary`` merges them, drops duplicates, and picks one per turn: seeded, filtered
by ``conditions`` on the current state, weighted, and never repeating a recent entry. It
satisfies the game loop's ``ScenarioSource`` protocol.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal, get_args

from pydantic import Field

from hog_sim.core.config import make_rng
from hog_sim.core.models import Category, Model
from hog_sim.core.state import WorldState
from hog_sim.game.records import TurnRecord
from hog_sim.llm.scenario_gen import GeneratedScenario

CATEGORIES: tuple[str, ...] = get_args(Category)

BUILTIN_DIR = Path(__file__).parent / "scenarios"
NO_REPEAT_TURNS = 15


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def normalise_title(title: str) -> str:
    return " ".join(slugify(title).split("-"))


class Provenance(Model):
    model: str | None = None
    prompt_version: str | None = None
    game_id: str | None = None
    turn: int | None = None
    created_at: datetime | None = None


class Condition(Model):
    """Holds when the node's value (indicator value, or group approval) compares true."""

    op: Literal["<", "<=", ">", ">="]
    value: float

    def holds(self, actual: float) -> bool:
        return {
            "<": actual < self.value,
            "<=": actual <= self.value,
            ">": actual > self.value,
            ">=": actual >= self.value,
        }[self.op]


class LibraryScenario(GeneratedScenario):
    id: str
    category: Category
    conditions: dict[str, Condition] = Field(
        default_factory=dict, description="node id -> condition; all must hold"
    )
    weight: float = Field(1.0, gt=0)
    origin: Literal["handwritten", "llm", "news"] = "handwritten"
    provenance: Provenance | None = None

    def node_ids(self) -> set[str]:
        ids = set(self.affected_nodes) | set(self.conditions)
        ids |= {s.node for s in self.shocks}
        ids |= {p.node for p in self.stakeholder_positions}
        return ids

    def applies_to(self, state: WorldState) -> bool:
        for node_id, cond in self.conditions.items():
            if node_id in state.indicators:
                actual = state.indicators[node_id].value
            elif node_id in state.groups:
                actual = state.groups[node_id].approval
            else:
                return False
            if not cond.holds(actual):
                return False
        return True


Source = str | Path | Iterable[LibraryScenario]


def _read(path: Path) -> list[LibraryScenario]:
    if path.is_dir():
        return [s for p in sorted(path.glob("*.json*")) for s in _read(p)]
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
    return [LibraryScenario.model_validate(r) for r in rows]


class ScenarioLibrary:
    def __init__(self, scenarios: Iterable[LibraryScenario], seed: int = 0) -> None:
        self.seed = seed
        self.scenarios: list[LibraryScenario] = []
        seen_ids: set[str] = set()
        seen_titles: set[str] = set()
        for s in scenarios:
            title = normalise_title(s.title)
            if s.id in seen_ids or title in seen_titles:
                continue
            seen_ids.add(s.id)
            seen_titles.add(title)
            self.scenarios.append(s)

    @classmethod
    def load(cls, *sources: Source, seed: int = 0, builtin: bool = True) -> ScenarioLibrary:
        """Merge sources in order: the built-in files first (unless ``builtin=False``), then
        each path (a .json list, a .jsonl export, or a directory of them) or iterable of
        ``LibraryScenario``. Earlier entries win on duplicate ids or titles."""
        merged: list[LibraryScenario] = []
        if builtin and BUILTIN_DIR.exists():
            merged += _read(BUILTIN_DIR)
        for source in sources:
            if isinstance(source, str | Path):
                merged += _read(Path(source))
            else:
                merged += list(source)
        return cls(merged, seed)

    def check(self, state: WorldState) -> list[str]:
        """Problems with entries that refer to nodes this world doesn't have."""
        known = set(state.node_ids())
        return [
            f"{s.id}: unknown node ids {sorted(s.node_ids() - known)}"
            for s in self.scenarios
            if not s.node_ids() <= known
        ]

    def compatible(self, state: WorldState) -> list[LibraryScenario]:
        known = set(state.node_ids())
        return [s for s in self.scenarios if s.node_ids() <= known]

    def next_scenario(self, state: WorldState, history: list[TurnRecord]) -> LibraryScenario:
        recent = {normalise_title(r.scenario.title) for r in history[-NO_REPEAT_TURNS:]}
        pool = self.compatible(state)
        if not pool:
            raise ValueError("the scenario library has nothing that fits this world")
        # Conditions are hard (EC-10): a scenario whose preconditions fail is never shown.
        # Freshness is soft: fitting and fresh, then fitting but not last turn's, then any
        # fitting one, even a repeat.
        fitting = [s for s in pool if s.applies_to(state)]
        if not fitting:
            raise ValueError("no scenario in the library has conditions that hold right now")
        last = normalise_title(history[-1].scenario.title) if history else None
        choices = (
            [s for s in fitting if normalise_title(s.title) not in recent]
            or [s for s in fitting if normalise_title(s.title) != last]
            or fitting
        )
        rng = make_rng(self.seed, state.turn, "library")
        pick = rng.choices(choices, weights=[s.weight for s in choices])[0]
        return pick.model_copy(deep=True)
