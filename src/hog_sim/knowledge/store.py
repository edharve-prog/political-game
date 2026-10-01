"""The knowledge store: validated LLM output kept in SQLite for reuse.

After each Claude turn is saved, ``harvest_turn`` files what Claude wrote and the game
accepted: the scenario (as a ``LibraryScenario`` for SD-3's offline library), how the
player's response was interpreted, the candidate outcomes with their scores, and every
proposed graph change. Later games read it back for prompt precedents (``recall.py``) and
offline play (``offline.py``).

Only output that already passed ``structured_call`` validation and the engine's checks is in
a ``TurnRecord``, so nothing unvalidated is stored. Payloads are validated again on read:
a row written by an older schema that no longer parses is skipped and logged.

By default the store shares the save file (``saves/game.db``) in its own table.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ValidationError

from hog_sim.content.library import Condition, LibraryScenario
from hog_sim.core.models import Scenario
from hog_sim.core.state import WorldState
from hog_sim.game.records import TurnRecord
from hog_sim.knowledge.entries import (
    GraphChangeEntry,
    InterpretationEntry,
    OutcomeEntry,
    Provenance,
    action_signature,
    guess_category,
    normalise_text,
    scenario_id,
)
from hog_sim.llm.summary import summarise_state

log = logging.getLogger(__name__)

Kind = Literal["scenario", "interpretation", "outcome", "graph_change"]
KINDS: tuple[Kind, ...] = ("scenario", "interpretation", "outcome", "graph_change")

_PAYLOAD: dict[str, type[BaseModel]] = {
    "scenario": LibraryScenario,
    "interpretation": InterpretationEntry,
    "outcome": OutcomeEntry,
    "graph_change": GraphChangeEntry,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    scenario_id TEXT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (kind, key)
);
CREATE INDEX IF NOT EXISTS knowledge_scenario ON knowledge (kind, scenario_id);
"""

T = TypeVar("T", bound=BaseModel)


def scenario_conditions(scenario: Scenario, state: WorldState) -> dict[str, Condition]:
    """Loose conditions under which a harvested scenario makes sense again offline.

    For each affected indicator that was stressed when the scenario was written, it must
    have moved at least half as far in the same direction; each affected group that was
    angry must be below 0.45 approval.
    """
    summary = summarise_state(state)
    conditions: dict[str, Condition] = {}
    for line in summary.indicators:
        if line.id not in scenario.affected_nodes or line.id not in summary.stressed_indicators:
            continue
        past = line.value / (1 + line.change) if line.change not in (None, -1) else line.value
        midpoint = round(past + (line.value - past) / 2, 3)
        conditions[line.id] = Condition(op=">=" if line.value >= past else "<=", value=midpoint)
    for group in summary.angry_groups:
        if group in scenario.affected_nodes:
            conditions[group] = Condition(op="<", value=0.45)
    return conditions


def to_library_scenario(
    scenario: Scenario, state: WorldState, provenance: Provenance
) -> LibraryScenario:
    data = scenario.model_dump()
    for extra in ("id", "category", "conditions", "weight", "origin", "provenance"):
        data.pop(extra, None)
    return LibraryScenario(
        **data,
        id=scenario_id(scenario),
        category=guess_category(scenario),
        conditions=scenario_conditions(scenario, state),
        origin="llm",
        provenance=provenance,
    )


class KnowledgeStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # --- writing ------------------------------------------------------------

    def put(self, kind: Kind, key: str, payload: BaseModel, scenario: str | None = None) -> None:
        """Insert or replace one entry; the newest entry for a key wins."""
        _PAYLOAD[kind].model_validate(payload.model_dump())  # right type for the kind
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO knowledge (kind, key, scenario_id, payload) "
                "VALUES (?, ?, ?, ?)",
                (kind, key, scenario, payload.model_dump_json()),
            )

    def harvest_turn(
        self,
        game_id: str,
        record: TurnRecord,
        state_before: WorldState,
        provenance: Provenance | None = None,
    ) -> int:
        """File one Claude turn. Returns the number of entries written.

        ``record.scenario`` should be the object the game played (a ``GeneratedScenario``
        keeps its stakeholder positions in memory even though the save drops them).
        """
        prov = (provenance or Provenance()).model_copy(
            update={"game_id": game_id, "turn": record.turn, "created_at": datetime.now(UTC)}
        )
        scenario = record.scenario
        sid = scenario_id(scenario)
        written = 0
        if getattr(scenario, "origin", "llm") == "llm":
            self.put("scenario", sid, to_library_scenario(scenario, state_before, prov), sid)
            written += 1
        # What the interpreter made of the response, before this turn's limits cut it down.
        interpreted = record.requested_actions or record.actions
        if interpreted:
            self.put(
                "interpretation",
                f"{sid}|{normalise_text(record.response)}",
                InterpretationEntry(
                    scenario_id=sid,
                    scenario_title=scenario.title,
                    response=record.response,
                    actions=interpreted,
                    provenance=prov,
                ),
                sid,
            )
            written += 1
        self.put(
            "outcome",
            f"{sid}|{action_signature(record.actions)}",
            OutcomeEntry(
                scenario_id=sid,
                scenario_title=scenario.title,
                affected_nodes=scenario.affected_nodes,
                response=record.response,
                actions=record.actions,
                candidates=record.candidates,
                chosen=record.chosen,
                provenance=prov,
            ),
            sid,
        )
        written += 1
        for i, candidate in enumerate(record.candidates):
            for j, change in enumerate(candidate.graph_changes):
                status = "applied" if i == record.chosen else "proposed"
                self.put(
                    "graph_change",
                    f"{game_id}|{record.turn}|{i}|{j}",
                    GraphChangeEntry(
                        change=change,
                        status=status,
                        scenario_title=scenario.title,
                        provenance=prov,
                    ),
                    sid,
                )
                written += 1
        return written

    # --- reading ------------------------------------------------------------

    def _read(self, kind: Kind, scenario: str | None = None) -> list[tuple[int, BaseModel]]:
        sql = "SELECT id, payload FROM knowledge WHERE kind = ?"
        args: list[object] = [kind]
        if scenario is not None:
            sql += " AND scenario_id = ?"
            args.append(scenario)
        rows = self.conn.execute(sql + " ORDER BY id", args).fetchall()
        out = []
        for row_id, payload in rows:
            try:
                out.append((row_id, _PAYLOAD[kind].model_validate_json(payload)))
            except ValidationError as exc:
                log.warning("skipping knowledge row %d (%s): %s", row_id, kind, exc.errors()[:1])
        return out

    def library_scenarios(self) -> list[LibraryScenario]:
        """Harvested scenarios, ready for ``ScenarioLibrary.load``."""
        return [p for _, p in self._read("scenario")]  # type: ignore[misc]

    def interpretations(self, scenario: str | None = None) -> list[InterpretationEntry]:
        return [p for _, p in self._read("interpretation", scenario)]  # type: ignore[misc]

    def outcomes(self, scenario: str | None = None) -> list[OutcomeEntry]:
        return [p for _, p in self._read("outcome", scenario)]  # type: ignore[misc]

    def graph_changes(self) -> list[GraphChangeEntry]:
        return [p for _, p in self._read("graph_change")]  # type: ignore[misc]

    def stats(self) -> dict[str, int]:
        counts = dict(self.conn.execute("SELECT kind, COUNT(*) FROM knowledge GROUP BY kind"))
        return {kind: counts.get(kind, 0) for kind in KINDS}

    def __len__(self) -> int:
        return sum(self.stats().values())

    # --- moving knowledge between installs -----------------------------------

    def export_jsonl(self, path: str | Path, kinds: tuple[Kind, ...] = KINDS) -> int:
        """Write every valid entry as one JSON object per line. Returns the count."""
        rows = self.conn.execute(
            "SELECT kind, key, scenario_id, payload FROM knowledge ORDER BY id"
        ).fetchall()
        n = 0
        with Path(path).open("w", encoding="utf-8") as f:
            for kind, key, sid, payload in rows:
                if kind not in kinds:
                    continue
                try:
                    _PAYLOAD[kind].model_validate_json(payload)
                except ValidationError:
                    continue
                entry = {
                    "kind": kind,
                    "key": key,
                    "scenario_id": sid,
                    "payload": json.loads(payload),
                }
                f.write(json.dumps(entry) + "\n")
                n += 1
        return n

    def import_jsonl(self, path: str | Path) -> int:
        """Load an ``export_jsonl`` file, validating every line. Returns the count."""
        n = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row["kind"]
            if kind not in _PAYLOAD:
                raise ValueError(f"unknown knowledge kind {kind!r}")
            self.put(
                kind,
                row["key"],
                _PAYLOAD[kind].model_validate(row["payload"]),
                row.get("scenario_id"),
            )
            n += 1
        return n

    def export_scenarios(self, path: str | Path) -> int:
        """Write harvested scenarios as SD-3 library JSONL (one ``LibraryScenario`` per line)."""
        scenarios = self.library_scenarios()
        with Path(path).open("w", encoding="utf-8") as f:
            for s in scenarios:
                f.write(s.model_dump_json() + "\n")
        return len(scenarios)
