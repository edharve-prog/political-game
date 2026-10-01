"""SQLite saves: one row per game, one row per turn."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from hog_sim.core.config import GameConfig
from hog_sim.core.state import WorldState
from hog_sim.game.records import TurnRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    config TEXT NOT NULL,
    start_state TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS turns (
    game_id TEXT NOT NULL REFERENCES games(game_id),
    turn INTEGER NOT NULL,
    record TEXT NOT NULL,
    PRIMARY KEY (game_id, turn)
);
"""


class SaveStore:
    def __init__(self, path: str | Path) -> None:
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def new_game(self, config: GameConfig, start: WorldState) -> str:
        game_id = uuid.uuid4().hex[:12]
        with self.conn:
            self.conn.execute(
                "INSERT INTO games (game_id, config, start_state) VALUES (?, ?, ?)",
                (game_id, config.model_dump_json(), start.to_json()),
            )
        return game_id

    def save_turn(self, game_id: str, record: TurnRecord) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO turns (game_id, turn, record) VALUES (?, ?, ?)",
                (game_id, record.turn, record.model_dump_json()),
            )

    def load(self, game_id: str) -> tuple[GameConfig, WorldState, list[TurnRecord]]:
        row = self.conn.execute(
            "SELECT config, start_state FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
        if row is None:
            raise KeyError(game_id)
        rows = self.conn.execute(
            "SELECT record FROM turns WHERE game_id = ? ORDER BY turn", (game_id,)
        ).fetchall()
        return (
            GameConfig.model_validate_json(row[0]),
            WorldState.from_json(row[1]),
            [TurnRecord.model_validate_json(r[0]) for r in rows],
        )

    def latest_game(self) -> str | None:
        row = self.conn.execute(
            "SELECT game_id FROM games ORDER BY created_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None
