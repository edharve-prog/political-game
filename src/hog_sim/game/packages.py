"""The policy library: action packages the player saved to reuse (backlog story RB-5).

At the review step ``save <name>`` keeps the confirmed actions under a name; at the response
prompt ``use <name>`` loads them back into the review step for editing, and ``packages``
lists them. Packages live in the save file, one set per game, and ``hog-sim packages``
exports and imports them as JSON.
"""

from __future__ import annotations

import json
import re
import sqlite3

from hog_sim.core.models import PolicyAction

_SCHEMA = """
CREATE TABLE IF NOT EXISTS packages (
    game_id TEXT NOT NULL,
    name TEXT NOT NULL,
    actions TEXT NOT NULL,
    saved_turn INTEGER NOT NULL,
    PRIMARY KEY (game_id, name)
);
"""

_NAME = re.compile(r"^[\w-]{1,40}$")


def check_name(name: str) -> str:
    """The package name, lower-cased. Raises ``ValueError`` if it isn't one short word."""
    if not _NAME.match(name):
        raise ValueError("name a package with one word (letters, digits, - or _)")
    return name.lower()


class PolicyLibrary:
    """The saved packages of one game, kept in the save file's ``packages`` table."""

    def __init__(self, conn: sqlite3.Connection, game_id: str) -> None:
        self.conn = conn
        self.game_id = game_id
        self.conn.executescript(_SCHEMA)

    def save(self, name: str, actions: list[PolicyAction], turn: int) -> str:
        name = check_name(name)
        if not actions:
            raise ValueError("there are no actions to save")
        data = json.dumps([a.model_dump() for a in actions])
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO packages (game_id, name, actions, saved_turn) "
                "VALUES (?, ?, ?, ?)",
                (self.game_id, name, data, turn),
            )
        return name

    def get(self, name: str) -> list[PolicyAction]:
        row = self.conn.execute(
            "SELECT actions FROM packages WHERE game_id = ? AND name = ?",
            (self.game_id, check_name(name)),
        ).fetchone()
        if row is None:
            known = ", ".join(self.names()) or "none saved yet"
            raise ValueError(f"no package called {name!r} (packages: {known})")
        return [PolicyAction.model_validate(a) for a in json.loads(row[0])]

    def names(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT name FROM packages WHERE game_id = ? ORDER BY name", (self.game_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def listing(self) -> str:
        if not self.names():
            return "  No packages saved yet. At the review step, type save <name>."
        lines = ["  Saved packages (use <name> to load one):"]
        for name in self.names():
            actions = self.get(name)
            what = "; ".join(f"{a.kind} {a.target} {a.magnitude:+.2f}" for a in actions)
            lines.append(f"    {name}: {what}")
        return "\n".join(lines)

    def export(self) -> str:
        rows = self.conn.execute(
            "SELECT name, actions, saved_turn FROM packages WHERE game_id = ? ORDER BY name",
            (self.game_id,),
        ).fetchall()
        packages = [{"name": n, "saved_turn": t, "actions": json.loads(a)} for n, a, t in rows]
        return json.dumps(packages, indent=2)

    def import_(self, text: str) -> int:
        """Add the packages in ``text`` (from ``export``), replacing any with the same name."""
        packages = json.loads(text)
        for p in packages:
            actions = [PolicyAction.model_validate(a) for a in p["actions"]]
            self.save(p["name"], actions, int(p.get("saved_turn", 0)))
        return len(packages)
