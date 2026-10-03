"""A log of every request sent to the model and every reply, kept small on disk.

``LoggingClient`` wraps any ``LLMClient`` and writes one row per call to a SQLite file
(``llm-log.db`` next to the save file by default): when, which game and turn, provider,
model, call type, prompt version, attempt number, tokens, latency, and either the reply or
the error.
``hog-sim logs`` reads it back as plain text.

Storage is built around how repetitive these calls are:
- Texts (system prompts, output schemas, prompts, replies) are stored once each in a
  ``texts`` table, found again by their hash. The same system prompt and schema go out on
  every call of a kind, and a retry resends the earlier prompt and reply, so most of a call
  is already stored and costs a small integer reference.
- Each turn's prompt of a kind is mostly the same as the last one (same briefing layout, a
  few numbers changed), so new texts are zlib-compressed against an earlier text of the same
  kind (its "base", used as a preset dictionary). When a text has drifted too far from its
  base to gain much, it is stored on its own and becomes the next base. Reading a text needs
  at most its base, never a chain.
- zlib at level 9 came within a few percent of lzma on these prompts at a fraction of the
  CPU, supports preset dictionaries, and is in every Python (zstd is built in from 3.14 only).
- Call rows hold numbers and short labels only, so listing and filtering never decompress.
- ``texts`` is an ordinary rowid table: kilobyte-sized rows in a WITHOUT ROWID table spill
  into overflow pages and more than doubled the file in a 24-turn test game.

The log is a separate file from the save so it can be deleted or copied without touching
games. Nothing here is read by the game itself; it is for people.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import zlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hog_sim.llm.client import LLMClient, LLMRequest, LLMResponse

log = logging.getLogger(__name__)

LOG_FILE = "llm-log.db"  # kept next to the save file
DEFAULT_PATH = f"saves/{LOG_FILE}"
LOG_ENV = "HOG_SIM_LLM_LOG"  # a path, or "off" to keep no log


def log_disabled(path: str) -> bool:
    return path.strip().lower() in ("off", "none", "no", "false", "0", "")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS texts (
    id INTEGER PRIMARY KEY,
    hash BLOB NOT NULL UNIQUE,
    raw_size INTEGER NOT NULL,
    slot TEXT,          -- set on bases: which kind of text this is the base for
    base_id INTEGER,    -- the base this text was compressed against, if any
    data BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS texts_slot ON texts (slot) WHERE slot IS NOT NULL;
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY,
    at TEXT NOT NULL,
    game_id TEXT,
    turn INTEGER,
    provider TEXT,
    model TEXT NOT NULL,
    served_model TEXT,
    schema_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    effort TEXT,
    attempt INTEGER NOT NULL,
    system INTEGER NOT NULL,
    output_schema INTEGER NOT NULL,
    messages TEXT NOT NULL,
    response INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    latency_ms INTEGER NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS calls_game_turn ON calls (game_id, turn);
"""

_HASH_BYTES = 16
# Keep a text compressed against its base only when that is at most this share of its size
# compressed alone; otherwise it becomes the new base.
_REBASE_RATIO = 0.6
_ROLES = {"user": "u", "assistant": "a"}  # messages are stored as e.g. "u12 a13 u14"


class CallLog:
    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(_SCHEMA)
        self._bases: dict[str, tuple[int, bytes] | None] = {}

    def close(self) -> None:
        self.conn.close()

    # --- Writing -----------------------------------------------------------

    def _base(self, slot: str) -> tuple[int, bytes] | None:
        if slot not in self._bases:
            row = self.conn.execute(
                "SELECT id, data FROM texts WHERE slot = ? ORDER BY id DESC LIMIT 1", (slot,)
            ).fetchone()
            self._bases[slot] = (row[0], zlib.decompress(row[1])) if row else None
        return self._bases[slot]

    def _put(self, text: str, slot: str) -> int:
        """The id of ``text`` in ``texts``, adding it compressed if it is new.

        ``slot`` names the kind of text (e.g. ``ScenarioDraft/prompt``); texts of one kind
        are compressed against that kind's current base.
        """
        raw = text.encode("utf-8")
        key = hashlib.blake2b(raw, digest_size=_HASH_BYTES).digest()
        row = self.conn.execute("SELECT id FROM texts WHERE hash = ?", (key,)).fetchone()
        if row:
            return row[0]
        alone = zlib.compress(raw, 9)
        base = self._base(slot)
        if base is not None:
            packer = zlib.compressobj(9, zdict=base[1])
            against = packer.compress(raw) + packer.flush()
            if len(against) <= _REBASE_RATIO * len(alone):
                return self.conn.execute(
                    "INSERT INTO texts (hash, raw_size, base_id, data) VALUES (?, ?, ?, ?)",
                    (key, len(raw), base[0], against),
                ).lastrowid
        text_id = self.conn.execute(
            "INSERT INTO texts (hash, raw_size, slot, data) VALUES (?, ?, ?, ?)",
            (key, len(raw), slot, alone),
        ).lastrowid
        self._bases[slot] = (text_id, raw)
        return text_id

    def _put_message(self, schema_name: str, index: int, message) -> str:
        if message.role == "assistant":
            slot = "reply"
        else:
            slot = "prompt" if index == 0 else "feedback"
        return f"{_ROLES[message.role]}{self._put(message.content, f'{schema_name}/{slot}')}"

    def record(
        self,
        request: LLMRequest,
        response: LLMResponse | None,
        *,
        latency_s: float,
        provider: str | None = None,
        game_id: str | None = None,
        turn: int | None = None,
        error: str | None = None,
    ) -> int:
        name = request.schema_name
        messages = " ".join(self._put_message(name, i, m) for i, m in enumerate(request.messages))
        cur = self.conn.execute(
            "INSERT INTO calls (at, game_id, turn, provider, model, served_model, schema_name,"
            " prompt_version, effort, attempt, system, output_schema, messages, response,"
            " input_tokens, output_tokens, latency_ms, error)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(UTC).isoformat(timespec="seconds"),
                game_id,
                turn,
                provider,
                request.model,
                response.served_model if response else None,
                request.schema_name,
                request.prompt_version,
                request.effort,
                (len(request.messages) + 1) // 2,
                self._put(request.system, f"{name}/system"),
                self._put(json.dumps(request.output_schema, sort_keys=True), f"{name}/schema"),
                messages,
                self._put(response.text, f"{name}/reply") if response else None,
                response.usage.input_tokens if response else None,
                response.usage.output_tokens if response else None,
                round(latency_s * 1000),
                error,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    # --- Reading -----------------------------------------------------------

    def _get(self, text_id: int | None) -> str | None:
        if text_id is None:
            return None
        row = self.conn.execute(
            "SELECT texts.data, base.data FROM texts LEFT JOIN texts AS base"
            " ON base.id = texts.base_id WHERE texts.id = ?",
            (text_id,),
        ).fetchone()
        if row is None:
            return None
        data, base = row
        if base is None:
            return zlib.decompress(data).decode("utf-8")
        unpacker = zlib.decompressobj(zdict=zlib.decompress(base))
        return (unpacker.decompress(data) + unpacker.flush()).decode("utf-8")

    _SUMMARY = (
        "id, at, game_id, turn, provider, model, served_model, schema_name, prompt_version,"
        " attempt, input_tokens, output_tokens, latency_ms, error"
    )

    def calls(
        self,
        *,
        game_id: str | None = None,
        turn: int | None = None,
        schema_name: str | None = None,
        errors_only: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Call summaries (no texts), oldest first; ``limit`` keeps the most recent."""
        where, params = [], []
        for column, value in (("game_id", game_id), ("turn", turn), ("schema_name", schema_name)):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        if errors_only:
            where.append("error IS NOT NULL")
        sql = f"SELECT {self._SUMMARY} FROM calls"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        cur = self.conn.execute(sql, params)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row, strict=True)) for row in reversed(cur.fetchall())]

    def call(self, call_id: int) -> dict[str, Any] | None:
        """One call in full: the summary plus system prompt, schema, messages and reply."""
        cur = self.conn.execute(
            f"SELECT {self._SUMMARY}, effort, system, output_schema, messages, response"
            " FROM calls WHERE id = ?",
            (call_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        entry = dict(zip([d[0] for d in cur.description], row, strict=True))
        entry["messages"] = [
            {"role": "user" if ref[0] == "u" else "assistant", "content": self._get(int(ref[1:]))}
            for ref in entry["messages"].split()
        ]
        entry["system"] = self._get(entry["system"])
        entry["output_schema"] = json.loads(self._get(entry["output_schema"]) or "null")
        entry["response"] = self._get(entry["response"])
        return entry

    def stats(self) -> dict[str, int]:
        """Call count, text bytes before and after compression, and the file size."""
        calls = self.conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        texts, raw, stored = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(raw_size), 0), COALESCE(SUM(LENGTH(data)), 0) FROM texts"
        ).fetchone()
        # What the texts would take if every call stored its own uncompressed copy.
        logical = 0
        sizes = dict(self.conn.execute("SELECT id, raw_size FROM texts"))
        for system, schema, messages, response in self.conn.execute(
            "SELECT system, output_schema, messages, response FROM calls"
        ):
            ids = [system, schema, response] + [int(ref[1:]) for ref in messages.split()]
            logical += sum(sizes.get(i, 0) for i in ids if i is not None)
        return {
            "calls": calls,
            "texts": texts,
            "logical_bytes": logical,
            "unique_bytes": raw,
            "stored_bytes": stored,
            "file_bytes": self.path.stat().st_size,
        }


class LoggingClient:
    """Wraps a client and writes every call, successful or not, to a ``CallLog``.

    Set ``game_id``, and ``turn_of`` to a function returning the turn being played, so rows
    can be filtered by game and turn. Other attributes (``usage``, ``name``, ``subscription``)
    are the wrapped client's.
    """

    def __init__(self, inner: LLMClient, log: CallLog, provider: str | None = None) -> None:
        self.inner = inner
        self.log = log
        self.provider = provider
        self.game_id: str | None = None
        self.turn_of: Callable[[], int | None] = lambda: None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        try:
            response = self.inner.complete(request)
        except Exception as exc:
            self._write(request, None, start, f"{type(exc).__name__}: {exc}")
            raise
        self._write(request, response, start, None)
        return response

    def _write(self, request, response, start, error) -> None:
        try:
            self.log.record(
                request,
                response,
                latency_s=time.perf_counter() - start,
                provider=self.provider,
                game_id=self.game_id,
                turn=self.turn_of(),
                error=error,
            )
        except sqlite3.Error as exc:  # a broken log must never stop the game
            log.warning("could not write the LLM call log: %s", exc)
