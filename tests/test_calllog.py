import json

import pytest

from hog_sim.core.models import Model
from hog_sim.llm.calllog import CallLog, LoggingClient
from hog_sim.llm.client import FakeClient, LLMUnavailable, structured_call
from hog_sim.ui import cli


class Ping(Model):
    reply: str


def ping(client, system="You are a connectivity check.", prompt="Say connected") -> Ping:
    return structured_call(
        client,
        output_type=Ping,
        system=system,
        prompt=prompt,
        model="claude-opus-5-5",
        prompt_version="ping-1",
        effort="low",
    )


@pytest.fixture
def calllog(tmp_path):
    log = CallLog(tmp_path / "llm-log.db")
    yield log
    log.close()


def test_logs_request_and_reply_with_game_and_turn(calllog) -> None:
    client = LoggingClient(FakeClient([{"reply": "connected"}]), calllog, "claude-code")
    client.game_id, client.turn_of = "g1", lambda: 4
    ping(client)

    [row] = calllog.calls()
    assert (row["game_id"], row["turn"], row["provider"]) == ("g1", 4, "claude-code")
    assert (row["schema_name"], row["attempt"], row["error"]) == ("Ping", 1, None)
    entry = calllog.call(row["id"])
    assert entry["system"] == "You are a connectivity check."
    assert entry["messages"] == [{"role": "user", "content": "Say connected"}]
    assert json.loads(entry["response"]) == {"reply": "connected"}
    assert entry["output_schema"]["title"] == "Ping"
    # The wrapper is transparent to code that reads the client's usage.
    assert len(client.usage.records) == 1


def test_retries_are_logged_and_repeated_text_is_stored_once(calllog) -> None:
    client = LoggingClient(FakeClient(["not json", {"reply": "ok"}]), calllog)
    ping(client)
    ping(LoggingClient(FakeClient([{"reply": "ok"}]), calllog))

    first, second, third = calllog.calls()
    assert (first["attempt"], second["attempt"], third["attempt"]) == (1, 2, 1)
    retry = calllog.call(second["id"])
    assert [m["role"] for m in retry["messages"]] == ["user", "assistant", "user"]
    assert retry["messages"][1]["content"] == "not json"
    assert "rejected" in retry["messages"][2]["content"]

    # system, schema, prompt, "not json", feedback, the reply: six texts for three calls.
    stats = calllog.stats()
    assert stats["texts"] == 6
    assert stats["unique_bytes"] < stats["logical_bytes"]


def test_failed_calls_are_logged_and_still_raise(calllog) -> None:
    class Down:
        usage = FakeClient().usage

        def complete(self, request):
            raise LLMUnavailable("Claude Code: rate limited")

    with pytest.raises(LLMUnavailable):
        ping(LoggingClient(Down(), calllog))
    [row] = calllog.calls(errors_only=True)
    assert row["error"] == "LLMUnavailable: Claude Code: rate limited"
    assert calllog.call(row["id"])["response"] is None


def test_long_texts_are_compressed(calllog) -> None:
    prompt = "Indicators:\n" + "\n".join(f"- indicator:{i} 3.5% (+1%)" for i in range(400))
    ping(LoggingClient(FakeClient([{"reply": "ok"}]), calllog), prompt=prompt)
    stats = calllog.stats()
    assert stats["stored_bytes"] * 4 < stats["unique_bytes"]
    assert calllog.call(1)["messages"][0]["content"] == prompt


def test_similar_prompts_are_compressed_against_a_base(calllog) -> None:
    client = LoggingClient(FakeClient(responder=lambda r: {"reply": "ok"}), calllog)

    def briefing(turn):
        # Each turn moves a few of the numbers; the rest of the briefing stays the same.
        lines = (f"- indicator:{i} {3.5 + (turn if i % 50 == 0 else 0)}%" for i in range(300))
        return f"Turn {turn}.\nIndicators:\n" + "\n".join(lines)

    for turn in range(3):
        ping(client, prompt=briefing(turn))
    ping(client, prompt="Something else entirely. " * 5 + "x" * 2000)

    rows = calllog.conn.execute(
        "SELECT id, slot, base_id, raw_size, LENGTH(data) FROM texts WHERE slot = 'Ping/prompt'"
        " OR base_id IN (SELECT id FROM texts WHERE slot = 'Ping/prompt') ORDER BY id"
    ).fetchall()
    first, second, third, other = rows
    assert first[1] == "Ping/prompt" and first[2] is None
    assert second[2] == first[0] and third[2] == first[0]
    assert second[4] < first[4] / 2
    assert other[1] == "Ping/prompt" and other[2] is None  # too different: a new base
    for i, turn in enumerate(range(3), start=1):
        assert calllog.call(i)["messages"][0]["content"] == briefing(turn)


def test_filters(calllog) -> None:
    client = LoggingClient(FakeClient(responder=lambda r: {"reply": "ok"}), calllog)
    for turn in (1, 1, 2):
        client.turn_of = lambda turn=turn: turn
        ping(client)
    assert len(calllog.calls(turn=1)) == 2
    assert [r["id"] for r in calllog.calls(limit=2)] == [2, 3]
    assert calllog.calls(schema_name="Nope") == []


def fake_backend(monkeypatch, replies):
    monkeypatch.setattr(
        "hog_sim.llm.providers.make_client",
        lambda provider: (FakeClient(replies), "claude-code", None),
    )


def test_check_llm_writes_the_log_next_to_the_save(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("HOG_SIM_LLM_LOG", raising=False)
    fake_backend(monkeypatch, [{"reply": "connected"}])
    cli.main(["--check-llm", "--save", str(tmp_path / "g.db")])
    db = str(tmp_path / "llm-log.db")

    cli.main(["logs", "--db", db, "list"])
    out = capsys.readouterr().out
    assert "Ping" in out and "claude-code" in out

    cli.main(["logs", "--db", db, "show", "last"])
    out = capsys.readouterr().out
    assert "SYSTEM PROMPT" in out and "Reply with the single word: connected" in out
    assert '"reply": "connected"' in out

    cli.main(["logs", "--db", db, "export", str(tmp_path / "calls.jsonl")])
    [line] = (tmp_path / "calls.jsonl").read_text().splitlines()
    assert json.loads(line)["schema_name"] == "Ping"

    cli.main(["logs", "--db", db, "stats"])
    assert "1 calls" in capsys.readouterr().out


def test_log_can_be_turned_off(tmp_path, monkeypatch) -> None:
    fake_backend(monkeypatch, [{"reply": "connected"}])
    cli.main(["--check-llm", "--save", str(tmp_path / "g.db"), "--llm-log", "off"])
    assert not (tmp_path / "llm-log.db").exists()


def test_logs_without_a_log_explains(tmp_path) -> None:
    with pytest.raises(SystemExit, match="No log at"):
        cli.main(["logs", "--db", str(tmp_path / "none.db"), "list"])
