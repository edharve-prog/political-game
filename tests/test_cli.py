import builtins

import pytest

from hog_sim.ui import cli
from hog_sim.ui.cli import check_llm


def test_offline_mode_says_so(tmp_path, monkeypatch, capsys) -> None:
    def no_input(prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", no_input)
    cli.main(["--save", str(tmp_path / "g.db")])
    assert "OFFLINE PRACTICE" in capsys.readouterr().out


def test_llm_mode_without_key_exits_with_help(monkeypatch) -> None:
    pytest.importorskip("anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        cli.main(["--llm"])


def test_check_llm_reports_the_model() -> None:
    from hog_sim.llm.client import FakeClient

    client = FakeClient([{"reply": "connected"}])
    assert "claude-opus-5-5 replied 'connected'" in check_llm(client)
