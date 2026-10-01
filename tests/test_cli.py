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


def test_llm_mode_without_any_backend_exits_with_help(tmp_path, monkeypatch) -> None:
    from hog_sim.llm import providers

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(providers.shutil, "which", lambda name: None)
    monkeypatch.chdir(tmp_path)  # no .env here
    with pytest.raises(SystemExit, match="Claude Code") as exc:
        cli.main(["--llm"])
    assert "ANTHROPIC_API_KEY=sk-ant-" in str(exc.value)


def test_check_llm_reports_the_model() -> None:
    from hog_sim.llm.client import FakeClient

    client = FakeClient([{"reply": "connected"}])
    assert "claude-opus-5-5 replied 'connected'" in check_llm(client)
