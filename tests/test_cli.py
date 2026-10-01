import builtins

import pytest

from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.ui import cli
from hog_sim.ui.cli import check_llm


def test_offline_mode_says_so(tmp_path, monkeypatch, capsys) -> None:
    def no_input(prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", no_input)
    cli.main(["--offline", "--save", str(tmp_path / "g.db")])
    out = capsys.readouterr().out
    assert "OFFLINE PRACTICE" in out
    assert "Could not reach Claude" not in out


def no_backend(tmp_path, monkeypatch) -> None:
    from hog_sim.llm import providers

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(providers.shutil, "which", lambda name: None)
    monkeypatch.chdir(tmp_path)  # no .env here


def test_plays_with_claude_by_default(tmp_path, monkeypatch, capsys) -> None:
    from hog_sim.llm.client import FakeClient

    def no_input(prompt=""):
        raise EOFError

    seen = {}

    def fake_make_client(provider):
        seen["provider"] = provider
        return FakeClient([]), "claude-code", None

    monkeypatch.setattr("hog_sim.llm.providers.make_client", fake_make_client)
    monkeypatch.setattr(
        "hog_sim.game.llm_plugins.llm_plugins",
        lambda client, **kw: (CannedScenarios(0), KeywordInterpreter(), EngineForecaster()),
    )
    monkeypatch.setattr(builtins, "input", no_input)
    monkeypatch.delenv("HOG_SIM_PROVIDER", raising=False)
    cli.main(["--save", str(tmp_path / "g.db")])
    out = capsys.readouterr().out
    assert "Mode: CLAUDE" in out and "OFFLINE" not in out
    assert seen["provider"] == "claude-code"


def test_falls_back_to_offline_and_says_why(tmp_path, monkeypatch, capsys) -> None:
    def no_input(prompt=""):
        raise EOFError

    no_backend(tmp_path, monkeypatch)
    monkeypatch.setattr(builtins, "input", no_input)
    cli.main(["--save", str(tmp_path / "g.db")])
    out = capsys.readouterr().out
    assert out.startswith(cli.NO_CLAUDE_NOTE)
    assert "ANTHROPIC_API_KEY=sk-ant-" in out and "OFFLINE PRACTICE" in out


def test_llm_mode_without_any_backend_exits_with_help(tmp_path, monkeypatch) -> None:
    no_backend(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="Claude Code") as exc:
        cli.main(["--llm"])
    assert "ANTHROPIC_API_KEY=sk-ant-" in str(exc.value)


def test_check_llm_reports_the_model() -> None:
    from hog_sim.llm.client import FakeClient

    client = FakeClient([{"reply": "connected"}])
    assert "claude-opus-5-5 replied 'connected'" in check_llm(client)
