import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from hog_sim.core.env import load_env, parse_env
from hog_sim.llm import providers
from hog_sim.llm.claude_code import ClaudeCodeClient
from hog_sim.llm.client import LLMError, LLMRequest, Message, structured_call
from hog_sim.llm.interpreter import Interpretation

# --- .env --------------------------------------------------------------------


def test_parse_env_handles_quotes_comments_and_export() -> None:
    text = (
        "# comment\n"
        "ANTHROPIC_API_KEY=sk-ant-123\n"
        'QUOTED="a b # not a comment"\n'
        "export EXPORTED=yes  # trailing comment\n"
        "BLANK=\n"
        "junk line\n"
    )
    assert parse_env(text) == {
        "ANTHROPIC_API_KEY": "sk-ant-123",
        "QUOTED": "a b # not a comment",
        "EXPORTED": "yes",
        "BLANK": "",
    }


def test_load_env_keeps_existing_values_and_skips_blanks(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("\ufeffHOG_A=from-file\nHOG_B=from-file\nHOG_C=\n", encoding="utf-8")
    monkeypatch.setenv("HOG_B", "from-env")
    monkeypatch.delenv("HOG_A", raising=False)
    monkeypatch.delenv("HOG_C", raising=False)
    assert load_env(env) == ["HOG_A"]
    assert os.environ["HOG_A"] == "from-file"
    assert os.environ["HOG_B"] == "from-env"
    assert "HOG_C" not in os.environ
    monkeypatch.delenv("HOG_A")


def test_load_env_without_file(tmp_path) -> None:
    assert load_env(tmp_path / "missing") == []


# --- Provider choice ---------------------------------------------------------


@pytest.fixture
def no_credentials(tmp_path, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    return tmp_path


def test_auto_prefers_api_credentials(no_credentials, monkeypatch) -> None:
    monkeypatch.setattr(providers.shutil, "which", lambda name: "/usr/bin/claude")
    assert providers.resolve_provider("auto") == "claude-code"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    assert providers.resolve_provider("auto") == "api"


def test_ant_login_profile_counts_as_api_credentials(no_credentials) -> None:
    assert not providers.has_api_credentials()
    (no_credentials / "credentials").mkdir()
    (no_credentials / "credentials" / "default.json").write_text("{}")
    assert providers.has_api_credentials()


def test_auto_with_nothing_explains_the_options(no_credentials, monkeypatch) -> None:
    monkeypatch.setattr(providers.shutil, "which", lambda name: None)
    with pytest.raises(LLMError, match="ant auth login"):
        providers.resolve_provider("auto")


def test_explicit_provider_is_respected(no_credentials) -> None:
    assert providers.resolve_provider("claude-code") == "claude-code"
    with pytest.raises(LLMError):
        providers.resolve_provider("openai")


# --- Claude Code client ------------------------------------------------------


def request(**overrides) -> LLMRequest:
    data = dict(
        model="claude-opus-5-5",
        system="You map text to actions.\nLine two.",
        messages=[Message(role="user", content="Tax energy firms")],
        output_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        schema_name="Interpretation",
        prompt_version="interpret-1",
        effort="low",
    )
    return LLMRequest(**(data | overrides))


def cli_output(structured=None, result="", is_error=False) -> str:
    return json.dumps(
        {
            "type": "result",
            "is_error": is_error,
            "result": result,
            "structured_output": structured,
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 5,
                "output_tokens": 40,
            },
            "modelUsage": {
                "claude-haiku-4-5": {"outputTokens": 3},
                "claude-opus-5-5": {"outputTokens": 40},
            },
        }
    )


class FakeRunner:
    def __init__(self, *stdouts: str) -> None:
        self.stdouts = list(stdouts)
        self.calls: list[tuple[list[str], str]] = []

    def __call__(self, argv, input, **kwargs):
        self.calls.append((argv, input))
        return SimpleNamespace(stdout=self.stdouts.pop(0), stderr="", returncode=0)


def test_claude_code_sends_system_and_schema_as_flags() -> None:
    runner = FakeRunner(cli_output(structured={"actions": []}))
    client = ClaudeCodeClient("/usr/bin/claude", runner=runner)
    response = client.complete(request())
    argv, stdin = runner.calls[0]
    assert argv[:2] == ["/usr/bin/claude", "-p"]
    assert argv[argv.index("--model") + 1] == "claude-opus-5-5"
    assert argv[argv.index("--effort") + 1] == "low"
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--system-prompt") + 1].startswith("You map text")
    assert json.loads(argv[argv.index("--json-schema") + 1])["type"] == "object"
    assert stdin == "Tax energy firms"
    assert json.loads(response.text) == {"actions": []}
    assert response.served_model == "claude-opus-5-5"
    record = client.usage.records[0]
    assert record.input_tokens == 115 and record.output_tokens == 40
    assert record.cost_usd == 0  # comes out of the Claude plan


def test_windows_cmd_shim_gets_free_text_on_stdin() -> None:
    runner = FakeRunner(cli_output(result='```json\n{"actions": []}\n```'))
    client = ClaudeCodeClient(r"C:\Users\ed\AppData\Roaming\npm\claude.CMD", runner=runner)
    response = client.complete(request())
    argv, stdin = runner.calls[0]
    assert "--system-prompt" not in argv and "--json-schema" not in argv
    assert stdin.startswith("You map text") and '"properties"' in stdin
    assert stdin.endswith("Tax energy firms")
    assert json.loads(response.text) == {"actions": []}


def test_claude_code_retry_inlines_the_conversation() -> None:
    runner = FakeRunner(
        cli_output(structured={"actions": [{"kind": "tax"}]}),
        cli_output(structured={"actions": [], "clarifying_question": "Which firms?"}),
    )
    client = ClaudeCodeClient("/usr/bin/claude", runner=runner)
    result = structured_call(
        client,
        output_type=Interpretation,
        system="sys",
        prompt="Tax them",
        model="claude-opus-5-5",
        prompt_version="v1",
    )
    assert result.clarifying_question == "Which firms?"
    second_stdin = runner.calls[1][1]
    assert "[Your earlier reply]" in second_stdin and "rejected" in second_stdin


def test_claude_code_errors_explain_sign_in() -> None:
    client = ClaudeCodeClient(
        "/usr/bin/claude", runner=FakeRunner(cli_output(result="Not logged in", is_error=True))
    )
    with pytest.raises(LLMError, match="/login"):
        client.complete(request())
    garbage = ClaudeCodeClient("/usr/bin/claude", runner=FakeRunner("not json"))
    with pytest.raises(LLMError, match="Claude Code failed"):
        garbage.complete(request())


def test_claude_code_timeout() -> None:
    def slow(argv, input, **kwargs):
        raise subprocess.TimeoutExpired(argv, 1)

    with pytest.raises(LLMError, match="did not answer"):
        ClaudeCodeClient("/usr/bin/claude", runner=slow, timeout_s=1).complete(request())


def test_missing_claude_code(monkeypatch) -> None:
    import hog_sim.llm.claude_code as cc

    monkeypatch.setattr(cc.shutil, "which", lambda name: None)
    with pytest.raises(LLMError, match="not installed"):
        ClaudeCodeClient()
