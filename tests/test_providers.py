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


def test_claude_code_is_the_default(no_credentials, monkeypatch) -> None:
    monkeypatch.setattr(providers.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(providers, "claude_code_signed_in", lambda exe, runner=None: True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    assert providers.DEFAULT_PROVIDER == "claude-code"
    assert providers.choose_provider("claude-code") == ("claude-code", None)
    assert providers.choose_provider("auto") == ("claude-code", None)
    assert providers.choose_provider("api") == ("api", None)


def test_falls_back_to_the_api_when_claude_code_is_missing(no_credentials, monkeypatch) -> None:
    monkeypatch.setattr(providers.shutil, "which", lambda name: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    provider, note = providers.choose_provider("claude-code")
    assert provider == "api"
    assert note.startswith("Claude Code isn't installed")


def test_falls_back_to_the_api_when_claude_code_is_signed_out(no_credentials, monkeypatch) -> None:
    monkeypatch.setattr(providers.shutil, "which", lambda name: "/usr/bin/claude")
    signed_out = lambda argv, **kw: SimpleNamespace(stdout='{"loggedIn": false}')  # noqa: E731
    with pytest.raises(LLMError, match="sign in"):
        providers.choose_provider("claude-code", runner=signed_out)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    provider, note = providers.choose_provider("claude-code", runner=signed_out)
    assert provider == "api"
    assert note.startswith("Claude Code isn't signed in")


def test_sign_in_check_that_cannot_answer_does_not_block() -> None:
    def old_cli(argv, **kw):
        assert argv[1:] == ["auth", "status"]
        assert "ANTHROPIC_API_KEY" not in kw["env"]
        return SimpleNamespace(stdout="unknown command")

    assert providers.claude_code_signed_in("/usr/bin/claude", old_cli) is None
    assert providers.claude_code_signed_in("/no/such/claude") is None


def test_ant_login_profile_counts_as_api_credentials(no_credentials) -> None:
    assert not providers.has_api_credentials()
    (no_credentials / "credentials").mkdir()
    (no_credentials / "credentials" / "default.json").write_text("{}")
    assert providers.has_api_credentials()


def test_nothing_available_explains_the_options(no_credentials, monkeypatch) -> None:
    monkeypatch.setattr(providers.shutil, "which", lambda name: None)
    with pytest.raises(LLMError, match="ant auth login"):
        providers.resolve_provider("claude-code")


def test_unknown_provider_is_rejected() -> None:
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

    with pytest.raises(LLMError, match="no answer within"):
        ClaudeCodeClient("/usr/bin/claude", runner=slow, timeout_s=1).complete(request())


def test_missing_claude_code(monkeypatch) -> None:
    import hog_sim.llm.claude_code as cc

    monkeypatch.setattr(cc.shutil, "which", lambda name: None)
    with pytest.raises(LLMError, match="not installed"):
        ClaudeCodeClient()


# --- Connection errors -------------------------------------------------------


def api_error(status: int, message: str):
    anthropic = pytest.importorskip("anthropic")
    import httpx2

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    return anthropic.APIStatusError(
        message, response=httpx2.Response(status, request=req), body=body
    )


@pytest.mark.parametrize(
    ("status", "message", "expected"),
    [
        (400, "Your credit balance is too low to access the Anthropic API.", "no credit"),
        (401, "invalid x-api-key", "rejected"),
        (429, "rate limited", "rate limit"),
        (529, "overloaded", "server error"),
    ],
)
def test_api_errors_become_one_line_with_the_other_route(status, message, expected) -> None:
    from hog_sim.llm.client import AnthropicClient, LLMUnavailable

    exc = api_error(status, message)

    def fail(**kwargs):
        raise exc

    sdk = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=fail)))
    with pytest.raises(LLMUnavailable) as caught:
        AnthropicClient(sdk).complete(request())
    text = str(caught.value)
    assert expected in text and "--provider api is set" in text
    assert "\n" not in text


def test_non_api_errors_are_not_swallowed() -> None:
    from hog_sim.llm.client import AnthropicClient

    def fail(**kwargs):
        raise ValueError("bug")

    sdk = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=fail)))
    with pytest.raises(ValueError):
        AnthropicClient(sdk).complete(request())


def test_claude_code_does_not_see_api_keys(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-no-credit")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    seen = {}

    def runner(argv, input, env, **kwargs):
        seen.update(env)
        return SimpleNamespace(stdout=cli_output(structured={}), stderr="", returncode=0)

    ClaudeCodeClient("/usr/bin/claude", runner=runner).complete(request())
    assert "ANTHROPIC_API_KEY" not in seen and "ANTHROPIC_AUTH_TOKEN" not in seen
    assert "PATH" in seen


def test_cli_prints_one_line_instead_of_a_traceback(monkeypatch) -> None:
    from hog_sim.llm.client import LLMUnavailable
    from hog_sim.ui import cli

    class Broken:
        def __init__(self) -> None:
            from hog_sim.llm.client import UsageLog

            self.usage = UsageLog()

        def complete(self, request):
            raise LLMUnavailable("Anthropic API: your Anthropic API account has no credit left.")

    monkeypatch.setattr(cli, "_llm_client", lambda provider: (Broken(), "api", None))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--check-llm"])
    assert str(exc.value).startswith("Could not reach Claude. Anthropic API: ")


def test_cli_says_when_it_fell_back_to_the_api(monkeypatch, capsys) -> None:
    from hog_sim.llm.client import FakeClient
    from hog_sim.ui import cli

    note = "Claude Code isn't installed, so the game is falling back to the Anthropic API."
    client = FakeClient([{"reply": "connected"}])
    monkeypatch.setattr(cli, "_llm_client", lambda provider: (client, "api", note))
    cli.main(["--check-llm"])
    out = capsys.readouterr().out
    assert out.startswith(note) and "via the Anthropic API" in out

    from hog_sim.forecasting.candidates import ForecastConfig
    from hog_sim.llm.client import ModelConfig

    banner = cli._llm_banner(ModelConfig(), ForecastConfig(), "api", note)
    assert banner.splitlines()[0] == note


# --- Codex client --------------------------------------------------------------


def codex_events(text=None, failed=None, errors=()) -> str:
    events = [{"type": "thread.started", "thread_id": "t1"}, {"type": "turn.started"}]
    events += [{"type": "error", "message": m} for m in errors]
    if text is not None:
        events.append(
            {"type": "item.completed", "item": {"id": "i0", "type": "agent_message", "text": text}}
        )
        events.append(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 120, "cached_input_tokens": 100, "output_tokens": 30},
            }
        )
    if failed is not None:
        events.append({"type": "turn.failed", "error": {"message": failed}})
    return "\n".join(json.dumps(e) for e in events)


def test_codex_runs_exec_read_only_with_everything_on_stdin(monkeypatch) -> None:
    from hog_sim.llm.codex import CodexClient

    monkeypatch.delenv("HOG_SIM_CODEX_MODEL", raising=False)
    runner = FakeRunner(codex_events(text='```json\n{"actions": []}\n```'))
    client = CodexClient("/usr/bin/codex", runner=runner)
    response = client.complete(request())
    argv, stdin = runner.calls[0]
    assert argv[:3] == ["/usr/bin/codex", "exec", "--json"]
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in argv and "--skip-git-repo-check" in argv
    assert 'model_reasoning_effort="low"' in argv
    assert "--model" not in argv  # Claude model names mean nothing to Codex
    assert argv[-1] == "-"
    assert stdin.startswith("You map text") and '"properties"' in stdin
    assert stdin.endswith("Tax energy firms")
    assert json.loads(response.text) == {"actions": []}
    record = client.usage.records[0]
    assert record.input_tokens == 120 and record.output_tokens == 30
    assert record.cost_usd == 0  # comes out of the ChatGPT plan
    assert record.served_model == "codex/codex-default"


def test_codex_model_and_effort(monkeypatch) -> None:
    from hog_sim.llm.codex import CodexClient

    monkeypatch.setenv("HOG_SIM_CODEX_MODEL", "gpt-5.5")
    runner = FakeRunner(codex_events(text="{}"))
    client = CodexClient("/usr/bin/codex", runner=runner)
    client.complete(request(effort="max"))
    argv, _ = runner.calls[0]
    assert argv[argv.index("--model") + 1] == "gpt-5.5"
    assert 'model_reasoning_effort="xhigh"' in argv
    assert client.usage.records[0].served_model == "codex/gpt-5.5"


def test_codex_retry_works_through_structured_call() -> None:
    from hog_sim.llm.codex import CodexClient

    runner = FakeRunner(
        codex_events(text='{"actions": [{"kind": "tax"}]}'),
        codex_events(text='{"actions": [], "clarifying_question": "Which firms?"}'),
    )
    result = structured_call(
        CodexClient("/usr/bin/codex", runner=runner),
        output_type=Interpretation,
        system="sys",
        prompt="Tax them",
        model="claude-opus-5-5",
        prompt_version="v1",
    )
    assert result.clarifying_question == "Which firms?"
    assert "[Your earlier reply]" in runner.calls[1][1]


def test_codex_errors_explain_sign_in() -> None:
    from hog_sim.llm.codex import CodexClient

    failed = CodexClient(
        "/usr/bin/codex", runner=FakeRunner(codex_events(errors=["retrying"], failed="401"))
    )
    with pytest.raises(LLMError, match="Codex failed: 401.*codex login"):
        failed.complete(request())
    offline = CodexClient(
        "/usr/bin/codex", runner=FakeRunner(codex_events(errors=["Reconnecting... 5/5"]))
    )
    with pytest.raises(LLMError, match="Reconnecting"):
        offline.complete(request())


def test_codex_does_not_see_openai_keys(monkeypatch) -> None:
    from hog_sim.llm.codex import CodexClient

    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj")
    monkeypatch.setenv("CODEX_API_KEY", "sk-proj")
    seen = {}

    def runner(argv, input, env, **kwargs):
        seen.update(env)
        return SimpleNamespace(stdout=codex_events(text="{}"), stderr="", returncode=0)

    CodexClient("/usr/bin/codex", runner=runner).complete(request())
    assert "OPENAI_API_KEY" not in seen and "CODEX_API_KEY" not in seen


def test_codex_provider_needs_codex_installed_and_signed_in(monkeypatch) -> None:
    monkeypatch.setattr(providers.shutil, "which", lambda name: None)
    with pytest.raises(LLMError, match="npm install -g @openai/codex"):
        providers.choose_provider("codex")

    monkeypatch.setattr(providers.shutil, "which", lambda name: f"/usr/bin/{name}")

    def status(code):
        return lambda argv, **kwargs: SimpleNamespace(stdout="", stderr="", returncode=code)

    with pytest.raises(LLMError, match="codex login"):
        providers.choose_provider("codex", runner=status(1))
    assert providers.choose_provider("codex", runner=status(0)) == ("codex", None)
