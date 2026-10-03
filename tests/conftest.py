import pytest


@pytest.fixture(autouse=True)
def no_llm_log(monkeypatch):
    """Tests that run the CLI with the default save path must not write a call log into the
    repository; tests of the log pass their own path or clear this."""
    monkeypatch.setenv("HOG_SIM_LLM_LOG", "off")
