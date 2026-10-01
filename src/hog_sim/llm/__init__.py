"""LLM layer: scenarios in, player intent out, always as validated structured data."""

from hog_sim.llm.client import (
    AnthropicClient,
    FakeClient,
    LLMClient,
    LLMError,
    LLMOutputError,
    LLMRefusal,
    ModelConfig,
    RecordingClient,
    structured_call,
)
from hog_sim.llm.interpreter import Interpretation, interpret
from hog_sim.llm.scenario_gen import GeneratedScenario, StakeholderPosition, generate_scenario
from hog_sim.llm.summary import StateSummary, summarise_state

__all__ = [
    "AnthropicClient",
    "FakeClient",
    "GeneratedScenario",
    "Interpretation",
    "LLMClient",
    "LLMError",
    "LLMOutputError",
    "LLMRefusal",
    "ModelConfig",
    "RecordingClient",
    "StakeholderPosition",
    "StateSummary",
    "generate_scenario",
    "interpret",
    "structured_call",
    "summarise_state",
]
