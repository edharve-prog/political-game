"""Wire the LLM layer (Projects 5 and 6) into the game loop's plug-in slots."""

from __future__ import annotations

from hog_sim.core.models import PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.forecasting.candidates import ForecastConfig, LLMForecaster
from hog_sim.game.interfaces import NeedsClarification
from hog_sim.knowledge.recall import Recaller
from hog_sim.llm.adapters import LLMInterpreter, LLMScenarioSource
from hog_sim.llm.client import LLMClient, ModelConfig
from hog_sim.policy.feasibility import Role


class ClarifyingInterpreter(LLMInterpreter):
    """LLMInterpreter that asks for clarification instead of playing an empty turn."""

    def interpret(self, text: str, state: WorldState, scenario: Scenario) -> list[PolicyAction]:
        actions = super().interpret(text, state, scenario)
        result = self.last_interpretation
        if result is not None and not result.actions and result.clarifying_question:
            raise NeedsClarification(result.clarifying_question)
        return actions

    def dropped(self) -> list[str]:
        """One line per action feasibility blocked on the last call."""
        report = self.last_feasibility
        if report is None:
            return []
        return [
            f"{c.action.kind} {c.action.target}: {'; '.join(c.blockers)}"
            for c in report.checks
            if not c.feasible
        ]


def llm_plugins(
    client: LLMClient,
    role: Role = "prime_minister",
    model_config: ModelConfig | None = None,
    forecast_config: ForecastConfig | None = None,
    recaller: Recaller | None = None,
) -> tuple[LLMScenarioSource, ClarifyingInterpreter, LLMForecaster]:
    """``recaller`` (optional) adds precedents from the knowledge store to the scenario and
    outcome prompts."""
    return (
        LLMScenarioSource(
            client, role, model_config, recall=recaller.for_scenario if recaller else None
        ),
        ClarifyingInterpreter(client, role, model_config),
        LLMForecaster(
            client, role, forecast_config, recall=recaller.for_outcomes if recaller else None
        ),
    )
