"""Advisers: ask for new options on the current issue (backlog story RB-3).

``advise`` makes one Claude call and returns 2-4 options, each with a one-line trade-off.
Asking never advances the turn; the game adds the options to the scenario's numbered list
(``Game.add_options``) so the player can pick and combine them as usual.
"""

from __future__ import annotations

from hog_sim.core.models import Model, Scenario
from hog_sim.core.state import WorldState
from hog_sim.llm.client import LLMClient, ModelConfig, structured_call
from hog_sim.llm.prompts import advise as prompt
from hog_sim.llm.scenario_gen import scenario_text
from hog_sim.llm.summary import StateSummary, summarise_state
from hog_sim.policy.feasibility import Role


class AdvisedOption(Model):
    option: str
    trade_off: str


class Advice(Model):
    options: list[AdvisedOption]


def check_advice(advice: Advice, existing: list[str]) -> list[str]:
    problems = []
    if not 2 <= len(advice.options) <= 4:
        problems.append("give 2-4 options")
    seen = {o.strip().lower() for o in existing}
    if any(o.option.strip().lower() in seen for o in advice.options):
        problems.append("options must be new, not repeats of the options on the table")
    if any(not o.trade_off.strip() for o in advice.options):
        problems.append("every option needs a trade_off")
    return problems


def advise(
    question: str,
    summary: StateSummary,
    scenario: Scenario,
    client: LLMClient,
    config: ModelConfig | None = None,
) -> list[AdvisedOption]:
    config = config or ModelConfig()
    return structured_call(
        client,
        output_type=Advice,
        system=prompt.SYSTEM,
        prompt=prompt.render(summary.to_prompt(), scenario_text(scenario), question),
        model=config.scenario_model,
        prompt_version=prompt.VERSION,
        effort=config.interpret_effort,
        max_tokens=config.max_tokens,
        max_attempts=config.max_attempts,
        check=lambda a: check_advice(a, scenario.suggested_options),
    ).options


class Advisers:
    def __init__(
        self, client: LLMClient, role: Role = "prime_minister", config: ModelConfig | None = None
    ) -> None:
        self.client = client
        self.role = role
        self.config = config

    def ask(self, question: str, state: WorldState, scenario: Scenario) -> list[AdvisedOption]:
        summary = summarise_state(state, self.role)
        return advise(question, summary, scenario, self.client, self.config)
