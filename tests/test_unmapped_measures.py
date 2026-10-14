"""Review of 13 Oct 2026, E21: measures the model could not map vanished silently. The CLI
asked the interpreter for ``dropped()``, which none defines, and the outcome prompt never
heard of them."""

from hog_sim.core.config import GameConfig
from hog_sim.core.models import PolicyAction
from hog_sim.game.loop import Game
from hog_sim.game.stubs import CannedScenarios, EngineForecaster, KeywordInterpreter
from hog_sim.ui import cli
from hog_sim.world.seed.toy import toy_world

UNMAPPED = ["ban the opposition from television"]


class Unmapping(KeywordInterpreter):
    def last_unmapped(self) -> list[str]:
        return list(UNMAPPED)


def new_game(forecaster=None) -> Game:
    config = GameConfig(seed=0, election_turn=6, k_draws=5)
    return Game(
        config,
        toy_world(),
        CannedScenarios(0, calendar=False),
        Unmapping(),
        forecaster or EngineForecaster(),
    )


def test_unmapped_measures_show_on_the_review_screen_and_survive_edits() -> None:
    game = new_game()
    proposal = game.propose("Fund a pay rise for nurses and teachers")
    assert proposal.unmapped == UNMAPPED
    assert f"Not possible: {UNMAPPED[0]}" in cli._proposal_text(proposal, game.state)
    edited = game.revise(
        proposal, [PolicyAction(kind="spend", target="sector:public", magnitude=0.5)]
    )
    assert edited.unmapped == UNMAPPED


def test_unmapped_measures_reach_the_outcome_prompt_and_the_record() -> None:
    seen = []

    class Recording(EngineForecaster):
        def forecast(self, *args, limits=None, **kwargs):
            seen.append(limits)
            return super().forecast(*args, limits=limits, **kwargs)

    game = new_game(Recording())
    record = game.play_turn("Fund a pay rise for nurses and teachers")
    assert any(UNMAPPED[0] in note and "not possible" in note for note in seen[0])
    assert record.unmapped == UNMAPPED


def test_the_llm_interpreter_reports_its_unmapped_measures() -> None:
    from hog_sim.llm.adapters import LLMInterpreter
    from hog_sim.llm.interpreter import Interpretation

    interpreter = LLMInterpreter(client=None)  # type: ignore[arg-type]
    assert interpreter.last_unmapped() == []
    interpreter.last_interpretation = Interpretation(unmapped=UNMAPPED)
    assert interpreter.last_unmapped() == UNMAPPED
