"""Precedents: stored turns that resemble what is happening now, as short prompt lines.

Retrieval is deterministic: entries are scored by how many nodes they share with the
current focus (the scenario's affected nodes and the action targets, or failing that the
stressed indicators and angry groups), plus title word overlap. Turns from the current game
come first; other games are capped so Claude is not nudged into replaying them.
"""

from __future__ import annotations

from hog_sim.core.models import PolicyAction, Scenario
from hog_sim.core.state import WorldState
from hog_sim.knowledge.entries import OutcomeEntry, normalise_text
from hog_sim.knowledge.store import KnowledgeStore
from hog_sim.llm.summary import summarise_state


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def precedent_line(entry: OutcomeEntry, game_id: str | None = None) -> str:
    prov = entry.provenance
    where = (
        f"Turn {prov.turn} of this game" if game_id and prov.game_id == game_id else "Another game"
    )
    acts = ", ".join(f"{a.kind} {a.target} {a.magnitude:+.1f}" for a in entry.actions) or "none"
    return (
        f'{where}: "{entry.scenario_title}". Response: {_clip(entry.response, 100)} '
        f"(actions: {acts}). Outcome: {_clip(entry.outcome.narrative, 180)}"
    )


def _words(text: str) -> set[str]:
    return {w for w in normalise_text(text).split() if len(w) > 3}


def recall(
    store: KnowledgeStore,
    state: WorldState,
    scenario: Scenario | None = None,
    actions: list[PolicyAction] | None = None,
    *,
    k: int = 3,
    game_id: str | None = None,
    max_other_games: int = 1,
    skip_recent_turns: int = 0,
) -> list[str]:
    """Up to ``k`` precedent lines for a prompt.

    ``skip_recent_turns`` leaves out this game's last few turns, which the prompt already
    lists as recent events.
    """
    if scenario is not None:
        focus = set(scenario.affected_nodes)
        title_words = _words(scenario.title)
    else:
        summary = summarise_state(state)
        focus = set(summary.stressed_indicators) | set(summary.angry_groups)
        title_words = set()
    focus |= {a.target for a in actions or [] if a.kind != "do_nothing"}
    if not focus and not title_words:
        return []

    scored = []
    for order, entry in enumerate(store.outcomes()):
        prov = entry.provenance
        same_game = bool(game_id) and prov.game_id == game_id
        if same_game and prov.turn is not None and prov.turn >= state.turn - skip_recent_turns:
            continue
        nodes = set(entry.affected_nodes) | {a.target for a in entry.actions}
        score = len(nodes & focus) + 2 * len(_words(entry.scenario_title) & title_words)
        if score:
            scored.append((same_game, score, order, entry))
    # Current game first, then by score, then newest.
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)

    lines, others = [], 0
    for same_game, _, _, entry in scored:
        if not same_game:
            if others >= max_other_games:
                continue
            others += 1
        lines.append(precedent_line(entry, game_id))
        if len(lines) == k:
            break
    return lines


class Recaller:
    """Binds a store to the game in progress, for the LLM plug-ins' ``recall`` hooks.

    ``game_id`` is set once the game exists (the plug-ins are built before it).
    """

    def __init__(self, store: KnowledgeStore, game_id: str | None = None) -> None:
        self.store = store
        self.game_id = game_id

    def for_scenario(self, state: WorldState) -> list[str]:
        # The scenario prompt already lists the last 8 turns as recent events.
        return recall(self.store, state, game_id=self.game_id, skip_recent_turns=8)

    def for_outcomes(
        self, state: WorldState, scenario: Scenario, actions: list[PolicyAction]
    ) -> list[str]:
        return recall(self.store, state, scenario, actions, game_id=self.game_id, max_other_games=3)
