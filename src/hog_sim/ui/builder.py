"""Response builder for the text interface (backlog stories RB-1 and RB-2).

``compose_response`` turns numbered option picks plus free text into one response for the
interpreter: ``"1 3 + also freeze rail fares"`` becomes both options and the extra words.

``edit_actions`` applies one review command to the interpreted actions before the turn is
committed:

* ``drop 2``: remove action 2
* ``2 size 0.3``: set action 2's magnitude (−1 to 1)
* ``2 turns 4``: set how many turns action 2 runs
"""

from __future__ import annotations

import re

from hog_sim.core.models import PolicyAction

_PICKS = re.compile(r"^\s*(\d+(?:\s*[, ]\s*\d+)*)\s*(?:\+\s*(.*))?$", re.S)


def compose_response(text: str, options: list[str]) -> str:
    """Expand numbered picks of ``options``; anything else passes through unchanged.

    Raises ``ValueError`` for a pick that isn't on the list.
    """
    match = _PICKS.match(text)
    if not match:
        return text
    picks = [int(n) for n in re.findall(r"\d+", match.group(1))]
    bad = [n for n in picks if not 1 <= n <= len(options)]
    if bad:
        raise ValueError(f"there is no option {bad[0]}; pick 1 to {len(options)}")
    chosen = list(dict.fromkeys(options[n - 1] for n in picks))
    extra = (match.group(2) or "").strip()
    parts = [f"Option: {o}" for o in chosen]
    if extra:
        parts.append(f"Also: {extra}")
    return "\n".join(parts)


def describe(action: PolicyAction) -> str:
    turns = f" for {action.duration_turns} turns" if action.duration_turns > 1 else ""
    return f"{action.kind} {action.target} size {action.magnitude:+.2f}{turns}"


def edit_actions(actions: list[PolicyAction], command: str) -> list[PolicyAction]:
    """Return a new action list with ``command`` applied. Raises ``ValueError`` if invalid."""
    words = command.lower().split()
    if len(words) == 2 and words[0] == "drop":
        index = _index(words[1], actions)
        return actions[:index] + actions[index + 1 :]
    if len(words) == 3 and words[1] in ("size", "turns"):
        index = _index(words[0], actions)
        try:
            value = float(words[2])
        except ValueError:
            raise ValueError(f"{words[2]!r} is not a number") from None
        if words[1] == "size":
            if not -1 <= value <= 1:
                raise ValueError("size must be between -1 and 1")
            update = {"magnitude": value}
        else:
            if value < 1 or value != int(value):
                raise ValueError("turns must be a whole number of at least 1")
            update = {"duration_turns": int(value)}
        edited = actions[index].model_copy(update=update)
        return actions[:index] + [edited] + actions[index + 1 :]
    raise ValueError("try: drop 2 · 2 size 0.3 · 2 turns 4")


def _index(word: str, actions: list[PolicyAction]) -> int:
    if not word.isdigit() or not 1 <= int(word) <= len(actions):
        raise ValueError(f"pick an action from 1 to {len(actions)}")
    return int(word) - 1
