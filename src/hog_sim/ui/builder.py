"""Response builder for the text interface (backlog stories RB-1 and RB-2).

``compose_response`` turns numbered option picks plus free text into one response for the
interpreter: ``"1 3 + also freeze rail fares"`` becomes both options and the extra words.

``edit_actions`` applies one review command to the interpreted actions before the turn is
committed:

* ``drop 2``: remove action 2
* ``2 size 0.3``: set action 2's magnitude (−1 to 1)
* ``2 turns 4``: set how many turns action 2 runs
* ``2 kind tax``: change what action 2 does
* ``2 target housing``: aim action 2 somewhere else (a node id, or a word that names one)

``build_actions`` is the offline policy builder (review O7): a menu that walks through kind,
target, size and turns for each measure, so offline play does not depend on keyword
matching. Free text still works; its reading is a starting package the player can edit.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import get_args

from hog_sim.core.models import ActionKind, PolicyAction
from hog_sim.core.state import WorldState
from hog_sim.policy.feasibility import compatibility_problem

KINDS: list[str] = [k for k in get_args(ActionKind) if k != "do_nothing"]

KIND_HELP = {
    "tax": "raise (+) or cut (-) a tax",
    "spend": "spend more (+) or cut spending (-)",
    "regulate": "tighten rules",
    "deregulate": "loosen rules",
    "legislate": "pass a law",
    "communicate": "a speech or campaign",
    "diplomatic": "court (+) or confront (-) a foreign country",
    "military": "military action",
    "appoint": "appoint someone to an institution",
}

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


def edit_actions(
    actions: list[PolicyAction], command: str, nodes: Sequence[str] = ()
) -> list[PolicyAction]:
    """Return a new action list with ``command`` applied. Raises ``ValueError`` if invalid.

    ``nodes`` are the world's node ids, for ``target``."""
    words = command.lower().split()
    if len(words) == 2 and words[0] == "drop":
        index = _index(words[1], actions)
        return actions[:index] + actions[index + 1 :]
    if len(words) >= 3 and words[1] in ("kind", "target"):
        index = _index(words[0], actions)
        value = " ".join(words[2:])
        if words[1] == "kind":
            if value not in KINDS:
                raise ValueError(f"kind must be one of: {', '.join(KINDS)}")
            update = {"kind": value}
        else:
            update = {"target": find_node(value, nodes)}
        edited = actions[index].model_copy(update=update)
        return actions[:index] + [edited] + actions[index + 1 :]
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
    raise ValueError("try: drop 2 · 2 size 0.3 · 2 turns 4 · 2 kind tax · 2 target housing")


def _index(word: str, actions: list[PolicyAction]) -> int:
    if not word.isdigit() or not 1 <= int(word) <= len(actions):
        raise ValueError(f"pick an action from 1 to {len(actions)}")
    return int(word) - 1


def find_node(text: str, nodes: Sequence[str]) -> str:
    """The node ``text`` names: its id, or the one id whose name part contains it
    ("housing" is ``sector:housing``). Raises ``ValueError`` when none or several match."""
    text = text.strip().lower()
    if text in nodes:
        return text
    name = text.replace(" ", "_")
    exact = [n for n in nodes if n.split(":", 1)[-1] == name]
    if len(exact) == 1:
        return exact[0]
    matches = [n for n in nodes if name in n.split(":", 1)[-1]]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"no target matches {text!r}")
    raise ValueError(f"{text!r} could be {', '.join(matches)}")


def targets_for(kind: str, state: WorldState) -> list[str]:
    """The nodes a ``kind`` of action can be aimed at in ``state``."""
    return [
        n
        for n in state.node_ids()
        if compatibility_problem(PolicyAction(kind=kind, target=n, magnitude=0.5), state) is None
    ]


def _label(state: WorldState, node: str) -> str:
    return f"{state.node(node).name} ({node})"


def _choose(
    title: str, choices: list[str], labels: list[str], ask: Callable[[str], str], say
) -> str | None:
    """A numbered menu; returns the choice, or None when the player presses Enter."""
    say(title)
    for i, label in enumerate(labels, 1):
        say(f"  {i}. {label}")
    while True:
        answer = ask("> ").strip().lower()
        if not answer:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        try:
            return find_node(answer, choices) if ":" in choices[0] else _word(answer, choices)
        except ValueError as err:
            say(f"  {err}")


def _word(answer: str, choices: list[str]) -> str:
    if answer in choices:
        return answer
    raise ValueError(f"pick 1 to {len(choices)}")


def _number(
    prompt: str, default: float, low: float, high: float, ask, say, whole: bool = False
) -> float:
    while True:
        answer = ask(prompt).strip()
        if not answer:
            return default
        try:
            value = float(answer)
        except ValueError:
            say(f"  {answer!r} is not a number")
            continue
        if not low <= value <= high or (whole and value != int(value)):
            kind = "a whole number" if whole else "a number"
            say(f"  enter {kind} from {low:g} to {high:g}")
            continue
        return value


def build_actions(
    state: WorldState,
    ask: Callable[[str], str] | None = None,
    say: Callable[[str], None] | None = None,
) -> list[PolicyAction]:
    """Walk the player through kind, target, size and turns for each measure (O7).

    Pressing Enter at the kind menu finishes; at the target menu it abandons that measure.
    Only targets the kind can be aimed at are offered. ``ask`` and ``say`` default to
    ``input`` and ``print``."""
    ask = ask or input
    say = say or print
    actions: list[PolicyAction] = []
    while True:
        heading = "Add a measure" if not actions else "Add another measure"
        kind = _choose(
            f"\n{heading} (Enter when done). What kind?",
            KINDS,
            [f"{k}: {KIND_HELP[k]}" for k in KINDS],
            ask,
            say,
        )
        if kind is None:
            return actions
        targets = targets_for(kind, state)
        if not targets:
            say(f"  Nothing in this world can take {kind}.")
            continue
        target = _choose(
            f"Aimed at? ({kind})", targets, [_label(state, n) for n in targets], ask, say
        )
        if target is None:
            continue
        hint = (
            "the way you want it to move" if target.startswith("indicator:") else "+ more, - less"
        )
        size = _number(f"Size from -1 to 1 ({hint}) [0.5]: ", 0.5, -1, 1, ask, say)
        turns = _number("Turns it runs [1]: ", 1, 1, 120, ask, say, whole=True)
        action = PolicyAction(
            kind=kind,  # type: ignore[arg-type]
            target=target,
            magnitude=size,
            duration_turns=int(turns),
            rationale="built",
        )
        actions.append(action)
        say(f"  Added: {describe(action)}")
