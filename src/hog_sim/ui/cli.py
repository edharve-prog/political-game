"""Minimal text interface: ``uv run hog-sim`` (add ``--resume`` to continue the last save).

The game plays with Claude whenever it can reach it, and drops to offline practice (canned
scenarios, keyword matching) only when it can't, or when asked to with ``--offline``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import textwrap
from functools import partial
from pathlib import Path

from hog_sim.content.library import ScenarioLibrary
from hog_sim.core.config import GameConfig
from hog_sim.core.models import PolicyAction, Reading, SideIssue
from hog_sim.core.state import WorldState
from hog_sim.game.advice import CostedOption, cost_option, costing_line
from hog_sim.game.explain import alternatives, why
from hog_sim.game.interfaces import NeedsClarification
from hog_sim.game.loop import Game, Proposal, StaleProposal
from hog_sim.game.packages import PolicyLibrary
from hog_sim.game.persistence import SaveStore
from hog_sim.game.preview import preview, preview_lines
from hog_sim.game.records import TurnRecord
from hog_sim.game.stubs import EngineForecaster, KeywordInterpreter
from hog_sim.knowledge.offline import StoredForecaster, StoredInterpreter
from hog_sim.knowledge.recall import Recaller
from hog_sim.knowledge.store import KnowledgeStore
from hog_sim.llm.advisers import Advisers
from hog_sim.policy.fiscal_events import scorecard_lines
from hog_sim.policy.limits import lifetime_cost
from hog_sim.policy.pledges import broken_by, pledge_line
from hog_sim.population.popularity import ElectionResult, national_approval, vote_intention
from hog_sim.ui.builder import build_actions, compose_response, describe, edit_actions
from hog_sim.world.calendar import OnCalendar, upcoming
from hog_sim.world.cast import active_cast
from hog_sim.world.opposition import opposition_line
from hog_sim.world.propagation import world_surprises
from hog_sim.world.seed.toy import toy_world
from hog_sim.world.storylines import addressed, harmed


def _width() -> int:
    """Line width for wrapped text: the terminal's, kept between 60 and 100 columns."""
    return max(60, min(shutil.get_terminal_size((80, 24)).columns, 100) - 1)


def _banner(title: str) -> str:
    """A heading that starts a new part of the turn, framed in plain ASCII rules."""
    rule = "=" * _width()
    return f"\n{rule}\n  {title}\n{rule}"


def _section(title: str) -> str:
    """A smaller heading inside a part of the turn."""
    return f"\n{title}\n{'-' * len(title)}"


def _wrap(text: str, indent: str = "", hanging: str | None = None) -> str:
    """Wrap ``text`` to the screen; ``hanging`` indents the lines after the first."""
    return textwrap.fill(
        text,
        width=_width(),
        initial_indent=indent,
        subsequent_indent=indent if hanging is None else hanging,
        break_on_hyphens=False,
    )


def _paragraphs(text: str, indent: str = "") -> str:
    """Wrap each paragraph of ``text`` on its own, with a blank line between them."""
    parts = [p for p in text.replace("\r", "").split("\n") if p.strip()]
    return "\n\n".join(_wrap(p.strip(), indent) for p in parts)


def _numbered(i: int, text: str, indent: str = "  ") -> str:
    """``  1. text`` with continuation lines lined up under the text."""
    label = f"{indent}{i}. "
    return _wrap(text, label, " " * len(label))


def _dashboard(
    start: WorldState,
    state: WorldState,
    previous: WorldState | None = None,
    election_turn: int | None = None,
) -> str:
    """Current scores, with the change since the last turn and since the game began."""
    previous = previous or start
    first = state.turn == 0
    columns = "" if first else f"{'last turn':>11}{'since start':>13}"
    title = f"TURN {state.turn}"
    if election_turn is not None and state.turn < election_turn:
        title += f"      (election at turn {election_turn})"
    lines = [_banner(title), ""]

    def row(label: str, now: str, last: str, total: str) -> str:
        changes = "" if first else f"{last:>11}{total:>13}"
        return f"  {label:<27}{now:<15}{changes}".rstrip()

    lines.append(f"{'ECONOMY':<29}{'now':<15}{columns}".rstrip())
    for ind in state.indicators.values():
        last = ind.value - previous.indicators[ind.id].value
        total = ind.value - start.indicators[ind.id].value
        now = f"{ind.value:.2f} {ind.unit}"
        lines.append(row(ind.name, now, f"{last:+.2f}", f"{total:+.2f}"))
    lines += ["", f"{'APPROVAL':<29}{'now':<15}{columns}".rstrip()]
    scores = [
        (g.name, g.approval, previous.groups[g.id].approval, start.groups[g.id].approval)
        for g in state.groups.values()
    ]
    scores += [
        (label, score(state), score(previous), score(start))
        for label, score in (("National", national_approval), ("Vote intention", vote_intention))
    ]
    for label, now, before, begin in scores:
        lines.append(
            row(
                label,
                f"{now * 100:.1f}%",
                f"{(now - before) * 100:+.1f}",
                f"{(now - begin) * 100:+.1f}",
            )
        )
    lines += ["", f"{'STANDING':<29}{'now':<15}{columns}".rstrip()]
    if state.capital is not None:
        # Capital is a stock (O5): what this turn may spend, refilled and partly banked.
        last = "" if previous.capital is None else f"{state.capital - previous.capital:+.2f}"
        lines.append(row("Political capital", f"{state.capital:.2f}", last, ""))
    lines.append(
        row(
            "Credibility",
            f"{state.credibility * 100:.0f}%",
            f"{(state.credibility - previous.credibility) * 100:+.0f}",
            f"{(state.credibility - start.credibility) * 100:+.0f}",
        )
    )
    cast = active_cast(state)
    if cast:
        lines += ["", f"{'PEOPLE':<24}{'loyalty':<12}role"]
        for c in cast:
            before = previous.characters.get(c.id)
            change = (
                ""
                if first
                else f"({(c.loyalty - before.loyalty) * 100:+.0f})"
                if before
                else "(new)"
            )
            loyalty = f"{c.loyalty * 100:>3.0f}% {change}"
            lines.append(_wrap(c.role, f"  {c.name:<22}{loyalty:<12}", " " * 36))
    lines += ["", "OPPOSITION", _wrap(opposition_line(state), "  ", "    ")]
    coming = upcoming(state.turn)
    if coming:
        lines += ["", "COMING UP"]
        lines += [f"  Turn {t}: {e.title}" for t, e in coming]
    if state.pledges:
        lines += ["", "PLEDGES"]
        lines += [_wrap(pledge_line(p), "  ", "    ") for p in state.pledges]
    return "\n".join(lines)


def _previous(game: Game) -> WorldState:
    """The state before the last turn played, so the dashboard can show what it changed."""
    return game.history[-2].state_after if len(game.history) >= 2 else game.start


def _report(record: TurnRecord) -> str:
    lines = [_section("What happened"), ""]
    lines += [_wrap(f"Note: {n}", "", "      ") for n in record.notes]
    if record.notes:
        lines.append("")
    lines.append(_paragraphs(record.outcome.narrative))
    if record.actions:
        lines += ["", "Actions taken:"]
        lines += [f"  - {a.kind} {a.target} {a.magnitude:+.2f}" for a in record.actions]
    if record.election:
        e = record.election
        lines += ["", _election_line(e)]
    return "\n".join(lines)


def _election_line(e: ElectionResult) -> str:
    return (
        f"ELECTION: {e.vote_share * 100:.1f}% of the vote, "
        f"{e.seats}/{e.total_seats} seats. {e.verdict}"
    )


def _surprises(state: WorldState, seed: int) -> str:
    """After the election: where this game's economy differed most from the forecast's
    central guess (D1)."""
    lines = [_section("How this economy differed from your advisers' guess"), ""]
    for edge, truth in world_surprises(state, seed):
        source, target = state.node(edge.source).name, state.node(edge.target).name
        lines.append(f"  {source} -> {target}: guessed {edge.weight:+.2f}, really {truth:+.2f}")
    if len(lines) == 2:
        lines.append("  Nowhere: this economy behaved exactly as guessed.")
    return "\n".join(lines)


def _finished(game: Game, save: str) -> str:
    """What to show when ``--resume`` opens a game whose election has already been held."""
    election = game.history[-1].election if game.history else None
    lines = ["This game is over."]
    if election is not None:
        lines.append(_election_line(election))
    if game.config.hidden_world:
        lines.append(_surprises(game.state, game.config.seed))
    lines.append(f"Start a new game with: hog-sim --save {save}")
    return "\n".join(lines)


REVIEW_PROMPT = (
    "\nPress Enter to confirm, or edit:\n"
    "  drop 2  |  2 size 0.3  |  2 turns 4  |  2 kind tax  |  2 target housing  |  add\n"
    "  save <name>  |  redo\n"
    "> "
)


def _lifetime(action: PolicyAction, state: WorldState | None) -> str:
    """What a measure costs over its whole life, for the review screen (O1)."""
    if state is None or action.kind == "do_nothing":
        return ""
    cost = lifetime_cost(action, state)
    deficit = (
        f", deficit {cost.deficit:+.2f} pts of GDP for {cost.deficit_turns} turns"
        if cost.deficit
        else ""
    )
    return f"  [over its life: {cost.capital:.2f} capital{deficit}]"


def _proposal_text(proposal: Proposal, state: WorldState | None = None) -> str:
    lines = [_section("Your advisers read that as"), ""]
    if proposal.actions:
        for i, a in enumerate(proposal.actions, 1):
            lines.append(f"  {i}. {describe(a)}{_lifetime(a, state)}")
            lines += _reading_lines(proposal.reading(i - 1))
    else:
        lines.append("  (no actions: this turn the government does nothing)")
    notes = [f"Not possible: {line}" for line in proposal.unmapped]
    notes += [f"Note: {n}" for n in proposal.notes]
    notes.append(f"Delivery: {proposal.delivery.describe()}")
    notes += [f'New pledge: "{p.text}"' for p in proposal.pledges]
    if state is not None:
        notes += [
            f"Sacking: {state.characters[c].name}, {state.characters[c].role} "
            "(type drop sacking to keep them)"
            for c in proposal.sacked
        ]
        notes += [
            f'Warning: this breaks your pledge "{p.text}" (made turn {p.made_turn})'
            for p in broken_by(state, proposal.actions)
        ]
    lines.append("")
    lines += [_wrap(n, "  ", "    ") for n in notes]
    card = scorecard_lines(proposal.actions, state, describe) if state is not None else []
    if card:
        lines += ["", *(_wrap(line, "  ", "      ") for line in card)]
    return "\n".join(lines)


def _reading_lines(reading: Reading | None) -> list[str]:
    """Where an action came from in the response, flagged when the interpreter guessed (D6)."""
    if reading is None:
        return []
    lines = []
    if reading.span:
        span = reading.span if len(reading.span) <= 70 else reading.span[:67] + "..."
        lines.append(f'     from "{span}"')
    if reading.fallback:
        lines.append("     CHECK: no target was named, so it was guessed from the scenario")
    elif reading.doubtful:
        lines.append(f"     CHECK: low confidence ({reading.confidence:.0%}) in this reading")
    return lines


def _preview_text(game: Game, proposal: Proposal) -> str:
    """The decision preview (D2): this package against doing nothing, to the election."""
    result = preview(
        game.start,
        game.state,
        game.scenario,
        proposal.actions,
        game.config,
        proposal.pledges,
        proposal.delivery,
        proposal.sacked,
        game.config.role,
    )
    lines = [_section("Looking ahead to the election"), ""]
    lines += preview_lines(result, describe)
    return "\n".join(lines)


MODEL_FAILED_PROMPT = (
    "Type r to try the model again, or o to use the offline reading for this turn.\n> "
)


def _retry_or_offline(what: str, err: Exception) -> str:
    """After a model failure, ask whether to try again (``"retry"``) or play this step
    offline (``"offline"``). Ctrl-D still saves and quits."""
    print("\n" + _wrap(f"The model failed while {what}: {err}", "", "  "))
    while True:
        choice = input(MODEL_FAILED_PROMPT).strip().lower()
        if choice in ("r", "retry", ""):
            return "retry"
        if choice in ("o", "offline"):
            return "offline"


def _guarded(game: Game, part: str, what: str, call, offline=None, ran_offline=None):
    """``call()``, and when the model fails (E22) ask to retry or to run it once with
    ``game.<part>`` (scenarios, interpreter or forecaster) swapped for ``offline``. Without
    an offline stand-in the error is raised as before. A step run offline adds ``part`` to
    the ``ran_offline`` set before it runs, so its result can be marked as not Claude's."""
    from hog_sim.llm.budget import LLMBudgetExceeded
    from hog_sim.llm.client import LLMError

    while True:
        try:
            return call()
        except LLMError as err:
            if offline is None:
                raise
            choice = _retry_or_offline(what, err)
            if choice == "retry" and isinstance(err, LLMBudgetExceeded):
                err.client.extend()
            if choice == "offline":
                if ran_offline is not None:
                    ran_offline.add(part)
                original = getattr(game, part)
                setattr(game, part, offline)
                try:
                    return call()
                finally:
                    setattr(game, part, original)


def _ask_for_turn(
    game: Game,
    advisers: Advisers | None = None,
    library: PolicyLibrary | None = None,
    offline: tuple | None = None,
    pending: Proposal | None = None,
) -> Proposal | None:
    """Build, review and confirm this turn's actions. ``None`` means start the response again.

    ``advise <question>`` asks the advisers for more options instead (Claude mode only).
    With a ``library``, ``use <name>`` loads a saved package, ``packages`` lists them and
    ``save <name>`` at the review step saves the actions (story RB-5). ``offline`` are the
    offline plugins to fall back on when the model fails (E22); ``pending`` is a proposal
    confirmed before the game stopped, offered again for review."""
    proposal = pending
    while proposal is None:
        raw = input(RESPONSE_PROMPT)
        words = raw.strip().lower().split()
        if words[:1] == ["advise"]:
            print(_advice(game, advisers, raw.strip()[len("advise") :].strip()))
            continue
        if raw.strip().lower() in EXPLAIN:
            print(_explain(game, raw.strip().lower()))
            continue
        if library is not None and words == ["packages"]:
            print(library.listing())
            continue
        if words == ["build"]:
            built = build_actions(game.state)
            if not built:
                continue
            proposal = game.propose_actions(
                "(built) " + "; ".join(describe(a) for a in built), built
            )
            break
        if library is not None and len(words) == 2 and words[0] == "use":
            try:
                proposal = game.propose_actions(f"(package {words[1]})", library.get(words[1]))
                break
            except ValueError as err:
                print(f"  {err}")
                continue
        options = game.scenario.suggested_options
        try:
            response = compose_response(raw, options)
            ran_offline: set[str] = set()
            proposal = _guarded(
                game,
                "interpreter",
                "reading your response",
                partial(game.propose, response),
                offline[1] if offline else None,
                ran_offline,
            )
            if ran_offline:  # read by the offline stand-in, not by Claude (O10)
                proposal = proposal.model_copy(update={"offline": sorted(ran_offline)})
            break
        except ValueError as err:
            print(f"  {err}")
        except NeedsClarification as ask:
            print("\n" + _wrap(f"Your advisers ask: {ask.question}", "", "  "))
    while True:
        print(_proposal_text(proposal, game.state))
        print(_preview_text(game, proposal))
        command = input(REVIEW_PROMPT).strip()
        if command.lower() in ("", "y", "yes", "ok"):
            return proposal
        if command.lower() == "redo":
            return None
        if command.lower() == "drop sacking" and proposal.sacked:
            proposal = game.revise(proposal, proposal.submitted, sacked=[])
            continue
        parts = command.split()
        if library is not None and len(parts) == 2 and parts[0].lower() == "save":
            try:
                name = library.save(parts[1], proposal.actions, game.state.turn)
                print(f"  Saved as {name}. Load it on a later turn with: use {name}")
            except ValueError as err:
                print(f"  {err}")
            continue
        if command.lower() == "add":
            proposal = game.revise(proposal, proposal.actions + build_actions(game.state))
            continue
        try:
            nodes = list(game.state.node_ids())
            proposal = game.revise(proposal, edit_actions(proposal.actions, command, nodes))
        except ValueError as err:
            print(f"  {err}")


EXPLAIN_HINT = "\n(Type why or alternatives at the next prompt to see how that came about.)"

RESPONSE_PROMPT = (
    "\nYour response: option numbers, your own words, or both (e.g. 1 3 + freeze fares).\n"
    "Other commands: build (pick each measure from menus)  |  advise <question>  |  "
    "use <package>  |  packages  |  why  |  alternatives\n"
    "> "
)


EXPLAIN = ("why", "alternatives")


def _explain(game: Game, command: str) -> str:
    """``why`` and ``alternatives`` for the last turn played (story TT-2)."""
    if not game.history:
        return "  Nothing has happened yet: play a turn first."
    record = game.history[-1]
    if command == "alternatives":
        return "\n".join(alternatives(record))
    return "\n".join(why(record, _previous(game), game.start, game.config))


def _advice(game: Game, advisers: Advisers | None, question: str) -> str:
    """Ask the advisers and add their options to the numbered list."""
    if advisers is None:
        return "  Advisers need Claude; this game is offline practice."
    if not question:
        return "  Ask them something, for example: advise something cheaper"
    from hog_sim.llm.client import LLMError

    try:
        advice = advisers.ask(question, game.state, game.scenario)
    except LLMError as err:
        return f"  The advisers couldn't answer: {err}"
    # Each option is read and played to the election before it is shown (D8). Reading an
    # option with Claude costs a call, so this may take a moment. If the model fails or the
    # turn's call budget runs out (O10), the options not yet costed are offered without
    # numbers and the turn carries on where it was.
    costed: list[CostedOption] = []
    failure: LLMError | None = None
    for a in advice:
        try:
            costed.append(cost_option(game, a.option, a.trade_off))
        except LLMError as err:
            failure = err
            break
    uncosted = [CostedOption(option=a.option, trade_off=a.trade_off) for a in advice[len(costed) :]]
    offered = [c for c in costed if c.blocked is None]
    start = len(game.scenario.suggested_options) + 1
    game.add_options([c.option for c in offered + uncosted])
    lines = [_section("Your advisers suggest")]
    for k, c in enumerate(offered + uncosted):
        i = start + k
        lines += ["", _numbered(i, c.option), _wrap(f"Trade-off: {c.trade_off}", "     ")]
        numbers = (
            costing_line(c.costing)
            if c.costing is not None
            else "Numbers: not costed, because the model failed"
            if k >= len(offered)
            else "Numbers: could not be read into measures, so not costed"
        )
        lines.append(_wrap(numbers, "     ", "     "))
    for c in costed:
        if c.blocked is not None:
            lines += ["", _wrap(f"Not offered: {c.option} ({c.blocked})", "  ", "    ")]
    if failure is not None:
        lines += ["", _wrap(f"The model failed while costing these options: {failure}", "  ")]
    return "\n".join(lines)


OFFLINE_BANNER = (
    "Mode: OFFLINE PRACTICE. Scenarios come from a built-in library and Claude is not used.\n"
    "Typed responses are read by keyword matching, which only suggests a package: check it\n"
    "and edit it at the review step, or type build to pick each measure from menus."
)

NO_CLAUDE_NOTE = "Could not reach Claude, so this game is offline practice."


def _llm_client(provider: str):
    """A client for ``provider``, or exit with a plain explanation of what is missing."""
    from hog_sim.llm.client import LLMError
    from hog_sim.llm.providers import make_client

    try:
        return make_client(provider)
    except LLMError as exc:
        raise SystemExit(str(exc)) from None


def _llm_banner(model_config, forecast_config, provider: str, note: str | None = None) -> str:
    from hog_sim.llm.providers import describe

    if provider == "codex":
        from hog_sim.llm.codex import DEFAULT_MODEL, MODEL_ENV

        return (
            (f"{note}\n" if note else "")
            + f"Mode: CODEX ({os.environ.get(MODEL_ENV) or DEFAULT_MODEL}) via "
            f"{describe(provider)}.\n"
            "Scenarios, your responses and outcomes are written by Codex. Each turn ends with\n"
            "a line counting the calls it made."
        )
    models = sorted(
        {
            model_config.scenario_model,
            model_config.interpret_model,
            forecast_config.outcome_model,
            forecast_config.judge_model,
        }
    )
    return (
        (f"{note}\n" if note else "")
        + f"Mode: CLAUDE ({', '.join(models)}) via {describe(provider)}.\n"
        "Scenarios, your responses and outcomes are written by Claude. Each turn ends with\n"
        "a line counting the calls it made."
    )


def check_llm(client) -> str:
    """Make one tiny call and say which model answered. Raises if the API is unreachable."""
    from hog_sim.core.models import Model
    from hog_sim.llm.client import structured_call

    class Ping(Model):
        reply: str

    result = structured_call(
        client,
        output_type=Ping,
        system="You are a connectivity check.",
        prompt="Reply with the single word: connected",
        model="claude-opus-5-5",
        prompt_version="ping-1",
        effort="low",
        max_tokens=2000,
        max_attempts=1,
    )
    record = client.usage.records[-1]
    served = record.served_model or record.model
    return (
        f"{getattr(client, 'name', 'Claude')} is connected. {served} replied {result.reply!r} "
        f"({record.input_tokens} tokens in, {record.output_tokens} out, "
        f"{record.latency_s:.1f}s)."
    )


def _usage_line(client, since: int) -> str:
    calls = client.usage.records[since:]
    served = sorted({r.served_model or r.model for r in calls})
    tokens_in = sum(r.input_tokens for r in calls)
    tokens_out = sum(r.output_tokens for r in calls)
    name = getattr(client, "name", "Claude")
    plan = "ChatGPT" if name == "Codex" else "Claude"
    spend = (
        f"counts against your {plan} plan"
        if getattr(client, "subscription", False)
        else f"${client.usage.total_cost_usd:.2f} so far"
    )
    return (
        f"  [{name}: {len(calls)} calls this turn via {', '.join(served) or 'none'}, "
        f"{tokens_in} tokens in, {tokens_out} out; {spend}]"
    )


def _provenance(client, since: int):
    """Which models and prompt versions wrote this turn, from the client's usage log."""
    from hog_sim.knowledge.entries import Provenance

    calls = client.usage.records[since:]
    models = sorted({r.served_model or r.model for r in calls})
    versions = sorted({r.prompt_version for r in calls})
    return Provenance(model=", ".join(models) or None, prompt_version=", ".join(versions) or None)


def packages_main(argv: list[str]) -> None:
    """``hog-sim packages list|export|import``: a game's saved policy packages (story RB-5)."""
    parser = argparse.ArgumentParser(
        prog="hog-sim packages", description="Saved policy packages, one set per game"
    )
    parser.add_argument("--save", default="saves/game.db", help="the save file")
    parser.add_argument("--game", help="game id (default: the latest game)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the game's packages")
    export = sub.add_parser("export", help="write the game's packages as JSON")
    export.add_argument("file", nargs="?", help="output file (default: print them)")
    load = sub.add_parser("import", help="add packages from an exported JSON file")
    load.add_argument("file")
    args = parser.parse_args(argv)
    store = SaveStore(args.save)
    try:
        game_id = args.game or store.latest_game()
        if game_id is None:
            raise SystemExit("No games in that save file yet.")
        library = PolicyLibrary(store.conn, game_id)
        if args.command == "list":
            print(library.listing())
        elif args.command == "export":
            text = library.export()
            if args.file:
                Path(args.file).write_text(text, encoding="utf-8")
                print(f"Wrote {len(library.names())} packages to {args.file}.")
            else:
                print(text)
        else:
            count = library.import_(Path(args.file).read_text(encoding="utf-8"))
            print(f"Imported {count} packages into game {game_id}.")
    finally:
        store.close()


def base_rates_main(argv: list[str]) -> None:
    """``hog-sim base-rates``: the event base rates, from placeholder to fitted, against how
    often each event happened in history and in the outcomes chosen in saved games (D7)."""
    from hog_sim.forecasting.base_rates import HISTORICAL, Observation, tag_frequencies
    from hog_sim.forecasting.scoring import BASE_RATES, FITTED_RATES

    parser = argparse.ArgumentParser(
        prog="hog-sim base-rates",
        description="Event base rates: placeholders, fitted to history, and saved games",
    )
    parser.add_argument("--save", default="saves/game.db", help="the save file")
    args = parser.parse_args(argv)
    played: list[Observation] = []
    if Path(args.save).exists():
        store = SaveStore(args.save)
        try:
            for (game_id,) in store.conn.execute("SELECT game_id FROM games").fetchall():
                for record in store.load(game_id)[2]:
                    played.append(Observation(episode=game_id, tags=record.outcome.events))
        finally:
            store.close()
    history, games = tag_frequencies(HISTORICAL), tag_frequencies(played)
    print(f"{'event':<22}{'placeholder':>12}{'fitted':>8}{'history':>9}{'games':>8}")
    for tag, prior in BASE_RATES.items():
        seen = f"{games.get(tag, 0.0):.2f}" if played else "-"
        print(
            f"{tag:<22}{prior:>12.2f}{FITTED_RATES[tag]:>8.2f}"
            f"{history.get(tag, 0.0):>9.2f}{seen:>8}"
        )
    print(
        f"\nHistory: {len(HISTORICAL)} UK episodes. Games: {len(played)} turns in {args.save}."
        "\nGames are not fitted (their outcomes are the model's own guesses); a large gap"
        " between games and fitted shows where the model's outcomes drift from history."
    )


def knowledge_main(argv: list[str]) -> None:
    """``hog-sim knowledge stats|export|import|export-scenarios``."""
    from hog_sim.knowledge.store import KnowledgeStore

    parser = argparse.ArgumentParser(
        prog="hog-sim knowledge",
        description="What the game has kept from Claude's turns, for prompts and offline play",
    )
    parser.add_argument("--db", default="saves/game.db", help="the save file holding the store")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stats", help="count stored entries by kind")
    for name, text in (
        ("export", "write every entry to a JSONL file"),
        ("import", "load entries from a JSONL file written by export"),
        ("export-scenarios", "write stored scenarios as scenario-library JSONL"),
    ):
        sub.add_parser(name, help=text).add_argument("file")
    args = parser.parse_args(argv)

    if args.command != "import" and not Path(args.db).exists():
        raise SystemExit(f"No save file at {args.db}; play a game with Claude first.")
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    store = KnowledgeStore(args.db)
    try:
        if args.command == "stats":
            for kind, n in store.stats().items():
                print(f"{kind:<16}{n:>6}")
            applied = sum(e.status == "applied" for e in store.graph_changes())
            print(f"({applied} graph changes were applied in play)")
        elif args.command == "export":
            print(f"Wrote {store.export_jsonl(args.file)} entries to {args.file}.")
        elif args.command == "import":
            print(f"Imported {store.import_jsonl(args.file)} entries into {args.db}.")
        else:
            print(f"Wrote {store.export_scenarios(args.file)} scenarios to {args.file}.")
    finally:
        store.close()


def _with_log(client, provider: str, path: str):
    """Wrap ``client`` so every call is written to the log at ``path`` ('off' for none)."""
    from hog_sim.llm.calllog import CallLog, LoggingClient, log_disabled

    if log_disabled(path):
        return client
    return LoggingClient(client, CallLog(path), provider)


def _log_turns(client, game_id: str, turn_of) -> None:
    from hog_sim.llm.calllog import LoggingClient

    if isinstance(client, LoggingClient):
        client.game_id, client.turn_of = game_id, turn_of


def _close_log(client) -> None:
    from hog_sim.llm.calllog import LoggingClient

    if isinstance(client, LoggingClient):
        client.log.close()


def logs_main(argv: list[str]) -> None:
    """``hog-sim logs list|show|export|stats``: read the log of model requests and replies."""
    from hog_sim.llm.calllog import DEFAULT_PATH, LOG_ENV, CallLog, log_disabled

    env_path = os.environ.get(LOG_ENV, "")

    parser = argparse.ArgumentParser(
        prog="hog-sim logs",
        description="Every request the game sent to the model, and the reply",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_PATH if log_disabled(env_path) else env_path,
        help=f"the log file (default {DEFAULT_PATH})",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="one line per call, most recent last")
    listing.add_argument("--game", help="only this game id")
    listing.add_argument("--turn", type=int, help="only this turn")
    listing.add_argument("--kind", help="only this call type, e.g. ScenarioDraft")
    listing.add_argument("--errors", action="store_true", help="only calls that failed")
    listing.add_argument("-n", "--limit", type=int, default=50, help="how many (default 50)")
    show = sub.add_parser("show", help="the full request and reply of one call")
    show.add_argument("id", help="a call id from list, or 'last'")
    show.add_argument("--schema", action="store_true", help="also print the output schema")
    sub.add_parser("export", help="write every call in full to a JSONL file").add_argument("file")
    sub.add_parser("stats", help="how many calls, and how much space they take")
    args = parser.parse_args(argv)

    if not Path(args.db).exists():
        raise SystemExit(f"No log at {args.db}; play a game with the model first.")
    calllog = CallLog(args.db)
    try:
        if args.command == "list":
            rows = calllog.calls(
                game_id=args.game,
                turn=args.turn,
                schema_name=args.kind,
                errors_only=args.errors,
                limit=args.limit,
            )
            print(
                f"{'id':>5}  {'time (UTC)':<19}  {'turn':>4}  {'call':<16} {'try':>3}  "
                f"{'tokens in/out':>13}  {'secs':>5}  model"
            )
            for r in rows:
                print(_log_line(r))
            if not rows:
                print("(no calls match)")
        elif args.command == "show":
            if args.id == "last":
                last = calllog.calls(limit=1)
                entry = calllog.call(last[0]["id"]) if last else None
            elif args.id.isdigit():
                entry = calllog.call(int(args.id))
            else:
                raise SystemExit("Give a call id from `hog-sim logs list`, or 'last'.")
            if entry is None:
                raise SystemExit(f"No call {args.id} in {args.db}.")
            print(_log_detail(entry, with_schema=args.schema))
        elif args.command == "export":
            with open(args.file, "w", encoding="utf-8") as f:
                rows = calllog.calls()
                for r in rows:
                    f.write(json.dumps(calllog.call(r["id"]), ensure_ascii=False) + "\n")
            print(f"Wrote {len(rows)} calls to {args.file}.")
        else:
            s = calllog.stats()
            kb = 1024
            print(f"{s['calls']} calls, {s['texts']} distinct texts")
            print(f"  as plain text, one copy per call: {s['logical_bytes'] / kb:,.0f} KB")
            print(f"  after removing repeats:           {s['unique_bytes'] / kb:,.0f} KB")
            print(f"  after compression:                {s['stored_bytes'] / kb:,.0f} KB")
            print(f"  log file on disk:                 {s['file_bytes'] / kb:,.0f} KB")
    finally:
        calllog.close()


def _log_line(r: dict) -> str:
    turn = "" if r["turn"] is None else r["turn"]
    tokens = f"{r['input_tokens']}/{r['output_tokens']}" if r["error"] is None else "FAILED"
    model = r["served_model"] or r["model"]
    if r["provider"]:
        model += f" ({r['provider']})"
    return (
        f"{r['id']:>5}  {r['at'][:19].replace('T', ' '):<19}  {turn:>4}  "
        f"{r['schema_name'][:16]:<16} {r['attempt']:>3}  {tokens:>13}  "
        f"{r['latency_ms'] / 1000:>5.1f}  {model}"
    )


def _log_detail(e: dict, with_schema: bool = False) -> str:
    rule = "-" * 72
    lines = [
        f"Call {e['id']} at {e['at']}: {e['schema_name']} (prompt {e['prompt_version']}), "
        f"attempt {e['attempt']}",
        f"Game {e['game_id'] or '-'}, turn {'-' if e['turn'] is None else e['turn']}. "
        f"Provider {e['provider'] or '-'}, model {e['model']}"
        + (f", answered by {e['served_model']}" if e["served_model"] else "")
        + f", effort {e['effort']}.",
    ]
    if e["error"] is None:
        lines.append(
            f"{e['input_tokens']} tokens in, {e['output_tokens']} out, "
            f"{e['latency_ms'] / 1000:.1f}s."
        )
    else:
        lines.append(f"FAILED after {e['latency_ms'] / 1000:.1f}s: {e['error']}")
    lines += [rule, "SYSTEM PROMPT", rule, e["system"]]
    if with_schema:
        lines += [rule, "OUTPUT SCHEMA", rule, json.dumps(e["output_schema"], indent=2)]
    for i, m in enumerate(e["messages"]):
        if m["role"] == "assistant":
            label = "EARLIER REPLY (rejected)"
        else:
            label = "PROMPT" if i == 0 else "FEEDBACK ON THE REJECTED REPLY"
        lines += [rule, label, rule, m["content"]]
    if e["response"] is not None:
        lines += [rule, "REPLY", rule, _pretty_json(e["response"])]
    return "\n".join(lines)


def _pretty_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except ValueError:
        return text


def _briefing(game: Game) -> str:
    """The turn's in-tray: the lead scenario with its options, then the other items."""
    s, state = game.scenario, game.state
    lines = [_banner(s.title), ""]
    story = state.storylines.get(s.storyline or "")
    if story is not None:
        lines += [_wrap(f"(Continues: {story.title}, now at stage {story.stage + 1})"), ""]
    lines.append(_paragraphs(s.briefing))
    lines.append(_section("Options"))
    for i, option in enumerate(s.suggested_options, 1):
        lines += ["", _numbered(i, option)]
    if s.secondary:
        lines.append(_section("Also in your in-tray"))
        lines.append("(act on any of them in your answer, or leave them)")
        for item in s.secondary:
            lines += ["", _wrap(f"- {item.title}: {item.briefing}", "  ", "    ")]
    shown = {s.storyline, *(item.storyline for item in s.secondary)}
    ignored = [
        t
        for t in state.storylines.values()
        if t.open and t.id not in shown and state.turn - t.last_addressed >= 2
    ]
    if ignored:
        listed = "; ".join(f"{t.title} (stage {t.stage})" for t in ignored)
        lines += ["", _wrap(f"Still unresolved: {listed}", "", "  ")]
    return "\n".join(lines)


def _left_alone(record: TurnRecord, played: WorldState) -> str:
    """The in-tray items this turn's decisions did not help (``played``: the world the turn
    was played in), and what became of them."""
    items = [i for i in record.scenario.secondary if not addressed(i, record.actions, played)]
    if not items:
        return ""
    stories = record.state_after.storylines

    def fate(item: SideIssue) -> str:
        story = stories.get(item.storyline or "")
        if story is not None and not story.open:
            return "faded away"
        if harmed(item, record.actions, played):
            return "made worse, carries over"
        return "carries over"

    parts = [f"{i.title} ({fate(i)})" for i in items]
    return "\n" + _wrap("Left in your in-tray: " + "; ".join(parts), "", "  ")


def main(argv: list[str] | None = None) -> None:
    import sys

    from hog_sim.llm.client import LLMUnavailable

    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["knowledge"]:
        knowledge_main(argv[1:])
        return
    if argv[:1] == ["logs"]:
        logs_main(argv[1:])
        return
    if argv[:1] == ["packages"]:
        packages_main(argv[1:])
        return
    if argv[:1] == ["base-rates"]:
        base_rates_main(argv[1:])
        return
    try:
        _main(argv)
    except LLMUnavailable as exc:
        # Turns already played are saved; the message says which route failed and why.
        raise SystemExit(f"Could not reach Claude. {exc}") from None


def _offline_plugins(knowledge: KnowledgeStore, seed: int) -> tuple:
    """The offline scenarios, interpreter and forecaster: for offline practice, and as the
    stand-ins for one step when the model fails (E22). The scenarios keep to the calendar
    (E19) on both paths."""
    return (
        OnCalendar(ScenarioLibrary.load(knowledge.library_scenarios(), seed=seed)),
        StoredInterpreter(knowledge, KeywordInterpreter()),
        StoredForecaster(knowledge, EngineForecaster()),
    )


def _env_call_budget() -> int:
    """The per-turn call budget from the environment, or the default when it is unset or
    not a whole number (with a message saying so)."""
    from hog_sim.llm.budget import BUDGET_ENV, DEFAULT_LIMIT

    raw = (os.environ.get(BUDGET_ENV) or "").strip()
    if not raw:
        return DEFAULT_LIMIT
    try:
        return int(raw)
    except ValueError:
        print(
            f"{BUDGET_ENV}={raw!r} is not a whole number of calls, so the default of "
            f"{DEFAULT_LIMIT} is used (0 means no cap)."
        )
        return DEFAULT_LIMIT


def _main(argv: list[str] | None) -> None:
    from hog_sim.core.env import load_env
    from hog_sim.llm.budget import DEFAULT_LIMIT, BudgetedClient
    from hog_sim.llm.calllog import LOG_ENV, LOG_FILE
    from hog_sim.llm.providers import DEFAULT_PROVIDER, PROVIDERS

    load_env()
    parser = argparse.ArgumentParser(description="Head of Government Simulator")
    parser.add_argument("--save", default="saves/game.db")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--central-world",
        action="store_true",
        help="new games only: the economy behaves exactly as the forecast's central guess, "
        "instead of as a hidden world drawn from the forecast's uncertainty",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--llm",
        action="store_true",
        help="insist on Claude: stop with an explanation instead of falling back to offline "
        "practice (Claude is already used by default whenever it can be reached)",
    )
    mode.add_argument(
        "--offline",
        action="store_true",
        help="offline practice: built-in scenarios and keyword matching, no Claude",
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=os.environ.get("HOG_SIM_PROVIDER", DEFAULT_PROVIDER),
        help="how to reach the model: claude-code (the default: your Claude Code sign-in, "
        "falling back to the API when Claude Code is missing or signed out and an API key "
        "exists), api (API key or `ant auth login`) or codex (OpenAI Codex signed in with "
        "ChatGPT via `codex login`; set HOG_SIM_CODEX_MODEL to pick its model)",
    )
    parser.add_argument(
        "--llm-log",
        default=os.environ.get(LOG_ENV),
        help="file that records every request sent to the model and its reply "
        "(default: llm-log.db next to the save file; 'off' keeps no log). "
        "Read it with: hog-sim logs",
    )
    parser.add_argument(
        "--luck",
        choices=["sample", "fair", "argmax", "hard"],
        default="sample",
        help="how each turn's outcome is picked (new games only): sample draws by probability "
        "(the default), fair also draws but spreads bad luck out so it can't cluster, argmax "
        "always takes the most likely, hard takes the worse of the two likeliest",
    )
    parser.add_argument(
        "--check-llm",
        action="store_true",
        help="make one small call to the model to confirm the connection works, then exit",
    )
    parser.add_argument(
        "--call-budget",
        type=int,
        default=_env_call_budget(),
        help=f"most model calls a turn may make before asking you (default {DEFAULT_LIMIT}; "
        "0 for no cap)",
    )
    args = parser.parse_args(argv)
    if args.llm_log is None:
        args.llm_log = str(Path(args.save).parent / LOG_FILE)

    if args.check_llm:
        from hog_sim.llm.providers import describe

        client, provider, note = _llm_client(args.provider)
        client = _with_log(client, provider, args.llm_log)
        if note:
            print(note)
        try:
            print(check_llm(client) + f" Connected via {describe(provider)}.")
        finally:
            _close_log(client)
        return

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    store = SaveStore(args.save)
    knowledge = KnowledgeStore(args.save)
    recaller = Recaller(knowledge)
    client = None
    advisers = None
    if not args.offline:
        from hog_sim.llm.client import LLMError
        from hog_sim.llm.providers import make_client

        try:
            client, provider, note = make_client(args.provider)
        except LLMError as exc:
            if args.llm:
                raise SystemExit(str(exc)) from None
            print(f"{NO_CLAUDE_NOTE}\n{exc}\n")
        else:
            budget = BudgetedClient(client, max(0, args.call_budget))
            client = _with_log(budget, provider, args.llm_log)
    if client:
        from hog_sim.forecasting.candidates import ForecastConfig
        from hog_sim.game.llm_plugins import llm_plugins
        from hog_sim.llm.client import ModelConfig

        model_config, forecast_config = ModelConfig(), ForecastConfig()
        plugins = llm_plugins(
            client,
            model_config=model_config,
            forecast_config=forecast_config,
            recaller=recaller,
        )
        advisers = Advisers(client, config=model_config)
        print(_llm_banner(model_config, forecast_config, provider, note))
    else:
        plugins = _offline_plugins(knowledge, args.seed)
        print(OFFLINE_BANNER)
        kept = knowledge.stats()
        if kept["scenario"]:
            print(
                f"Reusing {kept['scenario']} scenarios and {kept['outcome']} outcomes "
                "Claude wrote in earlier games."
            )
    game_id = store.latest_game() if args.resume else None
    if game_id:
        config, start, records = store.load(game_id)
        recaller.game_id = game_id
        _log_turns(client, game_id, lambda: (records[-1].state_after if records else start).turn)
        turn = (records[-1].state_after if records else start).turn
        game = Game.resume(
            config,
            start,
            records,
            *plugins,
            scenario=store.pending_scenario(game_id, turn),
            on_scenario=partial(store.save_scenario, game_id),
        )
        print(f"Resumed game {game_id} at turn {game.state.turn}.")
    else:
        config = GameConfig(
            seed=args.seed, selection_mode=args.luck, hidden_world=not args.central_world
        )
        start = toy_world()
        game_id = store.new_game(config, start)
        recaller.game_id = game_id
        _log_turns(client, game_id, lambda: start.turn)
        game = Game(config, start, *plugins, on_scenario=partial(store.save_scenario, game_id))
        print(f"New game {game_id}. Election at turn {config.election_turn}. Ctrl-D to quit.")

    _log_turns(client, game_id, lambda: game.state.turn)
    if client:
        budget.turn_of = lambda: game.state.turn
    library = PolicyLibrary(store.conn, game_id)
    # Offline stand-ins for one step when the model fails (E22); none in offline play.
    fallback = _offline_plugins(knowledge, args.seed) if client else None
    pending = store.pending_proposal(game_id, game.state.turn)
    if pending is not None:
        try:
            game.check(pending)
            print("Your confirmed response from last time is waiting; review it to play it.")
        except StaleProposal:
            pending = None
    try:
        if game.over:  # resumed after the election
            print(_finished(game, args.save))
            return
        while not game.over:
            print(_dashboard(game.start, game.state, _previous(game), game.config.election_turn))
            calls_before = len(client.usage.records) if client else 0
            ran_offline: set[str] = set()  # steps played offline after a model failure
            _guarded(
                game,
                "scenarios",
                "writing this turn's scenario",
                lambda: game.scenario,
                fallback[0] if fallback else None,
                ran_offline,
            )
            print(_briefing(game))
            state_before = game.state
            proposal = None
            while proposal is None:
                proposal = _ask_for_turn(game, advisers, library, fallback, pending)
                pending = None
            # Saved before the outcome is asked for, so a failure or a quit keeps it.
            store.save_proposal(game_id, proposal)
            record = _guarded(
                game,
                "forecaster",
                "writing what happened",
                partial(game.commit, proposal, offline=ran_offline),
                fallback[2] if fallback else None,
                ran_offline,
            )
            store.save_turn(game_id, record)
            if client:
                knowledge.harvest_turn(
                    game_id, record, state_before, _provenance(client, calls_before)
                )
            print(_report(record))
            left = _left_alone(record, state_before)
            if left:
                print(left)
            print(EXPLAIN_HINT)
            if client:
                print(_usage_line(client, calls_before))
        print(_dashboard(game.start, game.state, _previous(game), game.config.election_turn))
        if game.config.hidden_world:
            print(_surprises(game.state, game.config.seed))
    except (EOFError, KeyboardInterrupt):
        print(f"\nSaved. Resume with: hog-sim --resume --save {args.save}")
    finally:
        store.close()
        knowledge.close()
        _close_log(client)


if __name__ == "__main__":
    main()
