"""Minimal text interface: ``uv run hog-sim`` (add ``--resume`` to continue the last save).

The game plays with Claude whenever it can reach it, and drops to offline practice (canned
scenarios, keyword matching) only when it can't, or when asked to with ``--offline``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hog_sim.content.library import ScenarioLibrary
from hog_sim.core.config import GameConfig
from hog_sim.core.models import SideIssue
from hog_sim.core.state import WorldState
from hog_sim.game.interfaces import NeedsClarification
from hog_sim.game.loop import Game, Proposal
from hog_sim.game.persistence import SaveStore
from hog_sim.game.records import TurnRecord
from hog_sim.game.stubs import EngineForecaster, KeywordInterpreter
from hog_sim.knowledge.offline import StoredForecaster, StoredInterpreter
from hog_sim.knowledge.recall import Recaller
from hog_sim.knowledge.store import KnowledgeStore
from hog_sim.llm.advisers import Advisers
from hog_sim.population.popularity import national_approval, vote_intention
from hog_sim.ui.builder import compose_response, describe, edit_actions
from hog_sim.world.seed.toy import toy_world
from hog_sim.world.storylines import addressed


def _dashboard(start: WorldState, state: WorldState, previous: WorldState | None = None) -> str:
    """Current scores, with the change since the last turn and since the game began."""
    previous = previous or start
    lines = [f"Turn {state.turn}" + ("" if state.turn == 0 else "    (last turn, since start)")]
    for ind in state.indicators.values():
        last = ind.value - previous.indicators[ind.id].value
        total = ind.value - start.indicators[ind.id].value
        lines.append(f"  {ind.name:<26}{ind.value:>8.2f} {ind.unit:<6} ({last:+.2f}, {total:+.2f})")
    lines.append("  Approval")
    for g in state.groups.values():
        last = (g.approval - previous.groups[g.id].approval) * 100
        total = (g.approval - start.groups[g.id].approval) * 100
        lines.append(f"    {g.name:<24}{g.approval * 100:>6.1f}%  ({last:+.1f}, {total:+.1f})")
    national, vote = national_approval(state), vote_intention(state)
    lines.append(
        f"  National {national * 100:.1f}% ({(national - national_approval(previous)) * 100:+.1f})"
        f"  ·  Vote intention {vote * 100:.1f}% ({(vote - vote_intention(previous)) * 100:+.1f})"
    )
    return "\n".join(lines)


def _previous(game: Game) -> WorldState:
    """The state before the last turn played, so the dashboard can show what it changed."""
    return game.history[-2].state_after if len(game.history) >= 2 else game.start


def _report(record: TurnRecord) -> str:
    lines = [f"Note: {n}" for n in record.notes] + [record.outcome.narrative]
    lines += [f"  (action: {a.kind} {a.target} {a.magnitude:+.2f})" for a in record.actions]
    if record.election:
        e = record.election
        verdict = "You win a majority." if e.majority else "You lose your majority."
        lines.append(
            f"ELECTION: {e.vote_share * 100:.1f}% of the vote, "
            f"{e.seats}/{e.total_seats} seats. {verdict}"
        )
    return "\n".join(lines)


REVIEW_PROMPT = "\nEnter to confirm, or edit (drop 2 · 2 size 0.3 · 2 turns 4 · redo)> "


def _proposal_text(proposal: Proposal, dropped: list[str]) -> str:
    lines = ["\nYour advisers read that as:"]
    if proposal.actions:
        lines += [f"  {i}. {describe(a)}" for i, a in enumerate(proposal.actions, 1)]
    else:
        lines.append("  (no actions: this turn the government does nothing)")
    lines += [f"  (not possible: {line})" for line in dropped]
    lines += [f"  Note: {n}" for n in proposal.notes]
    lines.append(f"  Delivery: {proposal.delivery.describe()}")
    return "\n".join(lines)


def _ask_for_turn(game: Game, advisers: Advisers | None = None) -> Proposal | None:
    """Build, review and confirm this turn's actions. ``None`` means start the response again.

    ``advise <question>`` asks the advisers for more options instead (Claude mode only)."""
    while True:
        raw = input(RESPONSE_PROMPT)
        if raw.strip().lower().split()[:1] == ["advise"]:
            print(_advice(game, advisers, raw.strip()[len("advise") :].strip()))
            continue
        options = game.scenario.suggested_options
        try:
            response = compose_response(raw, options)
            proposal = game.propose(response)
            break
        except ValueError as err:
            print(f"  {err}")
        except NeedsClarification as ask:
            print(f"\nYour advisers ask: {ask.question}")
    dropped = getattr(game.interpreter, "dropped", lambda: [])()
    while True:
        print(_proposal_text(proposal, dropped))
        command = input(REVIEW_PROMPT).strip()
        if command.lower() in ("", "y", "yes", "ok"):
            return proposal
        if command.lower() == "redo":
            return None
        try:
            proposal = game.revise(proposal, edit_actions(proposal.actions, command))
            dropped = []
        except ValueError as err:
            print(f"  {err}")


RESPONSE_PROMPT = (
    "\nYour response (option numbers, words, or both: 1 3 + freeze fares; or advise <question>)> "
)


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
    start = len(game.scenario.suggested_options) + 1
    game.add_options([a.option for a in advice])
    lines = ["  Your advisers suggest:"]
    for i, a in enumerate(advice, start):
        lines += [f"  {i}. {a.option}", f"       Trade-off: {a.trade_off}"]
    return "\n".join(lines)


OFFLINE_BANNER = (
    "Mode: OFFLINE PRACTICE. Scenarios come from a built-in library and responses are\n"
    "matched by keywords; Claude is not used."
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
        from hog_sim.llm.codex import MODEL_ENV

        return (
            (f"{note}\n" if note else "")
            + f"Mode: CODEX ({os.environ.get(MODEL_ENV) or 'Codex default model'}) via "
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


def _briefing(game: Game) -> str:
    """The turn's in-tray: the lead scenario with its options, then the other items."""
    s, state = game.scenario, game.state
    lines = [f"\n== {s.title} =="]
    story = state.storylines.get(s.storyline or "")
    if story is not None:
        lines.append(f"(Continues: {story.title}, now at stage {story.stage + 1})")
    lines.append(s.briefing)
    lines += [f"  {i}. {option}" for i, option in enumerate(s.suggested_options, 1)]
    if s.secondary:
        lines.append("\nAlso in your in-tray (act on any of them in your answer, or leave them):")
        lines += [f"  - {item.title}: {item.briefing}" for item in s.secondary]
    shown = {s.storyline, *(item.storyline for item in s.secondary)}
    ignored = [
        t
        for t in state.storylines.values()
        if t.open and t.id not in shown and state.turn - t.last_addressed >= 2
    ]
    if ignored:
        listed = "; ".join(f"{t.title} (stage {t.stage})" for t in ignored)
        lines.append(f"Still unresolved: {listed}")
    return "\n".join(lines)


def _left_alone(record: TurnRecord) -> str:
    items = [i for i in record.scenario.secondary if not addressed(i, record.actions)]
    if not items:
        return ""
    stories = record.state_after.storylines

    def fate(item: SideIssue) -> str:
        story = stories.get(item.storyline or "")
        return "faded away" if story is not None and not story.open else "carries over"

    parts = [f"{i.title} ({fate(i)})" for i in items]
    return "Left in your in-tray: " + "; ".join(parts)


def main(argv: list[str] | None = None) -> None:
    import sys

    from hog_sim.llm.client import LLMUnavailable

    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["knowledge"]:
        knowledge_main(argv[1:])
        return
    try:
        _main(argv)
    except LLMUnavailable as exc:
        # Turns already played are saved; the message says which route failed and why.
        raise SystemExit(f"Could not reach Claude. {exc}") from None


def _main(argv: list[str] | None) -> None:
    from hog_sim.core.env import load_env
    from hog_sim.llm.providers import DEFAULT_PROVIDER, PROVIDERS

    load_env()
    parser = argparse.ArgumentParser(description="Head of Government Simulator")
    parser.add_argument("--save", default="saves/game.db")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
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
        "--check-llm",
        action="store_true",
        help="make one small call to the model to confirm the connection works, then exit",
    )
    args = parser.parse_args(argv)

    if args.check_llm:
        from hog_sim.llm.providers import describe

        client, provider, note = _llm_client(args.provider)
        if note:
            print(note)
        print(check_llm(client) + f" Connected via {describe(provider)}.")
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
        plugins = (
            ScenarioLibrary.load(knowledge.library_scenarios(), seed=args.seed),
            StoredInterpreter(knowledge, KeywordInterpreter()),
            StoredForecaster(knowledge, EngineForecaster()),
        )
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
        game = Game.resume(config, start, records, *plugins)
        print(f"Resumed game {game_id} at turn {game.state.turn}.")
    else:
        config, start = GameConfig(seed=args.seed), toy_world()
        game_id = store.new_game(config, start)
        recaller.game_id = game_id
        game = Game(config, start, *plugins)
        print(f"New game {game_id}. Election at turn {config.election_turn}. Ctrl-D to quit.")

    try:
        while not game.over:
            print("\n" + _dashboard(game.start, game.state, _previous(game)))
            print(_briefing(game))
            calls_before = len(client.usage.records) if client else 0
            state_before = game.state
            proposal = None
            while proposal is None:
                proposal = _ask_for_turn(game, advisers)
            record = game.commit(proposal)
            store.save_turn(game_id, record)
            if client:
                knowledge.harvest_turn(
                    game_id, record, state_before, _provenance(client, calls_before)
                )
            print("\n" + _report(record))
            left = _left_alone(record)
            if left:
                print(left)
            if client:
                print(_usage_line(client, calls_before))
        print("\n" + _dashboard(game.start, game.state, _previous(game)))
    except (EOFError, KeyboardInterrupt):
        print(f"\nSaved. Resume with: hog-sim --resume --save {args.save}")
    finally:
        store.close()
        knowledge.close()


if __name__ == "__main__":
    main()
