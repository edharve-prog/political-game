"""Run the interpreter eval cases against the live model and record a cassette.

    uv run --extra llm python -m hog_sim.llm.evaluate [--provider api|claude-code]

Writes tests/cassettes/interpreter.json (commit it so CI replays real model output) and
prints one line per case. Re-run after changing the interpreter prompt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hog_sim.core.env import load_env
from hog_sim.llm.client import LLMClient, RecordingClient
from hog_sim.llm.eval_cases import CASES, score
from hog_sim.llm.interpreter import interpret
from hog_sim.llm.providers import DEFAULT_PROVIDER, PROVIDERS, make_client
from hog_sim.llm.summary import summarise_state
from hog_sim.world.seed.toy import toy_world

DEFAULT_CASSETTE = Path("tests/cassettes/interpreter.json")


def run(client: LLMClient) -> list[tuple[str, list[str]]]:
    summary = summarise_state(toy_world())
    results = []
    for case in CASES:
        result = interpret(case.text, summary, client)
        results.append((case.text, score(case, result.actions, result.clarifying_question)))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cassette", type=Path, default=DEFAULT_CASSETTE)
    parser.add_argument("--provider", choices=PROVIDERS, default=DEFAULT_PROVIDER)
    args = parser.parse_args(argv)
    load_env()
    inner, _, note = make_client(args.provider)
    if note:
        print(note)
    client = RecordingClient(args.cassette, inner=inner)
    results = run(client)
    for text, problems in results:
        print(f"{'FAIL' if problems else 'ok  '} {text}" + "".join(f"\n     {p}" for p in problems))
    passed = sum(not p for _, p in results)
    print(f"\n{passed}/{len(results)} passed, ${inner.usage.total_cost_usd:.3f}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
