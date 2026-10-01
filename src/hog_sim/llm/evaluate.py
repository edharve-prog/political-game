"""Run the interpreter eval cases against the live model and record a cassette.

    ANTHROPIC_API_KEY=... uv run --extra llm python -m hog_sim.llm.evaluate

Writes tests/cassettes/interpreter.json (commit it so CI replays real model output) and
prints one line per case. Re-run after changing the interpreter prompt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hog_sim.llm.client import AnthropicClient, LLMClient, RecordingClient
from hog_sim.llm.eval_cases import CASES, score
from hog_sim.llm.interpreter import interpret
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
    args = parser.parse_args(argv)
    client = RecordingClient(args.cassette, inner=AnthropicClient())
    results = run(client)
    for text, problems in results:
        print(f"{'FAIL' if problems else 'ok  '} {text}" + "".join(f"\n     {p}" for p in problems))
    passed = sum(not p for _, p in results)
    print(f"\n{passed}/{len(results)} passed, ${client.usage.total_cost_usd:.3f}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
