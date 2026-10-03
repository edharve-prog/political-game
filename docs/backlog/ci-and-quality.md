# Backlog: CI and Test Quality

Raised by Ed on 2026-10-02 as eight items, CI-1 to CI-8. Story ids are stable: refer to them
in PRs. Assessed against main @ 5056df2 (after PR #23).

## Where CI stands today

- One job in `.github/workflows/ci.yml`: Python 3.12 only, `uv sync` (not `--locked`), ruff,
  ruff format, pytest. No `permissions`, no concurrency, no timeout, no Dependabot, no CodeQL.
- Full suite locally: 191 passed, 5 skipped in 27 s. The balance test (EB-1) takes about
  20 s of that (18 s fixture plus a 2 s 120-turn game).
- Checked in this sandbox: the suite passes on Python 3.11 and 3.13 (balance test
  excluded). Only a 3.14 release candidate was available here and pydantic fails to import
  on it, so 3.14 needs checking in CI on the final release.
- `uv sync --locked` passes, so the lock file is current.
- The wheel builds and includes the scenario JSON. The installed wheel runs an offline game
  on Python 3.11 from outside the repo, with input piped in, then saves and resumes.
- `actions/checkout` is at v7.0.1 (floating `v7` tag exists). `astral-sh/setup-uv` is at
  v10.2.0 and has **no** floating `v10` tag, so it must be pinned to an exact version or SHA.
  Both checked with `git ls-remote --tags` on 2026-10-02.

## Priority

**P1** = do now, cheap and protects everything after it. **P2** = next, once Project 17 has
landed. **P3** = later or when its trigger arrives.

| Order | Id | Item | Priority | Verdict | Depends on | Size |
|---|---|---|---|---|---|---|
| 1 | CI-7 | Modernise Actions | P1 | Keep. Versions confirmed | none | S |
| 2 | CI-6 | Workflow and repo hardening | P1 | Keep, with changes | CI-7 (same PR) | S |
| 3 | CI-3 | Python and package matrix | P1 | Keep, with changes | CI-6 | S |
| 4 | CI-1 | Engine semantic regression suite | P1 | Mostly delivered by Project 17 | Project 17 PRs 1-3 | XS here |
| 5 | CI-4 | Split fast tests from balance | P2 | Keep, with changes | CI-3, Project 17 | M |
| 6 | CI-2 | Property and invariant tests | P2 | Keep, with changes | Project 17 PRs 3-4 | M |
| 7 | CI-5 | Coverage and mutation testing | P3 (coverage P2) | Keep, report-only first | CI-4 | M |
| 8 | CI-8 | Web app contract and E2E lane | P3 | Defer until the web UI exists | Project 10 web UI | L |

Branch protection (part of CI-6) is a setting Ed changes on GitHub. It goes last in the P1
batch, because required checks are named by job, and CI-3 and CI-4 rename jobs.

## PR plan

- **PR A (CI-7 + CI-6 workflow part).** Bump actions, add permissions, concurrency,
  timeouts, Dependabot.
- **PR B (CI-3).** Matrix, `--locked`, wheel build and installed-CLI smoke test.
- **Ed, in GitHub settings, after PR B:** turn on CodeQL default setup and a branch rule
  for `main` (see CI-6).
- **CI-1** has no PR of its own. It is checked off as Project 17's PRs merge.
- **PR C (CI-4)** after Project 17 merges, because its fixes will move balance results.
- **PR D (CI-2)** after Project 17 PRs 3 and 4 (it asserts `PropagationError` and the new
  validators).
- **PR E (CI-5).**

---

## CI-1 Engine semantic regression suite (P1, via Project 17)

Verdict: right list, but almost all of it is already promised. Project 17 (engine
correctness, its own thread) has the rule that each story ships with a test that fails on
main and passes after the fix. The mapping:

| CI-1 item | Project 17 story |
|---|---|
| Military and diplomatic semantics | EC-1 |
| More than 3 deterministic action graph changes | EC-2 |
| Raw requested-action retention | EC-4 |
| Package fiscal limits | EC-5 |
| Exact group-policy duration | EC-7 |
| No-claim candidate scoring | EC-8 |
| `"none"` tag exclusivity | EC-9 |

Improvement: when Project 17 is done, check every row above has a named test, and add a
`semantics` pytest marker on them so the set can be run alone (`pytest -m semantics`) and
counted. Action–target compatibility (EC-3) and hard scenario conditions (EC-10) should join
the set, since they are the same kind of defect.

Done when every row has a passing, marked test on main.

## CI-2 Property and invariant tests (P2)

Verdict: worth doing. Example-based tests missed a sign error and an off-by-one, which is
the gap property tests fill.

Improvements:
- Add `hypothesis` to the dev group. Use a `ci` profile (about 50 examples,
  `derandomize=True`, no deadline) so PR runs are repeatable and never flaky, and a
  `nightly` profile with 1,000 examples in the scheduled job from CI-4.
- Generate actions from the real `PolicyAction` model over the toy world's valid kinds and
  targets, plus a small share of invalid ones to exercise blocking.
- Small random graphs should include cycles, since non-convergence is a stated invariant.

Invariants (from Ed's list, made testable):
- No NaN or Inf anywhere in state after any turn.
- Replay of a logged game gives an identical state.
- Applied state stays inside every indicator's bounds.
- Outcome probabilities are each in [0, 1] and sum to 1 within 1e-9.
- Every node and edge id in state, changes and outcomes refers to something that exists.
- An unstable graph raises `PropagationError` (EC-6) instead of returning a result.
- Invalid `GameConfig` and `WorldState` values are rejected at construction (EC-11).

Done when these run in the fast lane in under 10 s with the `ci` profile.

## CI-3 Python and package matrix (P1)

Verdict: keep. Today 3.11 is declared but never tested.

Improvements:
- Matrix: 3.11 (the floor), 3.12, 3.13 and 3.14. Use setup-uv's `python-version` input
  instead of a separate `uv python install` step. `fail-fast: false`.
- Run the balance test on one version only (3.12) once CI-4 splits it out. It checks game
  balance, not the interpreter.
- `uv sync --locked`, so a stale lock fails CI instead of being silently re-resolved.
- Package job: `uv build`, install the wheel into a fresh venv, change to a temporary
  directory outside the repo so `src/` cannot be imported, then:
  - `hog-sim --help` exits 0;
  - an offline game with two piped responses saves, and `--resume` loads it;
  - the wheel contains the scenario JSON files (prompts are Python modules, so the import covers them).
- This smoke test also covers create, respond, save and resume through the CLI now, which
  is the part of CI-8 that applies before a web UI exists.

Done when all four versions are green and the installed-wheel smoke job passes.

## CI-4 Split fast tests from balance and evaluation (P2)

Verdict: keep. The split matters less for speed today (20 s) than for clarity and for
room to grow seeds, and the "competent strategy" gap is real. `problems()` in
`game/balance.py` only checks that no single lever wins too often, that doing nothing
never wins and that acting is not punished. It passes if every strategy loses.

Plan:
- Mark balance tests `@pytest.mark.balance`. The fast lane runs `-m "not balance"`.
- A separate required `balance` job runs them, writes `table(results)` to a file and uploads
  it as an artifact on every run, so a reviewer can see the numbers without reading logs.
- A scheduled workflow (nightly, plus manual `workflow_dispatch`) runs 20 seeds over 24 turns
  and 5 seeds over 120 turns, uploads the tables, and runs Hypothesis with the `nightly`
  profile. A scheduled failure does not block PRs.
- **Competent strategy.** Add a "sensible PM" bot to `game/balance.py`, which reads each
  scenario and responds to the lead issue with a funded, mixed package. Add a
  `min_competent_win_share` to `LIMITS`, set from a measured baseline after Project 17 lands,
  not guessed now.

Overlap: `game/balance.py` and `LIMITS` belong to Project 16 (Back-end review thread, EB-1
and EB-7). Agree the new bot and threshold with that thread before the PR.

Done when the PR lane runs no balance test, the balance job uploads its table, the nightly
workflow runs green, and the balance test fails if the competent bot stops winning.

## CI-5 Coverage and mutation testing (P3; coverage P2)

Verdict: keep, but report-only first. A gate set before a baseline exists either blocks good
PRs or is too low to matter.

Plan:
- **Coverage (P2).** `pytest-cov` with `--cov-branch` on the fast lane, report in the job
  summary. After two weeks of baseline, set `fail_under` for `world/`, `policy/` and
  `population/` at the baseline minus 2 points.
- **Mutation (P3).** `mutmut` weekly and on demand, over `world/propagation.py`,
  `policy/limits.py`, `population/popularity.py` and `world/changes.py`. It runs the fast
  tests only (not the balance test), so each mutant takes seconds. The surviving-mutant
  report is uploaded as an artifact. Each surviving sign flip or comparison change is either
  killed by a new test or noted as equivalent.

Done when coverage shows on every PR and the weekly mutation report runs.

## CI-6 Workflow and repository hardening (P1)

Verdict: keep, with two changes.

In the workflow (PR A):
- Top-level `permissions: contents: read`.
- `concurrency: group: ci-${{ github.ref }}`, with `cancel-in-progress` only for pull
  requests, so pushes to main always finish.
- `timeout-minutes: 15` on each job.
- `.github/dependabot.yml`, weekly, with two ecosystems: `github-actions` and `uv`. Group
  minor and patch updates into one PR per ecosystem so Ed isn't merging a stream of single
  bumps.
- Improvement: pin actions to commit SHAs with the version in a comment. Dependabot keeps
  them current, and this protects against a moved tag.

In GitHub settings (Ed, after PR B):
- **CodeQL:** use default setup (Settings → Code security → CodeQL → Default). It needs no
  workflow file and is free for a public repo.
- **Branch rule for `main`:** require a pull request and require the status checks from
  CI-3 (and later the `balance` job). **Don't require approvals.** Ed is the only
  maintainer and GitHub doesn't let an author approve their own PR, so an approval rule
  would block every merge.

Done when the workflow has the settings above, Dependabot has opened its first PRs, CodeQL
shows a first scan, and a direct push to main is refused.

## CI-7 Modernise Actions dependencies (P1)

Verdict: keep. Versions confirmed on 2026-10-02: `actions/checkout` v7.0.1 and
`astral-sh/setup-uv` v10.2.0.

Plan: read both changelogs for breaking changes between v4→v7 and v5→v10 (input renames,
cache defaults), bump in PR A, and turn on setup-uv's `enable-cache` keyed on `uv.lock`.
Done when the run log has no Node-20 deprecation warning.

## CI-8 Web app contract and E2E lane (P3, deferred)

Verdict: right plan, wrong time. The game is CLI-only, and the web UI (FastAPI plus a front
end) is listed as "Later" under Project 10. Building the lane now would test nothing.

Plan: move this into Project 10's "Done when" as written by Ed: backend tests, an OpenAPI to
TypeScript drift check, front-end lint, typecheck and unit tests, and Playwright against
offline mode covering create game → interpret → commit → reload and resume. Playwright and
Chromium are available in this cloud environment. Until then, the installed-CLI smoke test
in CI-3 covers the same journey.
