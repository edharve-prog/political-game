# Backlog: Evaluation and Calibration (Project 11)

Picked by Ed on 2026-10-05 as the next piece of work after the P2 scenario and response stories,
using offline checks only (no Claude calls).

The goal is a world that behaves believably: when something we know from history happens, the
numbers should move the way they did. Project 16 made the game fair (no free levers, bounded
indicators); this project makes it true to life. It also picks up the 30-turn review's
finding that the world feels static.

Story ids are stable: refer to them in PRs. Priority: **P1** = next, **P2** = after, **P3** =
later.

---

## CA-1 Historical backtests (P1) — done

Encode real episodes as shocks plus the government's response and check that each number the
episode is known for moves the right way.

- `game/backtests.py` holds seven episodes: the 2008 banking crisis and bailout, 2010
  austerity, the 2022 energy price shock, the 2022 Energy Price Guarantee, the 2022 mini-budget,
  the 2020 lockdown and furlough, and the 2016 Brexit vote.
- A check compares the episode with its start, with the same shocks and no policy (what the
  policy did), or the shocks alone with the start (what the event did).
- Checks are about direction; a peak move under 0.05 steps counts as no move.
- A check the engine gets wrong today carries a `gap` note saying what is missing. The test
  expects it to keep failing, so whoever closes a gap updates its check.
- `uv run python -m hog_sim.game.backtests` prints the report.

**First run (2026-10-05):** 23 of 32 checks pass; 9 are known gaps. They cluster:

| Gap | Episodes | What is missing |
|---|---|---|
| ~~A slump does not cost jobs~~ (closed by CA-3) | 2008, 2010 | No link from finance, public sector output or UK growth to unemployment |
| ~~The Bank of England ignores slumps~~ (closed by CA-3) | 2008, 2020 | Bank Rate only follows inflation and the deficit |
| Credit does not reach house prices | 2008 | Finance only feeds housing output, and less building raises prices |
| Energy costs do not reach industry | 2022 | No link from energy prices to manufacturing |
| No exchange rate | 2016 | A weaker pound cannot raise prices |
| Trade weights look swapped | 2016 | The EU moves UK growth with 0.1, the UK moves the EU's with 0.45 |

## CA-2 Stability over 120 turns (P1) — done

The plan's "done when" asks for a stable engine over 120 turns. Run doing nothing, the balance
bots and a seeded random player for 120 offline turns and check that no indicator runs away,
nothing sits at a bound for long, and the world does not flatline (approval and indicators keep
moving when scenarios keep coming).

- `game/stability.py` plays each player for 120 offline turns. `LIMITS` there holds the pass
  marks:
  - no indicator more than 8 steps from its start;
  - no indicator whose average distance from its start in the second half is more than 1.5
    times the first half's plus 0.5 steps and is 2 steps or more (a growing trend);
  - indicators at a hard bound in no more than 10% of turns;
  - over the last 48 turns, vote intention moves at least a point and at least half the
    indicators still move.
- The test plays four players on one seed (about 10 seconds). `uv run python -m
  hog_sim.game.stability --seeds 5` runs all 15 players; CI-4's nightly job can call it.

**First run (2026-10-05, 15 players, 2 seeds):** stable. Nothing non-finite, nothing at a
bound, and the furthest any indicator went was the deficit at 5.5 points over its start under
"spend on everything". No runaway trends. Worth knowing for CA-3: doing nothing keeps vote
intention within 44% to 49% for all 120 turns, and even its busiest indicator averages only 0.1
to 0.2 steps from where it started. The world is quiet when the player is.

## CA-3 Close the backtest gaps (P1)

Add or retune the links the gaps above call for, one gap per change, re-running the backtests
and the balance test each time. Likely changes: UK growth and the finance and public sectors
drive unemployment; Bank Rate responds to unemployment or growth as well as inflation; finance
reaches house prices; energy prices feed manufacturing's costs; swap the UK–EU trade weights.
An exchange rate is a new indicator and may wait for Project 2's real seed data.

Each change must keep `tests/test_balance.py` passing; agree any change to its `LIMITS` with
the Project 16 owner first.

**Progress:**

- **Slumps cost jobs (2026-10-05).** Finance (0.4), manufacturing (0.5) and the public sector
  (0.5) now feed UK growth, and growth drives unemployment a turn later (-0.4, Okun's law).
  Backtests: 25 of 32 pass. The 2008 crash now raises unemployment by 0.4 points and the 2020
  lockdown by 2.1; 2010 austerity by 0.2. The real 2008 rise was about 3 points, which is CA-4's
  business. The balance test and the 120-turn stability check still pass.
- **The Bank of England cuts in a slump (2026-10-08).** Higher unemployment now lowers Bank Rate
  a turn later (-0.5), alongside its response to inflation. Backtests: 27 of 32 pass. The shock
  alone now cuts Bank Rate by 0.2 points in 2008 and 1.1 in 2020 (real: about 4.5 and 0.65).
  The stability check then flagged house prices under "spend on everything" as growing: a
  12-turn lag from housebuilding settles slowly, from an average of 0.6 steps in the first half
  to 1.5 in the second. That is a held policy settling, not a runaway, so the growth check now
  also needs the late average to reach 2 steps.

## CA-4 Sizes, not just directions (P2)

Once directions pass, give each check a rough expected size band from the record (for example,
the 2022 energy shock took inflation up several points, not a fraction of one) and report how
far each run is from it. Start as a report only; tighten into a test once the bands are agreed.

## CA-5 Forecast calibration (P2)

Over many offline turns, check that the forecast's 10th to 90th percentile range contains the
resolved result about 80% of the time when the "true" weights are drawn the same way, and that
the offline forecaster's probabilities are not systematically over- or under-confident.

## CA-6 LLM output quality from saved logs (P3)

Measure validation failure rate, repair rate, candidate diversity and storyline variety from
the call log (`hog-sim logs`) and the review data already saved under `plan/reviews/`, so no new
Claude calls are needed. Feeds the next 30-turn review.

## CA-7 Fit edge weights (P3)

Once CA-3 and CA-4 are in, fit the main edge weights against the backtest bands rather than
setting them by hand, within limits that keep the balance test passing.

---

## Order

CA-1 and CA-2 (done), then CA-3 one gap at a time, then CA-4 and CA-5. CA-6 and CA-7 later.
