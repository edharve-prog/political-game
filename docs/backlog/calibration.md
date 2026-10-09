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
| ~~Credit does not reach house prices~~ (closed by CA-3) | 2008 | Finance only feeds housing output, and less building raises prices |
| ~~Energy costs do not reach industry~~ (closed by CA-3) | 2022 | No link from energy prices to manufacturing |
| ~~No exchange rate~~ (closed by CA-3) | 2016 | A weaker pound cannot raise prices |
| ~~Trade weights look swapped~~ (closed by CA-3) | 2016 | The EU moves UK growth with 0.1, the UK moves the EU's with 0.45 |

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

## CA-3 Close the backtest gaps (P1) — done

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
- **Credit reaches house prices (2026-10-08).** Finance output now drives house prices a turn
  later (0.4): when banks lend more, buyers can borrow more. Backtests: 28 of 32 pass. The 2008
  shock alone now takes about 11 points off the house price index (real: roughly 15% to 20%).
- **Energy costs hit industry (2026-10-08).** Energy prices now cut manufacturing output a turn
  later (-0.2). Backtests: 29 of 32 pass. The 2022 shock alone takes 2.7% off manufacturing
  output (real: energy-intensive industry fell far more, manufacturing as a whole a few
  percent). Energy levers (deregulating, capping bills) now also help industry; the balance
  test is still within limits, with no single lever winning more than 1 game in 5.
- **UK–EU trade weights swapped (2026-10-08).** The EU now moves UK growth with weight 0.45
  and the UK moves the EU's with 0.1, matching the EU taking over 40% of UK exports. Backtests:
  30 of 32 pass. The Brexit episode now takes 0.17 points off UK growth.
- **EU demand reaches UK factories (2026-10-08).** EU growth now drives UK manufacturing a turn
  later (0.3): the EU takes about half of UK goods exports. Backtests: 31 of 32 pass. Brexit now
  takes 0.12 steps off manufacturing output. Courting the EU helps industry too; the balance
  test is still within limits. Left: the exchange rate, a new indicator.
- **The pound (2026-10-08, picked by Ed).** A new indicator, the sterling exchange rate (an index
  starting at 100). UK growth (0.3) and Bank Rate (0.2) lift it and the deficit (-0.4) weighs on
  it, all in the same turn; a weaker pound raises inflation two turns later (-0.3). Markets set
  it, so no policy aims at it directly. The Brexit episode now carries the pound's overnight fall
  as a shock (-15 index points), which raises inflation by 0.4 points (real: about 2). Three new
  checks: the pound falls in 2008 (it does, by 3 points against a real 20 to 30), after Brexit,
  and after the mini-budget. The mini-budget one is a new known gap: the tax cut's lift to growth
  and rates outweighs the borrowing, so the pound rises slightly. Backtests: 34 of 35 pass. The
  balance test and the stability check still pass.

## CA-4 Sizes, not just directions (P1, under way)

Once directions pass, give each check a rough expected size band from the record (for example,
the 2022 energy shock took inflation up several points, not a fraction of one) and report how
far each run is from it. Start as a report only; tighten into a test once the bands are agreed.

- 25 of the 32 checks now carry `real`, a rough range from the record for the episode's first
  year, in the number's own units. The ranges are first guesses and want Ed's eye before they
  become a test.
- `uv run python -m hog_sim.game.backtests` prints a second table: the engine's move, the real
  range, and the move as a share of the range's middle.

**First run (2026-10-08):** 6 of 25 sized checks fall inside their range, and the middle engine
move is 38% of the real one. The engine is mostly too timid:

| Pattern | Share of real | What it suggests |
|---|---|---|
| Bank Rate barely reacts to a crash or an energy shock | 5% (2008), 15% (2022) | Its links from inflation and unemployment are too weak |
| Energy shocks barely reach inflation | 15% (2022), 14% (price guarantee) | The energy-to-inflation link is too weak, or the shock too small (+30 index points against a real +50 to +180) |
| Every deficit move is exactly 1 point | 10% to 67% | The deficit only follows spending decisions; slumps do not widen it (no automatic stabilisers) |
| 2008 unemployment | 15% | Okun's link is too weak, or the finance shock too small |
| The mini-budget's poll hit | 17% | Markets turning on a government should cost more votes |

A few go too far: the 2010 public spending cut takes 6% off public sector output (real: 1% to
4%), and the 2020 lockdown raises unemployment 2.1 points (real: 0.5 to 1.5) because furlough
does too little (14% of the job losses it really prevented). The 2020 Bank Rate cut is also too
big, but the real one was limited by rates already being near zero.

**Progress:**

- **Energy shocks at real size (2026-10-08).** The 2022 episodes now put energy prices up 80
  index points, inside the real rise, instead of 30. With the old weights inflation then rose
  only 2.2 points and manufacturing fell 7% (real: 4 to 7 points and 1% to 5%). So energy prices
  now feed inflation twice as hard (0.6, was 0.3) and manufacturing half as hard (0.1, was 0.2).
  Bank Rate then overshot house prices, so its pull on them is halved (0.25, was 0.5). The 2022
  shock now raises inflation 4.3 points and Bank Rate 1.9, takes 4.4 points off house prices and
  3.6% off manufacturing, all in or near their ranges. 10 of 28 sized checks are in range (was
  7). Directions unchanged (34 of 35); the balance test and stability check still pass.
- **Slumps widen the deficit (2026-10-08).** Slower UK growth now raises the deficit a turn later
  (0.5): tax receipts fall and benefit bills rise, the "automatic stabilisers". That made Bank
  Rate rise in the 2008 crash, because borrowing pushed it up harder than the slump pulled it
  down, so the deficit's pull on Bank Rate is halved (0.15, was 0.3): Bank Rate is set by the
  Bank, and borrowing mostly shows in gilt yields, which the game does not have. The 2020
  lockdown now widens the deficit by 3.7 points (was 1.0; real 8 to 13) and the 2020 Bank Rate
  cut is in range. In play, spending now partly pays for itself through growth: a full-size
  public spending rise lifts the deficit about 0.8 points after a turn, not 1.25. Directions
  unchanged (34 of 35), 10 of 28 sizes in range; the balance test and stability check still
  pass. Still timid: Bank Rate in 2008 (the crash shock only takes 1.1 points off growth,
  against a real 4 to 6) and the mini-budget's poll hit.
- **2008 as a full recession (2026-10-08).** The 2008 episode only hit the banks, so growth fell
  1 point against a real swing of about 7 (from +2.4% in 2007 to -4.6% in 2009). It now also
  carries the world recession that came with it (-5 points of UK growth). And the Bank of England
  now cuts twice as hard when unemployment rises (1.0 per point, was 0.5). The crash alone now
  raises unemployment 2.3 points (real 2 to 3.5), widens the deficit 3.6 points (real 4 to 8),
  takes 25 points off the pound (real 20 to 30) and cuts Bank Rate 1.7 points (real about 4.5,
  when the Bank also cut in one go and began printing money). 12 of 28 sizes are in range (was
  10). The 2020 Bank Rate cut now overshoots, at 1.75 points against a real 0.65: the game
  starts Bank Rate at 4%, but in 2020 it was already at 0.75% and could fall no further.
  Directions unchanged (34 of 35); the balance test and stability check still pass.
- **Markets lose faith (2026-10-08, picked by Ed).** A new rule: an unfunded tax cut worth at
  least 0.75 points of deficit a turn makes Bank Rate jump (0.4 per point) and the pound fall (8
  index points per point), and costs every group 0.12 approval per point. Spending is not
  judged this way. The mini-budget now pushes Bank Rate up 0.95 points (real 0.25 to 1), the
  pound down 6 points (real 2 to 8) and vote intention down 8 points (real 8 to 12), which
  closes the last known gap: 35 of 35 backtests pass. 14 of 28 sizes are in range (was 12). The
  balance test and stability check still pass.
- **Job protection (2026-10-09, picked by Ed).** Spending aimed at unemployment is now a wage
  subsidy, like furlough. It does not push unemployment down; while it runs it holds back 80% of
  any rise in unemployment at full size (40% at half size), for shocks that land while it runs,
  and costs the deficit like any other spending. The 2020 episode now uses it (full size, 12
  turns) instead of spending on manufacturing: unemployment rises 0.5 points (real 0.5 to 1.5,
  was 2.1) and 2 points fewer than with no scheme (real 2 to 6, was 0.6). 16 of 28 sizes are in
  range (was 14). The balance test and stability check still pass.

Next: tune one pattern at a time towards its range, keeping the direction checks, the balance
test and the stability check passing, then agree the ranges and make them a test.

Still off, each needing a new mechanism rather than a weight: the 2010 cuts shrink public output
too much for the deficit they save (the balance between a spending change's output and budget
effects), and the 2020 Bank Rate cut overshoots because the game starts Bank Rate at 4%.

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

CA-1, CA-2 and CA-3 (done), CA-4 (under way), then CA-5. CA-6 and CA-7 later.
