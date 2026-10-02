# Backlog: Engine Balance and Economic Realism (Project 16)

Source: the back-end review of 2026-10-01 ([review/backend-review.md](../review/backend-review.md)).
In 45 scripted 24-turn offline games, repeating "huge deregulation of energy" won every
time: the energy price index fell from 100 to 3.4. Spending pushed the deficit to 17%,
building homes cut house prices to a quarter, and the starting position loses an election
when nothing is done. Ed approved Project 16 on 2026-10-01. Issue ids (C1–C4, H1–H5,
M1–M6, L1–L3) and enhancement numbers refer to the review.

## First batch (done in PR #16)

### EB-1 Balance test in CI (enhancement 1)
- Turn `balance_sim.py` into a pytest. It runs scripted strategy bots over several seeds,
  using the offline plug-ins only, and must stay fast enough for CI.
- It fails if any single repeated action wins more than an agreed share of games, if doing
  nothing wins, or if any indicator leaves its plausible bounds.
- The thresholds live in one place, and the test prints a results table like the review's.

### EB-2 Feasibility in offline play (C3)
- `Game.play_turn` runs `check_feasibility` on the interpreter's actions in every mode.
  Blocked actions are dropped before `resolve()` and reported to the player.
- Replay stays exact, because the log stores the actions that were actually applied.

### EB-3 Bounded economy with real costs (C1, C2, H3; enhancements 2 and 3)
- Each indicator gets bounds, an equilibrium and a persistence rate. Shocks fade unless
  they are sustained.
- Spending is a flow: it costs every turn it runs, so a policy's duration matters.
- Every lever has a cost. Regulation trims output somewhere, and diplomacy and military
  action move relationships. Repeating the same action brings diminishing returns.
- A small political-capital or legislative-time budget each turn limits how much the player
  can do.
- Done when EB-1 passes, no indicator leaves its bounds in 120 turns, and the stability
  tests still pass.

### EB-4 Actions move indicators the right way (H1, H2; enhancements 4 and 5)
- The interpreter states which way the target should move, for example "lower energy
  prices". The engine maps that intent onto the right node and sign.
- The Bank of England edge is fixed.
- A targeted group policy moves that group's target approval for as long as it runs,
  instead of a brief bump followed by a lasting cost.
- Done when a probe of each action kind moves its intended indicator in the stated
  direction (`effect_probe.py` as a test).

## Remaining (backlog, in rough value order)

- **EB-5 Event tags have consequences** (M2, enhancement 6). A table maps tags to shocks, for
  example strike → public output, market_selloff → Bank Rate and deficit, capital_flight →
  finance.
- **EB-6 Cap on summed approval events** (H4).
- **EB-7 Offline outcomes stop leaning negative** (H5).
- **EB-8 Unused state matters** (M1, enhancement 7). Relationships scale trade, stability
  raises the odds of outside shocks, sector sentiment feeds output, legislature support moves
  with rebellions, and resistance feeds rebellion odds.
- **EB-9 Debt stock and interest bill** (enhancement 9). Interest rate × debt feeds the
  deficit.
- **EB-10 Graph-change guardrails** (M3). Stop `edge_weight` changes from redefining group
  identity edges.
- **EB-11 Voters adapt** (M4). Approval is measured against a moving reference, not turn 0.
- **EB-12 Propagation limit** (M5). Make the 200-hop cut-off explicit, log it, and test it.
- **EB-13 Forecast scoring nits** (L1, L2). Revisit the consistency minimum spread, and make
  outcome reuse take size into account.

## Items placed in other projects

- **C4 and enhancement 8 (crises cost something if ignored)** join SD-1 and SD-7 in
  [scenarios-and-responses.md](scenarios-and-responses.md). Every scenario, including in
  Claude mode, carries a shock that grows each turn it is left unaddressed.
- **Enhancements 10 and 11 (richer approval, better elections)** join Project 4: issue
  salience, habituation, turnout driven by enthusiasm, multi-party vote shares with swing by
  group, and noisy polls.
- **M6 and L3 (resume shows a different scenario, saves grow)** join Project 8.
