# Back-end review (2026-10-01)

Reviewed: `main` at `0bd638a` (after PR #14). Scope: the engine (`world/`), popularity
(`population/`), forecasting and selection, the game loop, feasibility, the offline plug-ins
and the knowledge-store reuse path. Prompts and the CLI were read only where they feed the engine.

How it was tested: 143 tests pass. Then 45 full 24-turn offline games (9 scripted strategies
× 5 seeds, scenario library + keyword interpreter + engine forecaster), plus single-action
probes through `resolve()` on a quiet scenario. Both scripts are next to this file and run
from the repo root with `uv run python <path>`:

- `balance_sim.py`: strategy bots, prints vote share, seats, wins and final indicators
- `effect_probe.py`: what one action does to approval, deficit, energy prices and Bank Rate

### Results of the strategy runs

| Strategy (same response every turn) | Avg vote | Seats (range) | Wins /5 | Notable end state |
|---|---|---|---|---|
| "huge deregulation of energy" | **55.1%** | 338–464 | **5** | energy price index 100 → **3.4**, inflation 1.5% |
| first suggested option | 47.5% | 152–355 | 2 | |
| "huge tax on banks" | 45.9% | 184–303 | 0 | deficit **−8.9%** (surplus), Bank Rate 1.0% |
| "give a speech" | 46.3% | 199–291 | 0 | |
| do nothing | 45.0% | 211–255 | 0 | energy prices stay +19 |
| "huge spending on hospitals and schools" | 44.0% | 173–241 | 0 | deficit **17.1%** (limit is 10%) |
| "massive cuts to public spending" | 39.6% | 110–176 | 0 | |
| "huge tax cut for industry" | 39.3% | 99–165 | 0 | unemployment **−2.05%** |
| "massive investment to build homes" | 34.7% | 57–107 | 0 | house price index 100 → 25.9 |

The starting world already loses: vote intention 47.9% gives 284 of 650 seats.

---

## Issues, by severity

### Critical: these break the core loop

**C1. Deregulation is a free, repeatable win.** Only `spend` and `tax` carry a cost (the
deficit, `world/propagation.py:100-132`). `regulate`, `deregulate`, `diplomatic`,
`military`, `communicate`, `legislate` and `appoint` are free. Repeating "huge deregulation of
energy" every turn wins 5 of 5 games with up to 464 seats, while energy prices fall 97%.
Any free lever that moves something voters care about is a dominant strategy.

**C2. Indicators have no bounds and no pull back to normal.** Every shock is a permanent
level shift (`propagation.py:9-10`) and `apply_deltas` (`propagation.py:220-232`) clamps
only approval and support. Nothing returns inflation towards target, unemployment towards
its natural rate, or prices towards trend. The runs ended with unemployment at −2.05%, a
budget surplus of 8.9% of GDP, and house prices down 74%. Repeated actions add up without
limit, which is what makes C1 and the others exploitable.

**C3. Offline play never checks feasibility.** `Game.play_turn` (`game/loop.py:555`) trusts
the interpreter. Only `LLMInterpreter` (`llm/adapters.py:189`) and the stored path of
`StoredInterpreter` run `check_feasibility`. The keyword fallback doesn't. In practice mode the
deficit reached 17.1% even though the limit is 10%. You can also "appoint" at the Bank of
England, which feasibility would block as independent.

**C4. Most scenarios don't touch the world.** Claude-written scenarios can't carry shocks:
`ScenarioDraft` has no `shocks` field (`llm/scenario_gen.py:23-33`). 45 of the 67 library
scenarios have none either. Ignoring a strike or a crisis therefore costs nothing in about
two thirds of offline turns, and in every Claude turn apart from an outcome's small
`new_shocks`. The 22 library scenarios that do have shocks are all bad news, so the world
only ever drifts down. See H5.

### High: wrong or misleading effects

**H1. What an action does depends on which node the interpreter happens to target.** The
effect is sign × magnitude on the target's main metric (`propagation.py:83-95`), so:
- "Cap household bills" becomes `regulate sector:energy`, and energy prices go **up 2.7**.
- "Tax energy windfall profits" makes prices go **up 5.4** and the revenue funds nothing.
- An LLM reading a bill subsidy as `spend indicator:energy_prices` pushes prices **up 10**
  index points.
- A reassuring speech to the Bank of England raises Bank Rate by 0.18pp, and appointing an
  ally raises it by 0.45pp. The cause is that `institution:central_bank → interest_rate`
  is a weight-1.0 DRIVES edge from "support for the government" (`world/seed/toy.py:340`).

The interpreter prompt never says which direction an indicator target moves
(`llm/prompts/interpreter.py:12-23`).

**H2. Policies aimed at a group give a brief bump but cost for good.** A pension rise of 0.5
lifts pensioners by 2.4 points on turn 1, and they are back to baseline by turn 5. The
deficit stays 0.3pp higher for the rest of the game. A group-targeted shock is written
straight into `approval` (`propagation.py:220-232`), then `step_approval` pulls it back to
`lean`. A lasting benefit should shift the group's target, not its current value.

**H3. How long a policy lasts changes nothing.** A 0.5 spend for 1 turn and a 0.5 spend for
12 turns both add exactly +0.3pp to the deficit and have the same total effect
(`propagation.py:115-131` divides the total by the duration). A year-long programme costs the
same as a one-month one.

**H4. Approval events can stack without limit.** Each event is capped at ±0.1, but their sum
isn't (`population/popularity.py:440-444`). A +0.1 event every turn with a half-life of 3
settles at about +0.48 on the target, because 0.1 / (1 − 0.5^(1/3)) ≈ 0.48. In Claude mode, a
generous outcome writer could carry the election this way. This comes from the formula; I
didn't reproduce it with Claude.

**H5. Offline outcomes lean negative every turn.** `EngineForecaster`
(`game/stubs.py:485-503`) offers a −0.03 backlash at 25% and a +0.02 "decisive" outcome at
15% to every exposed group. That is about −0.0045 per turn in expectation. Add the one-way
scenario shocks (C4) and no recovery (C2), and doing nothing loses 5 of 5 games. A tough
start can be a fair design, but right now it happens by accident and should be a decision.

### Medium

**M1. Much of the state does nothing.** These fields never affect the engine or the score:
- Country `relationship` and `stability`
- Sector `sentiment`
- Institution `power` and `independence` (apart from the 0.7 threshold in feasibility)
- `employment_k` and `gdp_bn`
- Feasibility's `resistance`, which is computed and never read

So an LLM `node_attr` graph change (`world/changes.py:25-29`) only changes the briefing text.
A `diplomatic` action moves the partner's `growth_pct`, not its relationship with the UK.

**M2. The chosen outcome barely moves the economy.** `indicator_shifts` are only displayed.
What actually changes the state is `new_shocks` (at most 1 step), group effects and graph
changes. Event tags such as `strike` and `market_selloff` have no mechanical effect, so a
narrative that says "markets sold off" can leave Bank Rate unchanged.

**M3. Graph changes can rewrite who people are.** `edge_weight` changes are allowed on
CARES_ABOUT and EMPLOYS edges (`world/changes.py:82-88`), which contradicts the rule stated
at `changes.py:31-32`. With up to 3 changes a turn for 24 turns, a group's priorities can
drift or flip sign, because the clamp is only ±1.

**M4. Voters never adapt.** Approval is always measured against the turn-0 state
(`game/loop.py:506`). Any change since the start is felt at full strength for ever. This
should be revisited together with C2.

**M5. Propagation silently drops impulses past 200 hops a turn** (`propagation.py:31,
168-169`), and which ones are lost depends on stack order. That's fine for the toy world, but
the bigger graph in SD-8 and Project 2 will hit the cap. It should warn or iterate until the
values settle.

**M6. Resuming wastes a scenario and shows a different one.** `Game.__init__` generates a
scenario (`loop.py:527`) and `resume` generates another (`loop.py:544`), so the first Claude
call is thrown away. The scenario the player was looking at when they quit isn't saved.

### Low

- **L1.** The consistency score's minimum spread of 0.1 native units (`forecasting/scoring.py:387`)
  is very strict for index indicators, where one step is 10 units.
- **L2.** Stored outcomes are reused by kind, target and direction, ignoring size
  (`knowledge/entries.py:81-87`), so a small spend reuses outcomes written for a huge one.
  This was deliberate, but it is worth a size bucket.
- **L3.** Every saved turn stores the full state, including histories that keep growing. The
  save size is fine at 24 turns but grows quadratically, which matters for the 120-turn
  stability test.

---

## Enhancements, by value

These leave out what the backlog already covers: pledges (SD-5), calendar and budgets (SD-6),
outside shocks (SD-7), a bigger world (SD-8), delivery fields (RB-4), new levers (RB-6),
"why did that happen" (TT-2) and fail states (Project 9).

1. **A balance test in CI.** Turn `balance_sim.py` into a test with scripted bots over N seeds
   and assertions such as: no single repeated action wins more than X% of games; doing
   nothing doesn't win; every indicator stays within plausible bounds. This is cheap, and it
   is the safety net for every fix below.
2. **Baseline economy.** Each indicator gets bounds, an equilibrium and a persistence rate,
   so shocks fade unless they are sustained. Spending becomes a flow that costs every turn it
   runs (fixes C2, H3).
3. **Every lever has a cost.** Regulation trims output somewhere; diplomacy and military
   action move relationships. Repeating the same action brings diminishing returns, and a
   small political-capital or legislative-time budget each turn limits how much can be done
   (fixes C1).
4. **Clear direction for actions.** The interpreter states which way the target should move
   ("lower energy prices"), and the engine maps that onto the right node and sign. Also fix
   the Bank of England edge (fixes H1).
5. **Lasting group benefits.** Targeted policies move a group's target for as long as they run
   (fixes H2).
6. **Event tags have consequences.** A small table maps them to shocks, for example strike →
   public output, market_selloff → Bank Rate and deficit, capital_flight → finance (M2).
7. **Make the unused state matter.** Relationship scales trade edges; stability raises the
   odds of outside shocks; sector sentiment feeds output; legislature support moves with
   backbench rebellions and approval; resistance feeds rebellion odds (M1).
8. **Crises cost something if ignored.** Every scenario carries a shock, including in Claude
   mode, and the shock grows each turn it is left unaddressed (C4).
9. **A debt stock and interest bill.** Interest rate × debt feeds back into the deficit, so
   borrowing compounds and high rates hurt.
10. **Richer approval.** Issue salience (groups weigh what is in the news), habituation, and
    turnout that moves with enthusiasm.
11. **A better election model.** Multi-party vote shares with swing by group, and noisy polls
    shown to the player instead of exact vote intention.

## Proposed first batch

1. Enhancement 1, the balance test, first, so every change after it can be measured.
2. C3: run feasibility inside `Game.play_turn`. Small and contained.
3. C1, C2 and H3 together through enhancements 2 and 3: bounds, mean reversion, spending as
   a flow, and costs for free levers.
4. H1 and H2: action direction, the Bank of England edge, and lasting group benefits.

C4 (scenario shocks), H4 (a cap on summed events) and the medium items come next.

## Proposed additions to the plan

- A new **Project 16, Engine balance and economic realism**, holding C1–C3, H1–H5, M1–M5
  and enhancements 1–7 and 9.
- Extend SD-1 or SD-7 with "crises cost something if ignored" (C4, enhancement 8).
- Extend Project 4 with enhancements 10 and 11.
- Extend Project 8 with M6 (resume) and L3 (save size).
