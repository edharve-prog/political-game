# Project 3 — Simulation Engine (Propagation)

**Status:** In progress (first pass in review)
**Depends on:** 1   **Provides:** `propagate(state, shocks, horizon, k_draws, seed) -> DeltaDistribution`

## Goal
Given policy actions and external shocks, compute how the world changes.

## Design (first pass, `world/propagation.py`)
- **Linear difference equations on the graph.** Each node has one primary metric (sector output, country growth, indicator value, institution support, group approval).
- **Standard steps.** Propagation runs in normalised units so weights are comparable. One step is 5% of sector output, 1pp of growth, 1pp for `%` indicators, 10 points for index indicators, 0.1 institution support and 0.05 group approval. Results are converted back to native units.
- **Shocks** are impulses in standard steps (`node`, `delta`, `start_turn`, `duration_turns`). Impulses are level shifts and persist.
- **Edges** pass `impulse × weight × DAMPING (0.9)` and land after `lag` turns. Lag-0 edges land the same turn. Impulses below 1e-4 are dropped, and a hop cap guards same-turn loops. With |weight| ≤ 1 every loop decays geometrically.
- **Propagating edges:** DRIVES, SUPPLIES, TRADES_WITH and INFLUENCES. EMPLOYS, CARES_ABOUT and edges into groups are left to Project 4, so approval maths lives in one place.
- **Monte Carlo:** each draw samples every weight from Normal(weight, uncertainty), seeded by `(seed, turn, draw)`. The output is mean, p10 and p90 per node per turn.
- **Actions → shocks:** `actions_to_shocks` maps each PolicyAction kind to a sign. For example, a tax lowers the target's output and spending raises it. Magnitude 1.0 equals a two-step shock, spread over `duration_turns`. This is a placeholder until the Project 5 interpreter emits shocks directly.
- `apply_deltas(state, deltas)` returns a new state with native deltas added, and clamps approval and support to 0..1.

## Toy world changes
- Energy output → energy prices is now **−0.6**: more supply means lower prices, so an energy tax raises prices.
- The central bank's reaction is now `inflation → Bank Rate` (DRIVES, +0.5, lag 1), replacing an INFLUENCES edge into the institution.

## Example: tax on energy, magnitude 0.5 (toy world)
Energy output −£4bn; energy prices +5.3 index points at once; inflation +0.14pp from turn 1, easing slightly once higher rates bite after 6 turns; Bank Rate +6bp from turn 2; house prices −0.28 points from turn 5; unemployment +0.02–0.03pp.

## Done when
A tax rise on energy gives sensibly signed effects on energy prices and inflation with a lagged tail, and results reproduce with a fixed seed. This is met by `tests/test_propagation.py`. The effect on low-income approval is checked in Project 4.

## Open questions
- **Carrying effects across turns.** `propagate` returns a full trajectory from "now". Project 8 must decide whether to keep a log of past shocks and re-simulate, or store pending arrivals in WorldState. Storing pending arrivals is the leading option.
- **Mean reversion.** Shocks currently persist forever. Some metrics, such as energy prices and growth, should probably drift back. Add a per-node reversion rate?
- **Non-linearities** (saturation, thresholds such as a debt crisis). Deferred until backtests in Project 11 show they are needed.
