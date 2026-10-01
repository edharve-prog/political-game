# Project 4 — Population & Popularity

**Status:** In progress (first pass in review)
**Depends on:** 1, 3   **Provides:** `step_approval`, `target_approval`, `national_approval`, `vote_intention`, `run_election`

## Design (`population/popularity.py`)
Each turn, each group's approval closes 30% of the gap to a target:

- **target** = `lean`
  - plus K × Σ CARES_ABOUT weight × indicator change (in standard steps, against a reference state)
  - plus K × Σ EMPLOYS weight × sector output change
  - plus K × Σ INFLUENCES weight × institution support relative to 0.5 (e.g. media framing)
  - plus active `ApprovalEvent`s, each halving every `half_life_turns`
- K = 0.05, so a one-step move in something a group weighs at 1.0 shifts that group's target by 5 points.
- **Memory:** partial adjustment means shocks are felt over several turns and fade back to lean. Events age each turn and are dropped once negligible.
- **National approval** is weighted by population share. **Vote intention** is weighted by share × turnout.
- **Elections** use a two-party cube law on vote intention and 650 seats. This is an MVP simplification; MRP-lite by geography is an open question in the main plan.

## Model changes
- `Group.lean`: the approval a group drifts back to.
- `ApprovalEvent` and `WorldState.events`: one-off approval hits or boosts that fade over time. They come from outcomes in Project 6 and scandals in Project 12.
- Toy leans: pensioners 0.55, young renters 0.35, public workers 0.45, business 0.50. The toy government starts at about 48% vote intention, which projects to around 287 seats (short of a majority).

## Example: energy tax, magnitude 0.8, after 12 turns (toy)
Compared with doing nothing: pensioners 55.0 → 53.3, public workers 45.5 → 45.2, business 50.0 → 49.7, young renters flat. National approval 46.5 → 45.8, and projected seats 287 → 272.

## Done when
24 turns with no action keep approval stable at lean, and a known-unpopular action moves the expected groups down. Both are met by `tests/test_popularity.py`.

## Open questions
- **Reference state:** judge change against the start, a rolling 12-turn-old snapshot, or expectations? For now the game loop (Project 8) chooses.
- **Partisanship:** should some groups be harder to move? A per-group sensitivity multiplier would do it.
- **Group coverage:** the toy has no explicit low-income group yet; Project 2 adds the real group list.
