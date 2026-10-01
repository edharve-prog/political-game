# Project 15 — LLM Knowledge Store

**Status:** In review (PR, stacked on #11)
**Depends on:** 1 (WorldState, graph), 3 (engine), 5 (LLM layer), 6 (forecasting), 8 (saves), 13/SD-3 (scenario library)
**Provides:** `KnowledgeStore` (SQLite), `GraphChange` + `validate_graph_changes()` / `apply_graph_changes()`, `Outcome.graph_changes`, `recall()` precedents for prompts, `StoredInterpreter` and `StoredForecaster` for offline play, `KnowledgeStore.library_scenarios()` for SD-3's `ScenarioLibrary`

Raised by Ed on 2026-10-01: "a way of storing LLM responses, especially in a form which is
interpreted by the LLM to update the graph network and any DB, in order to be used in
determining future responses, or runs of the Offline game."

## Goal

Every validated thing Claude writes during a game (scenarios, interpretations of the
player's words, candidate outcomes, and proposed changes to the world graph) is kept as
structured data, so that:

1. it can change the world graph, through the engine and never around it;
2. later Claude prompts can draw on it (precedents from earlier turns and games);
3. offline games can reuse it, so offline play gets Claude-written content without Claude.

## Scope

**In**
- A knowledge store in SQLite, in the same file as game saves by default.
- Harvesting each Claude turn into the store after it is saved.
- Graph changes: a schema the LLM can propose in candidate outcomes, deterministic
  validation (ids, whitelisted fields, caps), and application inside `resolve()` so replay
  stays exact.
- Recall: retrieving relevant precedents and adding them to the scenario and outcome prompts.
- Offline reuse: stored interpretations and outcomes used by offline plug-ins; stored
  scenarios exported as `LibraryScenario` records for SD-3's library.
- CLI: `hog-sim knowledge stats | export | import | export-scenarios`.

**Out**
- Carrying graph changes from one game into the start state of another. They are stored and
  counted, but turning them into permanent edge weights is calibration work (Project 11) and
  needs a human review step.
- Embeddings / vector search. Retrieval is deterministic token and node overlap, which is
  enough at this scale and keeps tests stable.
- Selection of offline scenarios (SD-3 owns it).

## Design

### Principle
Same rule as the rest of the game: the LLM proposes, the engine decides. The store only ever
holds output that already passed `structured_call` validation and the semantic checks, and a
graph change only reaches `WorldState` through `apply_graph_changes()` inside `resolve()`.

### What is stored

One table, one row per entry, with a typed, versioned Pydantic payload per kind:

| kind | key | payload |
|---|---|---|
| `scenario` | scenario id | `LibraryScenario` (SD-3 model) with `origin="llm"` and provenance |
| `interpretation` | scenario id + normalised response | response text, actions, unmapped, clarifying question |
| `outcome` | scenario id + action signature | actions, all candidates with scores, chosen index |
| `graph_change` | game id + turn + index | the change, status (applied / rejected) and the reason |

Every payload carries provenance: game id, turn, model(s), prompt version(s), created time.
Payloads are re-validated on read; a row that no longer parses (old schema) is skipped and
logged, never fed to the engine.

```sql
CREATE TABLE knowledge (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    scenario_id TEXT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (kind, key)
);
```

Why SQLite in the save file: one file for Ed to keep or delete, transactions, no new
dependency. JSONL export/import exists so curated knowledge can be committed to the repo and
loaded on a fresh install.

### Capture
Harvesting reads the `TurnRecord` after `SaveStore.save_turn`, so it needs no hook in the LLM
client and stores exactly what the game accepted. The CLI calls `harvest_turn()` only in Claude
mode and passes the models and prompt versions from the client's usage log. The scenario is
taken from the in-memory record, so a `GeneratedScenario` keeps its stakeholder positions in
the store even though the save file still drops them.

### Graph changes (how LLM output updates the network)
`core/models.py` gets:

```python
class GraphChange(Model):
    kind: Literal["edge_weight", "add_edge", "node_attr"]
    source: str | None      # edge changes
    target: str | None
    edge_kind: EdgeKind | None
    node: str | None        # node_attr
    attr: str | None        # whitelisted per node kind
    delta: float
    reason: str
```

- Whitelisted attributes: country `relationship`, `stability`; institution `support`,
  `independence`; sector `sentiment`. Values clamp to the field bounds.
- Caps: |delta| ≤ 0.2 for attributes; an edge weight moves by at most 0.1 or 25% of its size,
  whichever is larger; a new edge starts at |weight| ≤ 0.3 with lag ≤ 6; at most 3 changes per
  outcome; no new edge between nodes that already have one of that kind.
- `CandidateDraft` gains `graph_changes`, checked in `check_candidates` (unknown ids and cap
  breaches are sent back to the model). `Outcome.graph_changes` defaults to empty, so old saves
  load unchanged.
- `resolve()` applies the chosen outcome's changes after propagation. They are in the log, so
  `replay()` reproduces them exactly.

### Recall (reuse in future prompts)
`recall(store, state, scenario=None, actions=None, k=3)` scores stored entries by overlap with
what is happening now (stressed indicators, angry groups, the scenario's affected nodes, the
action targets, title tokens) and returns up to k short lines such as:

> Precedent: "Rail unions ballot for strikes". Response: fund a 4% pay rise. Outcome: strike
> called off, deficit up.

`StateSummary` gets optional `precedents` and `links` lists (links: the graph edges touching
the scenario and the action targets, so the outcome call can propose `edge_weight` changes to
edges it can see), shown in prompts as a separate section only
when non-empty (so existing cassettes keep their keys). The scenario generator uses it for
variety and continuity; the outcome forecaster uses it as a reference class. Precedents from
the current game come first.

### Offline reuse
- **Scenarios:** `KnowledgeStore.library_scenarios()` returns `LibraryScenario` records
  (`origin="llm"`, id `llm-<slug>-<hash8>`, category and conditions filled in).
  `ScenarioLibrary.load()` (SD-3) merges them with the handwritten ones and handles selection.
- **Interpretations:** `StoredInterpreter` reuses a stored interpretation for the same scenario
  when the response matches exactly or closely (token overlap ≥ 0.6), re-checks the targets
  against the current world and feasibility, and otherwise falls back to `KeywordInterpreter`.
- **Outcomes:** `StoredForecaster` reuses stored candidates for the same scenario and the same
  action signature (kind, target, sign), with the engine's current deltas replacing the stored
  numbers and graph changes re-validated; otherwise it falls back to `EngineForecaster`.

## Interface

```python
class KnowledgeStore:
    def __init__(self, path: str | Path): ...
    def harvest_turn(self, game_id: str, record: TurnRecord, state_before: WorldState,
                     provenance: Provenance | None = None) -> int: ...
    def library_scenarios(self) -> list[LibraryScenario]: ...
    def interpretations(self, scenario_id: str | None = None) -> list[InterpretationEntry]: ...
    def outcomes(self, scenario_id: str | None = None) -> list[OutcomeEntry]: ...
    def graph_changes(self) -> list[GraphChangeEntry]: ...
    def export_jsonl(self, path) / import_jsonl(self, path) / export_scenarios(self, path) -> int

    def stats(self) -> dict[str, int]: ...

def validate_graph_changes(state, changes) -> list[str]
def apply_graph_changes(state, changes) -> WorldState
def recall(store, state, scenario=None, actions=None, *, k=3, game_id=None,
           max_other_games=1, skip_recent_turns=0) -> list[str]
class Recaller  # binds a store to the game in progress for llm_plugins(recaller=...)
def scenario_id(scenario: Scenario) -> str
class StoredInterpreter(Interpreter); class StoredForecaster(Forecaster)
```

## Tasks
- [x] `GraphChange`, validation, application in `resolve()`; `Outcome.graph_changes`
- [x] `CandidateDraft.graph_changes` + checks; outcomes prompt mentions the field
- [x] `KnowledgeStore` with schema, harvest, typed reads, JSONL export/import, stats
- [x] `library_scenarios()` against SD-3's `LibraryScenario`
- [x] `recall()` and `StateSummary.precedents`, wired into scenario and outcome calls
- [x] `StoredInterpreter`, `StoredForecaster`; offline CLI uses them when the store has entries
- [x] CLI harvesting in Claude mode, `hog-sim knowledge ...` commands
- [x] Tests: below

## Tests / Done when
- A FakeClient Claude game of 6 turns fills the store with scenarios, interpretations,
  outcomes and graph changes, all re-validating on read.
- An invalid graph change (unknown node, field off the whitelist, over the cap) is rejected by
  `check_candidates` and never reaches the state; a valid one changes the edge or attribute,
  and `replay()` reproduces the final state exactly.
- The turn N scenario prompt contains a precedent from an earlier turn when relevant, and no
  precedent section when the store is empty.
- An offline game started after a Claude game reuses a stored interpretation and stored
  outcome narratives for a matching scenario and response, with no LLM calls.
- Exported JSONL imports into a fresh store with identical entries; `library_scenarios()`
  output loads in SD-3's `ScenarioLibrary`.

## Decisions
- 2026-10-01: Store validated structured output, not raw text. Harvest from the `TurnRecord`
  rather than wrapping the client, so only accepted output is kept.
- 2026-10-01: One SQLite file shared with saves; JSONL for moving knowledge between installs.
- 2026-10-01: Graph changes are bounded, whitelisted and applied in `resolve()`; cross-game
  carry-over of graph changes is out of scope until Project 11 can review them.
- 2026-10-01: Stored scenarios feed SD-3's `ScenarioLibrary` as `LibraryScenario` records
  (`origin="llm"`), not a second library.

## Open questions
- Should Ed be able to curate (approve or delete) stored entries before they are used offline?
  Proposed: `hog-sim knowledge` gets `list` and `drop <id>` later if needed.
- Should precedents from other games ever be withheld from Claude to avoid repetition? Recall
  currently prefers the current game and caps cross-game precedents at one per prompt.
