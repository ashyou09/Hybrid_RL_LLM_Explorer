# v0 Demo — How the Original Prototype Works

This is the first prototype (`src/`, `run_experiment.py`, `app.py`). It is kept
working as a demo. The research code for Levels 1–6 is being built separately in
`research/`. Known limitations of this prototype are listed in
[future_plan/01_current_state_review.md](../future_plan/01_current_state_review.md).

## Goal

Survive `MiniGrid-LavaRoom-v0` using:

1. **Reinforcement-style exploration** for local, greedy penalty avoidance.
2. **An LLM** for semantic generalisation ("red lava is bad", not "tile (3,4) is bad").

## Flow

### 1. Configuration screen (Pygame)

- **Deaths to collect** — how many fatal events Phase 1 must log.
- **Auto-increase lava** — adds one lava tile after each death.
- **Phase 3 episodes** and **Phase 3 lava tiles**.

### 2. Phase 1 — Exploration (`src/core/rl_core.py`, `RLExplorer`)

- Every step costs −3; revisiting a tile adds more penalty.
- Staying on a tile (spinning, bumping walls) grows an extra "stay" penalty;
  at −60 on one tile the agent is force-killed (auto-death).
- Each death writes a **semantic** fact to `learning_log.json`
  (e.g. `"tile_death — red lava"`), without coordinates.

### 3. Phase 2 — LLM rule synthesis (`src/agents/reflection_engine.py`)

- The log is sent to a local LLM (`llama3.2:3b` via Ollama).
- The LLM returns JSON rules: `rule`, `forbidden_action`, `trigger_feature`.
- If the LLM is unavailable, a deterministic fallback builds the rules.
- Rules are stored in a ChromaDB vector store (`src/core/memory_hub.py`).

### 4. Phase 3 — Smart explorer (`src/agents/planner_agent.py`)

- New random lava layouts every episode.
- The agent checks all four neighbouring tiles against the memory hub
  (embedding similarity ≥ 0.65).
- A matched dangerous tile gets a −99 999 penalty, so the greedy choice turns away.

## Bug fixes in v0

- **Infinite hump:** checking only the front tile let the agent turn back toward the
  trap; fixed by checking all four neighbours.
- **`None` wall hallucination:** empty cells (`None` in MiniGrid) were treated as
  walls; fixed with proper bounds checks.
