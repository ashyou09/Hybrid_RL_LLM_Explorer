---
title: Hybrid RL-LLM Explorer
emoji: 🧠
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# Hybrid RL-LLM Agentic Explorer (Exp-1)

This document serves as the high-level master plan and explanation for **Exp-1: Hybrid RL → LLM → Smart Explorer**. It covers the architecture, the 3-phase flow, and the technical logic guiding our AI agent to navigate dynamic hazards without relying on positional memory.

## 🎯 Experiment Goal
The objective is to train an agent to survive a deadly grid environment (`MiniGrid-LavaRoom-v0`) using a hybrid approach:
1. **Reinforcement Learning (RL)**: Used for local, greedy exploration and penalty avoidance.
2. **Large Language Models (LLM)**: Used for high-level semantic generalization (e.g., learning that "red lava is bad," rather than "coordinate x:3, y:4 is bad").

## 🛠️ Step-by-Step Architecture

### 1. Configuration UI (Pre-Flight)
Before the game begins, a `Pygame` configuration screen is presented. The researcher can dynamically configure the difficulty boundaries across phases:
* **Deaths to Collect**: How many fatal errors the Phase 1 agent must commit before the learning phase concludes.
* **Auto-Increase Lava (+1 per death)**: A toggle that automatically scales difficulty. Every time the agent learns a fatal lesson, the number of lava tiles on the map increments by 1.
* **Phase 3 Episodes**: How many times the final, "smart" agent must prove it can survive.
* **Phase 3 Lava Tiles**: The number of obstacles active during the final testing phase.

---

### 2. Phase 1: RL Learning (`RLExplorer`)
The agent begins entirely blind with no rules. It navigates using a pure **greedy penalty system**. 
* Every step costs `-3` penalty.
* Returning to a visited tile costs extra (accumulating higher negative penalties) to push the agent outwards.
* **The "Stay" Penalty**: If the agent gets stuck bumping into walls or rotating in place, an exponential stay-penalty builds up. If a tile reaches `-60`, an **Auto-Death** kill-switch is triggered to prevent infinite loops.
* **Learning Log**: Whenever the agent steps into lava or auto-dies, it writes a purely semantic JSON fact to `learning_log.json` (e.g., `"tile_death — red lava"`). Positional coordinates `(x, y)` are deliberately omitted to force semantic generalization.

---

### 3. Phase 2: LLM Rule Synthesis (`reflection_engine.py`)
Once enough deaths are logged, the terminal freezes and the log is passed to a local LLM (`llama3.2:3b`).
* The LLM synthesizes natural-language rules based on the JSON facts.
* It outputs structured JSON validating exactly what triggered the death and what the agent must do to avoid it.
* **Memory Hub**: These generated rules are stored permanently in a **ChromaDB vector database** (`.chroma_db`).

---

### 4. Phase 3: Smart Explorer (`OnlineExplorerAgent`)
The agent is placed into **brand new, randomly generated lava layouts**. Positional memorization will fail here. The agent must use the learned rules.
* The agent evaluates all 4 valid directions: `Front`, `Left`, `Right`, and `Behind`.
* **Memory Lookups**: Before picking a path, the agent queries the ChromaDB `MemoryHub` with the color and type of the adjacent tiles (e.g., `"red lava"`).
* **The Wall Hallucination Fix**: Empty floor spaces (`None` in the MiniGrid backend) correctly bypass the MemoryHub and evaluate strictly to their numeric RL penalty.
* **Rule Blocking**: If the MemoryHub matches a tile to a fatal LLM rule, that specific tile receives a massive artificial constraint penalty (`-99999.0`).
* **Greedy Escape**: The RL agent selects the direction with the highest penalty score (closest to 0). Finding that lava is `-99999`, it effortlessly turns away and proceeds onto safer, less penalized paths.

## 🐛 Recent Iterations & Bug Fixes
* **The Infinite Hump**: Fixed an issue where the agent queried the MemoryHub *only* for the tile in front of it. When turning sideways to avoid a trap, it evaluated the trap's penalty as `0` and oscilated back toward it in an infinite loop. We fixed this by forcing the database lookup on **all 4 adjacent tiles simultaneously**.
* **NoneType Wall Hallucinations**: Fixed a defect where unvisited empty spaces evaluated to `None`, which the logic falsely parsed as a fatal out-of-bounds wall (`-99999`). This forced the agent to walk directly into lava because everything around it appeared identically fatal. Proper bounds-checking was restored.
