# Shared Memory Framework: RL-LLM Episodic Learning

## Overview

This document describes three integrated research features implementing a **shared memory layer** between a Reinforcement Learning agent and a Large Language Model (LLM). The system treats LLM consultation as a strategic advisor that learns from the exploration agent's path history, revisit patterns, and repeated failures.

**Research Question:** *Can an LLM, by analyzing exploration trajectories and failure patterns across multiple episodes, help a non-RL agent escape local optima and navigate more strategically?*

---

## Feature 1: Revisit Penalty System

### Motivation
Exploration efficiency decreases when an agent revisits the same tiles repeatedly. By penalizing revisits, we:
- Signal inefficiency to the LLM for real-time guidance
- Create a learnable signal that reveals when the agent is stuck in loops
- Give the LLM actionable data to distinguish between productive exploration and circular motion

### Implementation

**Location:** `src/agents/planner_agent.py` — `_apply_revisit_penalty()`

**Grace Period:** 10 steps (after 10 steps, revisits incur penalty)

**Penalty Formula:**
```
penalty = -1.0 × revisit_count

where revisit_count = number of times the agent has entered this cell previously
```

**Cumulative Tracking:**
```python
self._episode_penalty += penalty  # running total stored for LLM payload
```

### Behavior Example

```
Step 5:   Agent at (2,3) — first visit → no penalty
Step 10:  Agent at (2,3) again → revisit_count=2 → penalty=-2.0
Step 15:  Agent at (2,3) again → revisit_count=3 → penalty=-3.0
Step 20:  Agent at (2,3) again → revisit_count=4 → penalty=-4.0

Cumulative penalty = -2.0 + (-3.0) + (-4.0) = -9.0
```

### LLM Integration
The cumulative `_episode_penalty` is sent to the LLM every 10-15 steps:
- If penalty > -500: LLM receives it as context for navigation hints
- If penalty < -500: Agent force-stops (too stuck; time to analyze)

**Console Output:**
```
[⚠ Revisit #1] pos=(2,3) | penalty=-1.0, cumulative=-1.0
[⚠ Revisit #2] pos=(2,3) | penalty=-2.0, cumulative=-3.0
[⚠ Revisit #3] pos=(2,3) | penalty=-3.0, cumulative=-6.0
```

---

## Feature 2: Periodic LLM Navigation Consultation

### Motivation
Rather than waiting for failure, the system proactively consults the LLM every 10-15 steps to:
- Identify emerging loop patterns early
- Suggest optimal next moves based on trajectory analysis
- Store intermediate navigation rules in ChromaDB for future use

### Implementation

**Location:** `src/agents/planner_agent.py` — `_should_consult_llm()` and `_consult_llm()`

**Schedule:**
- Randomized interval: 10-15 steps
- Triggered when: `step_count >= _next_llm_at` AND `unique_tiles_visited >= 10`

### Data Payload Sent to LLM

```python
{
    "environment": "MiniGrid-CombinedTesting-v0",
    "total_steps": 47,
    "unique_tiles_visited": 18,
    "accumulated_revisit_penalty": -12.5,
    "top_revisited_cells": [
        ((4, 3), 5),    # cell (4,3) visited 5 times
        ((3, 3), 4),    # cell (3,3) visited 4 times
        ((2, 3), 3)     # cell (2,3) visited 3 times
    ],
    "trajectory": [
        "  step 40: pos=(2,4) front='red lava' action=TurnRight revisit=False",
        "  step 41: pos=(2,4) front='wall' action=TurnRight revisit=False",
        "  step 42: pos=(2,4) front='empty space' action=MoveForward revisit=False",
        "  step 43: pos=(3,4) front='wall' action=TurnLeft revisit=False",
        "  step 44: pos=(3,4) front='empty space' action=MoveForward revisit=False",
        "  step 45: pos=(4,4) front='empty space' action=MoveForward revisit=False",
        "  step 46: pos=(5,4) front='revisited tile' action=TurnRight revisit=True",
        "  step 47: pos=(5,4) front='sand' action=TurnRight revisit=False"
    ]
}
```

### LLM Prompt

```
"You are a navigation advisor for an autonomous grid-world agent.
Environment: MiniGrid-CombinedTesting-v0
Steps taken: 47, Unique tiles visited: 18
Revisit penalty accumulated: -12.5 (each revisit past step 10 costs −1 × revisit_count)
Most revisited cells: pos(4,3)×5, pos(3,3)×4, pos(2,3)×3

Recent trajectory (step: pos, front tile, action taken, revisit?):
[last 10 steps shown]

Based on the trajectory and penalty, give ONE concise navigation rule
to help the agent escape loops and explore more efficiently.
Output ONLY a valid JSON object with exactly these keys: 
rule, forbidden_action, trigger_feature."
```

### LLM Response Example

```json
{
    "rule": "Avoid entering cells already visited 3+ times; they lead to loops",
    "forbidden_action": "Move Forward",
    "trigger_feature": "high_revisit_hotspot"
}
```

### Storage & Reuse
The returned rule is **embedded and stored in ChromaDB**:
- Future queries for "high_revisit_hotspot" will match semantically similar triggers
- Multiple navigation hints can coexist with hazard-avoidance rules (from Phase 1 & 2)

**Console Output:**
```
[🧠 LLM Consult #1] step=47, unique_cells=18, penalty_so_far=-12.5
  [LLM] Asking llama3.2 for navigation hint (step 47)…
  [LLM Hint] Rule received: Avoid entering cells already visited 3+ times; they lead to loops
  [Memory Hub] Stored rule for 'high_revisit_hotspot'.
```

---

## Feature 3: Multi-Death Episodic Analysis

### Motivation
Repeated failures in the same environment suggest the agent has discovered a structural obstacle or the environment has a specific challenge that requires meta-strategic thinking. By collecting path data from up to 3 consecutive deaths, the LLM can analyze:
- **Failure clustering:** Do deaths occur at the same coordinates?
- **Loop escalation:** Is the revisit penalty getting worse each attempt?
- **Unexplored bias:** Did the agent consistently avoid certain regions?
- **Structural insight:** What layout feature (bottleneck, dead end, etc.) is causing the problem?

### Implementation

**Location:** `src/agents/planner_agent.py`
- `record_death()`: Records path data when agent dies
- `analyze_multiple_deaths()`: Sends multi-death payload to LLM
- `episode_summary()`: Includes death count

**Triggered in:** `run_experiment.py` — `run_final_exam()` death handler

### Death Recording

**What gets stored per death** (up to 3 deaths):
```python
{
    "death_number": 1,                      # which death this is
    "step_count": 35,                       # how many steps before death
    "unique_tiles": 12,                     # different cells visited
    "revisit_penalty": -4.0,                # cumulative revisit penalty
    "failure_location": (4, 5),             # where agent died
    "revisit_counts": {                     # detailed revisit histogram
        (2, 3): 3,
        (3, 3): 2,
        (4, 4): 1
    },
    "path": [                               # last N trajectory entries
        {"step": 30, "pos": (2,4), "front": "wall", "action": "TurnLeft", "revisit": False},
        ...
    ]
}
```

**Storage Behavior:**
```
Episode 1: Agent dies → record Death #1 → continue play
Episode 2: Agent dies → record Death #2 → continue play
Episode 3: Agent dies → record Death #3 → TRIGGER multi-death analysis
```

Once 3 deaths are collected, `analyze_multiple_deaths()` is called automatically.

### Multi-Death Analysis Payload

**Location:** `src/agents/reflection_engine.py` — `analyze_multi_death_pattern()`

```python
{
    "environment": "MiniGrid-CombinedTesting-v0",
    "analysis_type": "multi_death_episodic",
    "death_count": 3,
    "death_summaries": [
        "Death #1: step=35, unique=12, penalty=-4.0, failure_pos=(4,5), top_revisits=[(2,3);3, (3,3);2]",
        "Death #2: step=42, unique=14, penalty=-7.5, failure_pos=(4,5), top_revisits=[(2,3);4, (3,4);2]",
        "Death #3: step=38, unique=11, penalty=-6.0, failure_pos=(4,6), top_revisits=[(2,3);4, (4,4);3]"
    ],
    "all_death_records": [
        {full death 1 data},
        {full death 2 data},
        {full death 3 data}
    ]
}
```

### Multi-Death LLM Prompt

```
"You are a strategic navigation advisor analyzing REPEATED FAILURES.
Environment: MiniGrid-CombinedTesting-v0
Number of consecutive deaths: 3

Death sequence:
  Death #1: step=35, unique=12, penalty=-4.0, failure_pos=(4,5), top_revisits=...
  Death #2: step=42, unique=14, penalty=-7.5, failure_pos=(4,5), top_revisits=...
  Death #3: step=38, unique=11, penalty=-6.0, failure_pos=(4,6), top_revisits=...

Most common failure locations: pos(4,5)×2, pos(4,6)×1
Penalty trend: worsening

Analyze the pattern across these failures. What structural or strategic 
lesson can the agent learn? For example:
  - Is there a bottleneck or dead end being repeatedly hit?
  - Are revisits getting worse, indicating convergence to local minima?
  - Should the agent avoid certain tile types or patterns?

Output ONLY a valid JSON object with exactly these keys: 
rule, forbidden_action, trigger_feature.
The rule should describe a STRATEGIC PATTERN to avoid, not just a single tile."
```

### Multi-Death LLM Response Example

```json
{
    "rule": "Death cluster around (4,5)-(4,6) indicates a bottleneck; explore alternative eastern paths only",
    "forbidden_action": "Move Forward",
    "trigger_feature": "bottleneck_zone"
}
```

### Strategic Rule Storage
The multi-death rule is **also stored in ChromaDB** and can suppress or override single-hazard rules if queried with "bottleneck" or related semantics.

**Console Output:**
```
╔════════════════════════════════════════════════╗
║  MULTI-DEATH EPISODIC ANALYSIS (Deaths: 3)    ║
╚════════════════════════════════════════════════╝
    Death #1: step=35, unique=12, penalty=-4.0, failure_pos=(4,5), top_revisits=[(2,3);3, (3,3);2]
    Death #2: step=42, unique=14, penalty=-7.5, failure_pos=(4,5), top_revisits=[(2,3);4, (3,4);2]
    Death #3: step=38, unique=11, penalty=-6.0, failure_pos=(4,6), top_revisits=[(2,3);4, (4,4);3]

  [Multi-Death LLM] Analyzing 3 deaths for strategic pattern…
  [Multi-Death Strategy] Death cluster around (4,5)-(4,6) indicates a bottleneck; explore alternative eastern paths only
  [Memory Hub] Stored rule for 'bottleneck_zone'.
```

---

## Integration: How They Work Together

### Timeline of a Multi-Episode Run

```
PHASE 3 — FINAL EXAM (CombinedTesting)
══════════════════════════════════════

Episode 1:
  [Phase 3] Episode 1 starting…
  Step 10:  [🧠 LLM Consult #1] Periodic check → no strong signal yet
  Step 20:  [🧠 LLM Consult #2] Penalty @ -2.5 → suggests "avoid revisited tiles"
  Step 35:  [💀] Died at (4,5) after 35 steps
  [📊 Death Recording] Death #1 recorded. (Stored 1/3)
  
Episode 2:
  [Phase 3] Episode 2 starting…
  Step 12:  [🧠 LLM Consult #1] Periodic check
  Step 24:  [🧠 LLM Consult #2] Penalty @ -4.0 → similar pattern, stronger hint
  Step 42:  [💀] Died at (4,5) after 42 steps (same location!)
  [📊 Death Recording] Death #2 recorded. (Stored 2/3)
  
Episode 3:
  [Phase 3] Episode 3 starting…
  Step 15:  [🧠 LLM Consult #1] Periodic check
  Step 25:  [⚠ Revisit #5] at (2,3) - penalty escalating
  Step 38:  [💀] Died at (4,6) after 38 steps (same region!)
  [📊 Death Recording] Death #3 recorded. (Stored 3/3)
  [📊] Three deaths recorded! Requesting LLM meta-analysis…
  
  ╔════════════════════════════════════════════════╗
  ║  MULTI-DEATH EPISODIC ANALYSIS (Deaths: 3)    ║
  ╚════════════════════════════════════════════════╝
  → LLM analyzes failure cluster at (4,5)-(4,6)
  → Returns: "Bottleneck detected — avoid western approach"
  → Stored in ChromaDB as "bottleneck_zone" rule
  
Episode 4:
  [Phase 3] Episode 4 starting…
  Step 8:   Agent approaches (4,5)
  Step 9:   Agent sees "revisited tile" ahead
  [🧠 LLM KNOWLEDGE] ChromaDB matches "bottleneck_zone" → Turn Right
  Step 15:  Agent explores EASTERN route (new path!)
  Step 20:  Agent discovers goal
  ✓ FLAWLESS SUCCESS — Goal reached in 20 steps!
```

---

## Data Flow Diagram

```
┌─────────────────────────────────┐
│     OnlineExplorerAgent         │
│    (Real-Time Navigation)       │
└──────────┬──────────────────────┘
           │
           ├─ Track: position, tiles visited, revisit_counts
           ├─ Compute: _episode_penalty (-1 per revisit)
           ├─ Buffer: trajectory_log (last 30 steps)
           │
    ┌──────┴──────────────────┐
    │                         │
    v Every 10-15 steps       v On death
┌─────────────────────┐  ┌──────────────┐
│ _should_consult     │  │ record_death │
│ _consult_llm()      │  │ (up to 3)    │
│   (Periodic)        │  └──────┬───────┘
└────────┬────────────┘         │ 3 deaths collected
         │                      v
         │            ┌─────────────────────────┐
         └────►   LLM (llama3.2:3b)             │
         │       get_navigation_hint()    ◄─────┘
         │       analyze_multi_death_pattern()
         │
         v Returns JSON rule
    ┌─────────────────┐
    │ MemoryHub       │
    │ (ChromaDB)      │
    │ store_verified_ │
    │ rule()          │
    └────────┬────────┘
             │
    ┌────────v─────────────┐
    │ Semantic Vector DB   │
    │ (cosine similarity)  │
    │                      │
    │ Stored rules:        │
    │ • "red lava"         │
    │ • "sand"             │
    │ • "revisited_tile"   │
    │ • "bottleneck_zone"  │
    └─────────┬────────────┘
              │
    ┌─────────v──────────────┐
    │ Explorer act() loop     │
    │ query_local_context()  │
    │ "If forward dangerous?" │
    │ → Apply masking        │
    └────────────────────────┘
```

---

## Fallback Behaviors (When Ollama Unavailable)

### Feature 1: Revisit Penalty
- **Always works locally.** No LLM required. Penalty is computed in-memory.

### Feature 2: Periodic LLM Consultation
- **Fallback:** Deterministic heuristic:
  - If `penalty < -2.0` or `top_revisited > 3`: Suggest "avoid revisited tiles"
  - Otherwise: Return `None` (no hint this round)

### Feature 3: Multi-Death Episodic Analysis
- **Fallback:** Deterministic heuristic:
  - If failures cluster in ≤2 locations: Suggest "avoid failure hotspot"
  - If penalty worsening: Suggest "break loops at all costs"
  - Otherwise: Suggest "prioritize reaching unexplored regions"

---

## Configuration Constants

**File:** `src/agents/planner_agent.py`

```python
_TRAJ_MAX = 30              # rolling trajectory buffer size
_REVISIT_GRACE = 10         # steps before revisit penalty applies
_LLM_INTERVAL_MIN = 10      # earliest periodic consultation (steps)
_LLM_INTERVAL_MAX = 15      # latest periodic consultation (steps)
_MAX_DEATHS_TRACKED = 3     # cap on recorded deaths for analysis
```

---

## Experimental Insights & Future Work

### What This Tests
1. **Revisit penalties as learning signals:** Can sparse penalties alone guide the agent to explore efficiently?
2. **Proactive LLM consultation:** Is mid-episode guidance better than post-hoc failure analysis?
3. **Episodic meta-learning:** Can LLMs extract structural insights from repeated failures and apply them strategically?

### Future Enhancements
- **Confidence weighting:** Attach confidence scores to LLM hints (0.0–1.0) to control masking strength
- **Hierarchical rules:** Extend from single-tile recommendations to multi-step behavioral constraints (e.g., "never enter corridor near hazard")
- **Cross-environment transfer:** Use multi-death rules from one environment to warm-start exploration in structurally similar environments
- **Quantitative benchmarking:** Measure success rate, steps-to-goal, and total revisit penalty against pure RL and pure DFS baselines
- **Dynamic grace period:** Adjust `_REVISIT_GRACE` based on environment size and exploration phase

---

## References

- **Hybrid RL-LLM Explorer:** `/Users/ash/CascadeProjects/projects..../game_Exp1/README.md`
- **Core Implementation:** `src/agents/planner_agent.py`, `src/agents/reflection_engine.py`
- **Orchestration:** `run_experiment.py`
- **Vector Store:** `src/core/memory_hub.py`
