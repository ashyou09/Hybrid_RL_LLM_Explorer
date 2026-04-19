# Example Scenarios: Shared Memory in Action

This document walks through three realistic scenarios showing how the revisit penalty, periodic LLM consultation, and multi-death episodic analysis work together.

---

## Scenario 1: Early Loop Detection via Periodic Consultation

**Environment:** CombinedTesting (9×9 with red lava + sand clusters)

### Timeline

```
Step 0:   Explorer spawns at (4,0) facing north
          visited = {(4,0)}
          _step_count = 1

Step 5:   pos=(5,0) → no special events
Step 10:  pos=(6,0) → no special events
          _next_llm_at = 12 (scheduled randomized)

Step 12:  [🧠 LLM Consult #1 TRIGGERED]
          Condition: _step_count (12) >= _next_llm_at (12) AND len(visited) (10+) >= 10
          
          Payload sent to llama3.2:
          {
            "environment": "CombinedTesting",
            "total_steps": 12,
            "unique_tiles_visited": 11,
            "accumulated_revisit_penalty": 0.0,  ← no revisits yet (< grace period)
            "top_revisited_cells": [],
            "trajectory": [last 12 steps...]
          }
          
          LLM Response (fallback):
            Since penalty = 0.0 and no strong revisit signal,
            → Returns: None (no hint this round)
          
          Schedule next: _next_llm_at = 12 + random(10,15) = 24

Step 15:  pos=(4,2) — front = "red lava" (from Phase 1 learning)
          [🧠 LLM KNOWLEDGE] ChromaDB match (dist=0.00) → mask Move Forward
          Agent turns right instead
          
Step 20:  pos=(3,3) — agent is turning in place (same pos)
          revisit_counts[(3,3)] = 2
          penalty = -1.0 (first revisit)
          _episode_penalty = -1.0
          
Step 24:  [🧠 LLM Consult #2 TRIGGERED]
          
          Payload sent:
          {
            "environment": "CombinedTesting",
            "total_steps": 24,
            "unique_tiles_visited": 14,
            "accumulated_revisit_penalty": -3.5,  ← some revisits accumulating
            "top_revisited_cells": [
              ((3,3), 3),   ← this tile is a problem!
              ((4,3), 2)
            ],
            "trajectory": [
              "step 20: pos=(3,3) front='wall' action=TurnLeft revisit=True",
              "step 21: pos=(3,3) front='wall' action=TurnRight revisit=False",
              "step 22: pos=(3,3) front='empty' action=MoveForward revisit=False",
              ...
            ]
          }
          
          LLM Prompt (simplified):
            "Agent is stuck at (3,3) with -3.5 penalty. Recent path shows 
             3 revisits to same tile. Top revisited: (3,3)×3 times.
             How can we escape this loop?"
          
          LLM Response:
          {
            "rule": "Position (3,3) is a revisit hotspot—avoid returning there",
            "forbidden_action": "Move Forward",
            "trigger_feature": "revisited hotspot (3,3)"
          }
          
          [Memory Hub] Stores new rule in ChromaDB
          Next random schedule: _next_llm_at = 24 + random(10,15) = 36

Step 25:  Agent attempts to move toward (3,3)
          query_local_context("revisited hotspot (3,3)") → MATCH (cosine~0.95)
          [🧠 LLM KNOWLEDGE UPDATED] Agent learns to avoid (3,3)
          Turns aggressively to explore new regions
          
RESULT:   By step 24, the agent detected its own looping behavior
          and received tactical advice from the LLM.
          Agent then explores eastern routes successfully.
```

### Key Insight
**Periodic consultation enabled early loop detection.** Without it, the agent might have spent 50+ steps in that corner. The -3.5 penalty signal alerted the LLM to the problem *while the agent is still trapped*, not after failure.

---

## Scenario 2: Escalating Revisit Penalty (Single Episode Growing Worse)

**Environment:** CombinedTesting (9×9)

### Timeline

```
Step 10:  _step_count (10) > _REVISIT_GRACE (10) → revisit penalties NOW active

Step 11:  pos=(3,3) visited before → revisit_counts[(3,3)] += 1 = 1
          penalty = -1.0 × 1 = -1.0
          _episode_penalty = -1.0
          
          Output: [⚠ Revisit #1] pos=(3,3) | penalty=-1.0, cumulative=-1.0

Step 14:  pos=(3,3) again
          revisit_counts[(3,3)] += 1 = 2
          penalty = -1.0 × 2 = -2.0
          _episode_penalty = -3.0
          
          Output: [⚠ Revisit #2] pos=(3,3) | penalty=-2.0, cumulative=-3.0

Step 17:  pos=(3,3) again
          revisit_counts[(3,3)] += 1 = 3
          penalty = -1.0 × 3 = -3.0
          _episode_penalty = -6.0
          
          Output: [⚠ Revisit #3] pos=(3,3) | penalty=-3.0, cumulative=-6.0

Step 20:  pos=(3,3) again
          revisit_counts[(3,3)] += 1 = 4
          penalty = -1.0 × 4 = -4.0
          _episode_penalty = -10.0
          
          Output: [⚠ Revisit #4] pos=(3,3) | penalty=-4.0, cumulative=-10.0

Step 23:  pos=(3,3) again (and again...)
          revisit_counts[(3,3)] += 1 = 5
          penalty = -1.0 × 5 = -5.0
          _episode_penalty = -15.0
          
          Output: [⚠ Revisit #5] pos=(3,3) | penalty=-5.0, cumulative=-15.0

...

Step 65:  pos=(3,3) again (now 50th time!)
          revisit_counts[(3,3)] += 1 = 50
          penalty = -1.0 × 50 = -50.0
          _episode_penalty = -500.0
          
          Check: _episode_penalty (-500.0) <= -500 → TRIGGER FORCE_STOP
          
          Output: [🛑 CRITICAL] Revisit penalty exceeded -500. Force-stopping game.
          force_stop = True
```

### Console Output Pattern

```
[⚠ Revisit #1] pos=(3,3) | penalty=-1.0, cumulative=-1.0
[⚠ Revisit #2] pos=(3,3) | penalty=-2.0, cumulative=-3.0
[⚠ Revisit #3] pos=(3,3) | penalty=-3.0, cumulative=-6.0
[⚠ Revisit #4] pos=(3,3) | penalty=-4.0, cumulative=-10.0
[⚠ Revisit #5] pos=(3,3) | penalty=-5.0, cumulative=-15.0
...
[🛑 CRITICAL] Revisit penalty exceeded -500. Force-stopping game.
```

### Key Insight
**Linear penalty escalation creates a hard signal to the LLM:**
- Small revisits (-1 to -5): Normal exploration noise
- Medium revisits (-50 to -100): Clear looping pattern detected
- Large revisits (-500+): Force-stop; agent is completely stuck

This allows the LLM's multi-death analyzer to later say: *"Episode 1 degraded rapidly (penalty -500), Episode 2 also degraded, Episodes 3 identical—there's a structural bottleneck here."*

---

## Scenario 3: Full Multi-Death Episodic Analysis

**Environment:** CombinedTesting (9×9 with a narrow bottleneck at coordinates (5,4)-(5,5))

### Episode 1 (Death #1)

```
Exploration:
  Steps 1-10:   Normal exploration in western section
  Steps 11-20:  Encounter red lava, use Phase 1 rule successfully
  Steps 21-30:  Explore southern routes, good progress
  Steps 31-35:  Agent navigates toward goal, finds bottleneck at (5,4)
  
At step 35:
  Agent at (5,3), sees bottleneck at (5,4)
  [⚠ Revisit #2] pos=(5,3) | penalty=-2.0, cumulative=-4.0
  Agent tries to force through (no warning from ChromaDB yet)
  [💀] Death at (5,4) by red lava
  
  explorer.record_death((5,4)) called
  
  Death #1 Record:
  {
    "death_number": 1,
    "step_count": 35,
    "unique_tiles": 12,
    "revisit_penalty": -4.0,
    "failure_location": (5,4),
    "revisit_counts": {(5,3): 2, (3,2): 1},
    "path": [last 30 steps...]
  }
  
  Episode Stats:
    Steps taken    : 35
    Unique tiles   : 12
    Revisit penalty: -4.0
    LLM consults   : 2
    Deaths tracked : 1/3
```

### Episode 2 (Death #2)

```
Exploration:
  Steps 1-15:   Agent avoids western dead-end (learned from Ep 1 LLM hints)
  Steps 16-25:  Explores southern section more thoroughly
  Steps 26-42:  Navigates back toward central area
  
  Step 38:  Agent at (4,4), approaching bottleneck again
  [🧠 LLM Consult #2] Gets periodic navigation hint
  But hint not specific enough to block (4,4) region
  
  Step 40:  Agent enters (5,4) region
  [💀] Death at (5,4) again by red lava
  
  explorer.record_death((5,4)) called
  
  Death #2 Record:
  {
    "death_number": 2,
    "step_count": 42,
    "unique_tiles": 14,
    "revisit_penalty": -7.5,
    "failure_location": (5,4),  ← SAME LOCATION AS EP 1!
    "revisit_counts": {(5,3): 3, (3,2): 1, (4,4): 2},
    "path": [last 30 steps...]
  }
  
  Episode Stats:
    Steps taken    : 42
    Unique tiles   : 14
    Revisit penalty: -7.5
    LLM consults   : 3
    Deaths tracked : 2/3
```

### Episode 3 (Death #3 → Multi-Death Analysis Triggered)

```
Exploration:
  Steps 1-20:   Agent still avoiding some Western routes (residual learning)
  Steps 21-38:  Systematic exploration, but penalty accumulating faster
  
  Step 25:  [⚠ Revisit #4] pos=(3,2) | penalty=-4.0, cumulative=-8.0
  Step 26:  [⚠ Revisit #5] pos=(3,2) | penalty=-5.0, cumulative=-13.0
  
  Step 38:  Agent approaches bottleneck from different angle
  [💀] Death at (5,6) (nearby region)
  
  explorer.record_death((5,6)) called
  
  Death #3 Record:
  {
    "death_number": 3,
    "step_count": 38,
    "unique_tiles": 11,
    "revisit_penalty": -13.0,
    "failure_location": (5,6),  ← SAME GENERAL REGION!
    "revisit_counts": {(3,2): 5, (4,4): 3, (5,3): 2},
    "path": [last 30 steps...]
  }
  
  Episode Stats:
    Steps taken    : 38
    Unique tiles   : 11
    Revisit penalty: -13.0
    LLM consults   : 2
    Deaths tracked : 3/3


  ╔════════════════════════════════════════════════╗
  ║  MULTI-DEATH EPISODIC ANALYSIS (Deaths: 3)    ║
  ╚════════════════════════════════════════════════╝
  
  Death Summary Analysis:
    Death #1: step=35, unique=12, penalty=-4.0, failure_pos=(5,4), top_revisits=[(5,3);2, (3,2);1]
    Death #2: step=42, unique=14, penalty=-7.5, failure_pos=(5,4), top_revisits=[(5,3);3, (3,2);1, (4,4);2]
    Death #3: step=38, unique=11, penalty=-13.0, failure_pos=(5,6), top_revisits=[(3,2);5, (4,4);3, (5,3);2]
    
  Computed Metrics:
    Most common failure locations: pos(5,4)×2, pos(5,6)×1
    Penalty trend: worsening (-4.0 → -7.5 → -13.0)
    Revisit intensity: (3,2) visited 5 times in Ep3 (vs 1 time in Ep1)
    
  LLM Prompt:
    "You are a strategic advisor analyzing REPEATED FAILURES.
     Environment: CombinedTesting
     Number of consecutive deaths: 3
     
     Death sequence:
       Death #1: 35 steps, failed at (5,4)
       Death #2: 42 steps, failed at (5,4)
       Death #3: 38 steps, failed at (5,6)
     
     Most common failures: (5,4) and (5,6) — adjacent cells in same region
     Penalty trend: worsening (-4.0 → -7.5 → -13.0)
     
     What structural lesson can be learned?"
  
  LLM Response:
  {
    "rule": "Narrow bottleneck zone detected at (5,4)-(5,6); all eastern approaches fail repeatedly. Explore alternative WESTERN or SOUTHERN routes exclusively.",
    "forbidden_action": "Move Forward",
    "trigger_feature": "bottleneck_eastern_approach"
  }
  
  [Memory Hub] Stores rule for 'bottleneck_eastern_approach'
  
  Output:
    [🧠 Strategic Hint] Narrow bottleneck zone detected at (5,4)-(5,6); all eastern approaches fail repeatedly. Explore alternative WESTERN or SOUTHERN routes exclusively.
    [Memory Hub] Stored rule for 'bottleneck_eastern_approach'.
```

### Episode 4 (Success!)

```
Exploration:
  Steps 1-10:   Normal exploration
  
  Step 12:  [🧠 LLM Consult #1]
            Payload includes new rule: 'bottleneck_eastern_approach'
  
  Step 15:  Agent approaches region (5,3)
            query_local_context("bottleneck_eastern_approach", threshold=0.60)
            ChromaDB semantic match (cosine ~0.88) → RULE FIRES
            [🧠 LLM KNOWLEDGE] bottleneck detected — mask Move Forward
            Agent turns and explores WESTERN routes
  
  Steps 16-25:  Systematic exploration of western section (never tried before)
  Step 26:    Agent discovers clear path to goal!
  Step 28:    [✓ GOAL REACHED] 28 steps
  
  ✓ FLAWLESS SUCCESS — Episode goal reached in 28 steps!
  
  Episode Stats:
    Steps taken    : 28
    Unique tiles   : 22  ← more thorough exploration of safe zones
    Revisit penalty: -0.5  ← minimal revisits (avoided bottleneck)
    LLM consults   : 1
    Deaths tracked : 0 (success!)
```

### Key Insights

1. **Failure clustering detected:** Deaths #1 & #2 at same location → structural bottleneck hypothesis
2. **Penalty trend as meta-signal:** Escalating penalty (-4 → -7.5 → -13) shows worsening performance, not learning
3. **Semantic generalization:** Rule for "(5,4)-(5,6)" generalizes to block any "eastern_approach" query
4. **Success follows strategy:** Once LLM identified the structural issue, agent simply avoided it and succeeded
5. **Human-like reasoning:** LLM output resembles human strategy: *"Don't keep ramming into that wall; try a different direction!"*

---

## Scenario 4: Comparing TrajectorySizes at Different LLM Consult Intervals

### Comparison: 10-15 Steps vs. 25-35 Steps

**With 10-15 step intervals (NEW):**
```
Episode 8, Step 15: [🧠 LLM Consult #3]
  → Gets hint early; has only 15 steps of context
  → LLM: "You're oscillating between (3,3) and (4,3). Try (2,3) instead."
  → Agent implements; breaks loop immediately
  → Total episode: 40 steps, 1 LLM intervention

Episode 9, Step 15: [🧠 LLM Consult #3]
  → Gets hint again; corrects path
  → Total episode: 42 steps
```

**With 25-35 step intervals (from original code):**
```
Episode 8, Step 25: [🧠 LLM Consult #2]
  → Gets hint late; has 25 steps of repeated spinning
  → LLM: "You've been looping badly. You're in a local minimum."
  → Agent responds, but 10+ steps already wasted
  → Total episode: 65 steps, 1 LLM intervention (late)

Episode 9, Step 35: [🧠 LLM Consult #2]
  → Doesn't consult until step 35
  → May already be dead or too committed to a bad path
  → Total episode: 75 steps
```

**Research Takeaway:**
Shorter intervals (10-15) allow **proactive course correction**, while longer intervals (25-35) result in **reactive damage control**. For loop-prone environments, more frequent consultation is better.

---

## Summary: Shared Memory in Action

These scenarios demonstrate:

1. **Revisit Penalty:** Grows linearly with each revisit, creating a learnable signal
2. **Periodic Consultation:** Every 10-15 steps, the agent gets tactical hints without waiting for failure
3. **Multi-Death Analysis:** After 3 consecutive failures, LLM synthesizes a strategic meta-rule
4. **Semantic Storage:** All rules coexist in ChromaDB and compose naturally

The result: **A learning system where an RL agent's failures become exploration knowledge that helps an independent rule-guided agent navigate successfully.**
