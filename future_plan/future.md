# future.md — The 6-Level Plan

We build the research in **six levels**, from very small to large. Each level is a
complete, working mini-project with its own question, code, results and write-up.
Only when a level is finished (and written up in `conclusion/`) do we **hand over**
to the next one.

Level 1 is written in full detail below. Levels 2–6 are outlines; each gets
expanded to Level-1 detail at its handover.

Progress is tracked in [tracker.md](tracker.md).

---

## The ladder at a glance

| Level | Name | Question in one line | Size |
|---|---|---|---|
| **1** | **First Transfer** | Can rules learned from deaths on *some* hot tiles protect a new agent on *other, never-seen* hot tiles, without fearing harmless look-alikes? | 1 learner, 1 novice, 1 danger category |
| 2 | Experience vs. Common Sense | When the world breaks common sense (water kills, fire is safe), does experience override the LLM's prior? | + 4 categories, counter-intuitive worlds |
| 3 | The Truth Filter | Can testing rules against reward catch wrong (hallucinated / noisy) rules? | + verification loop, Verified / Rumor / Myth, 10% / 30% noise |
| 4 | When to Think | How often must the LLM be called, and can a small model replace a big one? | + triggers E1–E5, model-size comparison |
| 5 | Two Agents Share | Does verified rule-sharing between agents beat learning alone, and does trust stop a corrupted agent? | 2–3 agents, trust score |
| 6 | The Village | Does Semantic Osmosis scale to many agents with limited bandwidth? | N agents, Villa, ShareScore |

Mapping to the research doc: L1–L2 = zero-shot transfer and the hallucination test;
L3 = Truth Filter, Myth state, E6/E7; L4 = RQ1, RQ6, E1–E5; L5 = trust score and
Gap 1 Layer 3; L6 = Villa, ShareScore, the Coordination Gap.

---

## Level 1 — First Transfer (full detail)

### 1.1 Research question

> After a learner agent dies on a few **hot** tiles (e.g. *red lava*, *campfire*),
> can an LLM turn that experience into rules that let a **novice agent** avoid
> **never-seen hot tiles** (e.g. *molten rock*) — while still walking over
> **harmless look-alikes** (e.g. *red carpet*)?

### 1.2 Why this is the right first step

- It is the smallest experiment where the LLM **must** do something a lookup table
  cannot: the test tiles were never seen.
- It fixes the reviewer problems in today's code (see `01_current_state_review.md`):
  real RL (C1), an LLM that adds value (C2), local perception (C3), statistics (C4),
  and a test of the embedding threshold (C5).
- It is cheap: about 20 distinct tiles, LLM answers cached → a few dozen API calls.

### 1.3 Hypotheses for Level 1

| ID | Hypothesis |
|---|---|
| L1-H1 | Exact lookup protects only on seen tiles; on unseen hot tiles it is no better than no shield. |
| L1-H2 | The LLM shield (with experience) avoids most unseen hot tiles. |
| L1-H3 | Embedding retrieval (current ChromaDB method) confuses decoys with hazards more often than the LLM shield does. |
| L1-H4 | In this *intuitive* world, the LLM with no experience does about as well as the LLM with experience. (This sets up Level 2.) |

### 1.4 The world

- **Grid:** 7×7 MiniGrid room, walls around, start top-left, goal bottom-right.
- **Tiles:** every special tile has a natural-language description. Only one
  category is deadly in Level 1: **heat**.

| Group | Example tiles (proposal — to be discussed) | Deadly? |
|---|---|---|
| Heat — train | red lava, campfire, glowing hot coals | yes |
| Heat — test (never seen in training) | molten rock, burning log pile, white-hot metal plate | yes |
| Plain — train | grass, wooden floor | no |
| Decoys — test | red carpet, red flower bed, painting of a fire, orange rug, cold stove | no |

- **Layouts:** random tile positions each episode, always with a safe path to the goal.
- **Perception:** the agent sees only the text of the tiles in front, left and right.
  No coordinates, no map, no hidden "deadly" label.
- **Rewards:** −1 per step, −10 and episode end on a deadly tile, +10 at the goal.

### 1.5 The agents

**Agent A — the learner (real RL).** Tabular Q-learning.
- State: (front-tile token, goal direction relative to facing).
- Actions: turn left, turn right, move forward.
- Trains for a fixed number of episodes in *training* worlds (train tiles only).
- Writes an **experience log**: `died_on: "red lava"`, `walked_safely_on: "grass"`,
  `reached_goal`. Never coordinates.

**The LLM — sparse reflection.** Called once after training. Reads the experience
log and writes **structured rules**:
`{id, if: "tile is hot / burning", then: "do not move forward", confidence, evidence_ids}`.

**Agent B — the novice.** Starts with Agent A's Q-table for navigation (so we test
the rules, not navigation skill) and a **shield** that blocks "move forward" when the
front tile is judged dangerous. Plays *test* worlds with unseen hot tiles and decoys.

### 1.6 Shields compared

| ID | Shield | How it decides "dangerous?" |
|---|---|---|
| S0 | None | never blocks |
| S1 | Exact lookup | tile name is in the list of tiles that killed A |
| S2 | Embedding retrieval | cosine similarity to a killer tile ≥ 0.65 (current code) |
| S3 | LLM prior | asks the LLM with **no** experience |
| S4 | **LLM + experience (ours)** | asks the LLM with A's rules and log |
| S6 | Oracle | knows the true category |

(S5, "with verification", arrives in Level 3.)

### 1.7 Metrics

| Metric | Meaning |
|---|---|
| Success rate | episodes reaching the goal |
| Unseen-hazard death rate | deaths on test hot tiles per episode |
| Decoy false-avoidance | decoys blocked / decoys met |
| Shield precision and recall | over all distinct test tiles |
| Steps to goal | efficiency |
| LLM calls, tokens | cost |

Protocol: 10 seeds × 100 test episodes per shield, separate training and test seeds,
95% bootstrap confidence intervals.

### 1.8 Steps (each one: discuss → build → push → review)

| Step | What | Done when |
|---|---|---|
| L1.1 | Create `research/` package, `seed_everything`, `pytest`, research requirements | `pytest` passes; same seed gives identical runs |
| L1.2 | Tile catalog for Level 1 (table in 1.4) | catalog loads; train/test/decoy split tested |
| L1.3 | Text-layer MiniGrid env + layout generator + local perception | deadly tile ends episode; 1 000 layouts all solvable; no coordinates in obs |
| L1.4 | Agent A: tabular Q-learning + action-mask hook + training curve | > 90% success in a no-hazard world; curve saved |
| L1.5 | Experience log (structured, no coordinates) | schema test passes |
| L1.6 | LLM adapter (Groq / OpenRouter / Claude / fake) + disk cache | fake provider tests pass; second run makes 0 API calls |
| L1.7 | Reflection prompt: log → structured rules | rules parse and validate |
| L1.8 | Shields S0–S4, S6 behind one interface | each passes a small labelled test |
| L1.9 | **First result:** tile-classification benchmark (no RL) | precision/recall table for every shield |
| L1.10 | Novice evaluation with seeds and confidence intervals | results table + plots in `results/level_1/` |
| L1.11 | Write `conclusion/level_1/paper.md` and `explained.md` | both written, every number linked to a command |
| L1.12 | Handover: expand Level 2 in this file to full detail | tracker updated, Level 2 detailed |

### 1.9 Level 1 deliverables

- `research/` code with tests
- `results/level_1/` (configs, metrics JSON, plots, LLM cache)
- `conclusion/level_1/paper.md` — paper format
- `conclusion/level_1/explained.md` — plain-language explanation
- `tracker.md` updated with the handover note

### 1.10 Handover checklist (Level 1 → Level 2)

- [ ] All L1 steps ticked in the tracker
- [ ] Hypotheses L1-H1…H4 each marked supported / not supported, with numbers
- [ ] Surprises and problems listed in `explained.md`
- [ ] Level 2 section below rewritten in full detail using what we learned

---

## Level 2 — Experience vs. Common Sense (outline)

- **Question:** when a world's dangers contradict common sense, does the LLM with
  experience adapt where the LLM prior fails?
- **Adds:** four categories (heat, liquid, sharp, chemical); intuitive vs.
  counter-intuitive worlds; split results by world type.
- **Key comparison:** S3 (prior) vs. S4 (experience). This is the doc's
  "hallucination test".
- **Reuses:** all Level 1 code; only catalog, world spec and evaluation splits grow.

## Level 3 — The Truth Filter (outline)

- **Question:** does testing rules against reward stop wrong rules from being used?
- **Adds:** verification test (probe episodes with vs. without a rule);
  knowledge states Verified → Rumor → Myth; Myth archive; noise injection 10% / 30%
  (doc E6, E7); shield S5.
- **Metrics:** wrong-rule usage, recovery rate, tests saved by the Myth archive.

## Level 4 — When to Think (outline)

- **Question:** how often should the LLM be called, and can a small model replace a large one?
- **Adds:** verification triggers E1–E5 (none / always / on failure / every 5 / every 10
  steps); success rate vs. LLM calls trade-off curve; small vs. large vs. Claude models;
  latency measurement.
- **Answers:** doc RQ1 and RQ6.

## Level 5 — Two Agents Share (outline)

- **Question:** does sharing verified rules beat learning alone, and does a trust score
  stop a corrupted agent?
- **Adds:** 2–3 learners in different worlds; shared memory hub; trust score T
  (role accuracy, past accuracy, experience depth, recency); one corrupted agent.
- **Metrics:** redundant trap hits, time to safe behaviour, spread of wrong rules.

## Level 6 — The Village (outline)

- **Question:** does Semantic Osmosis scale to many agents with limited bandwidth?
- **Adds:** N agents (up to the 20-agent Villa), roles, ShareScore, 3-tier memory.
- **Metrics:** redundant trap hits vs. N, bytes shared, convergence time.
- **Only if** Levels 1–5 show clear effects.
