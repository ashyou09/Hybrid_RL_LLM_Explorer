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

### 1.8 The Lab — see it happen, turn the knobs

The experiment should feel like a real lab: you can **watch** what the agents do and
**change one thing at a time** to see what it causes.

**Knobs (independent variables)** — all live in one config file, so every run is
fully described by its config:

| Knob | Example values | What it changes |
|---|---|---|
| `grid_size` | 5, 7, 9 | room size |
| `hazard_count` | 1 … 8 | how many deadly tiles per episode (difficulty) |
| `decoy_count` | 0 … 6 | how many harmless look-alikes |
| `experience_budget` | 1, 3, 5, 10 deaths | how much the learner experienced before the LLM writes rules |
| `train_episodes` | 200 … 5 000 | how long Agent A trains |
| `shield` | S0 … S6 | which decision method the novice uses |
| `llm_model` | small / large open model, Claude | which brain |
| `embed_threshold` | 0.5 … 0.9 | how strict embedding matching is |
| `seeds`, `episodes` | 10, 100 | statistical power |

**Measurements (dependent variables):** success rate, unseen-hazard deaths, decoy
false-avoidance, steps to goal, LLM calls.

**What you can see:**

1. **Episode replay** — the grid drawn frame by frame (saved as GIF), with tile names,
   and every shield decision shown: *"molten rock → BLOCKED (rule R2: hot things kill)"*.
2. **Lab panel** (Streamlit, runs locally or on the Hugging Face Space) — sliders for
   the knobs, a "Run" button, the replay, and live result charts.
3. **Sweeps** — turn one knob across a range, keep everything else fixed, plot the effect.
   Planned Level 1 sweeps:
   - *Difficulty sweep:* death rate vs. `hazard_count`, one line per shield.
   - *Strictness sweep:* `embed_threshold` vs. hazards caught and decoys wrongly
     blocked — shows there is no threshold where embeddings beat the LLM (or that
     there is, which would be a finding).
   - *Experience sweep:* how many deaths does the learner need before the rules work?

### 1.9 Steps (each one: discuss → build → push → review)

| Step | What | Done when |
|---|---|---|
| L1.1 | Create `research/` package, `seed_everything`, `pytest`, research requirements, config file with all knobs | `pytest` passes; same seed + config gives identical runs |
| L1.2 | Tile catalog for Level 1 (table in 1.4) | catalog loads; train/test/decoy split tested |
| L1.3 | Text-layer MiniGrid env + layout generator + local perception | deadly tile ends episode; 1 000 layouts all solvable; no coordinates in obs |
| L1.4 | Episode replay: draw the grid with tile names, save GIF | a random-agent episode is saved as a GIF you can watch |
| L1.5 | Agent A: tabular Q-learning + action-mask hook + training curve | > 90% success in a no-hazard world; curve saved |
| L1.6 | Experience log (structured, no coordinates) | schema test passes |
| L1.7 | LLM adapter (Groq / OpenRouter / Ollama / Claude / fake) + disk cache | fake provider tests pass; second run makes 0 API calls |
| L1.8 | Reflection prompt: log → structured rules | rules parse and validate |
| L1.9 | Shields S0–S4, S6 behind one interface; decisions shown in the replay | each passes a small labelled test |
| L1.10 | **First result:** tile-classification benchmark (no RL) | precision/recall table for every shield |
| L1.11 | Novice evaluation with seeds and confidence intervals | results table + plots in `results/level_1/` |
| L1.12 | Sweeps (difficulty, strictness, experience) | three plots in `results/level_1/` |
| L1.13 | Lab panel (Streamlit): knobs, run, replay, charts | panel runs locally and on the HF Space |
| L1.14 | Write `conclusion/level_1/paper.md` and `explained.md` | both written, every number linked to a command |
| L1.15 | Handover: expand Level 2 in this file to full detail | tracker updated, Level 2 detailed |

### 1.10 Tools and where to run

- **LLM:** open-weight models first, so anyone can reproduce the results.
  - **Main:** Llama-family open models through **Groq's free tier** (fast; free key).
    The same models can run offline with **Ollama** for people who prefer local.
  - **Backup:** OpenRouter free models.
  - **Reference "big brain":** Claude, used sparingly (cached answers cost cents) — mainly in Level 4's model comparison.
- **Machine:** nothing heavy runs locally. Tabular Q-learning needs no GPU, and the
  LLM runs on the API. Develop and run in a cloud session or Google Colab; host the
  Lab panel on the existing Hugging Face Space. The laptop only needs a browser.
  The embedding shield (S2) needs CPU PyTorch, which is the largest install — it is
  kept as an optional extra.

### 1.11 Level 1 deliverables

- `research/` code with tests
- `results/level_1/` (configs, metrics JSON, plots, replay GIFs, LLM cache)
- the Lab panel (Streamlit), runnable locally and on the HF Space
- `conclusion/level_1/paper.md` — paper format
- `conclusion/level_1/explained.md` — plain-language explanation
- `tracker.md` updated with the handover note

### 1.12 Handover checklist (Level 1 → Level 2)

- [ ] All L1 steps ticked in the tracker *(L1.10–L1.12 still need the LLM / embedding runs)*
- [ ] Hypotheses L1-H1…H4 each marked supported / not supported, with numbers *(H1 done; H2–H4 pending)*
- [x] Surprises and problems listed in `explained.md`
- [x] Level 2 section below rewritten in full detail using what we learned

---

## Level 2 — Experience vs. Common Sense (full detail)

*Expanded at the Level 1 handover. Level 1's LLM runs (S2–S4) must be finished first,
because Level 2 builds on their numbers.*

### 2.1 Research question

> When a world's dangers **contradict common sense** — water kills, fire is harmless —
> does an LLM that reads the agent's experience (S4) adapt, while an LLM relying on
> common sense alone (S3) fails?

This is the research doc's "hallucination test": the LLM's plausible belief ("fire is
dangerous") is wrong in this world, and only grounded experience can correct it.

### 2.2 What Level 1 taught us (inputs to this design)

- Exact lookup cannot transfer at all; the oracle bounds the gain at ~35 points of success.
- A blocked learner times out ~7% of the time; report time-outs separately.
- The experience budget matters: with 5 deaths the learner never died on a campfire.
  Level 2 sweeps the budget per category.
- Fake keyword matching blocks "painting of a fire": decoys work as intended.

### 2.3 Hypotheses

| ID | Hypothesis |
|---|---|
| L2-H1 | In intuitive worlds, S3 ≈ S4 (common sense is enough). |
| L2-H2 | In counter-intuitive worlds, S3 dies on the "surprising" category and over-blocks the harmless "scary" one; S4 does neither. |
| L2-H3 | S4's advantage grows with the experience budget. |
| L2-H4 | Embedding retrieval (S2) fails in both world types on held-out tiles. |

### 2.4 The worlds

Four categories, each with ≥ 3 train tiles, ≥ 3 held-out test tiles and look-alike decoys:

| Category | Train (examples) | Test (examples) | Decoys (examples) |
|---|---|---|---|
| heat | red lava, campfire, hot coals | molten rock, burning logs, white-hot plate | red carpet, painting of a fire |
| liquid | deep water, flooded pit, swamp | river crossing, ice-cold lake, tar pool | blue rug, puddle sticker |
| sharp | spike trap, broken glass, thorn bush | blade floor, nail bed, razor wire | grey gravel path, cactus painting |
| chemical | acid pool, toxic sludge, bleach spill | chemical drum leak, corrosive puddle | green paint, lime-green mat |

| World family | Deadly categories | Harmless categories |
|---|---|---|
| Intuitive | heat, sharp, chemical | liquid (shallow, safe here) |
| Counter-intuitive | liquid | heat, sharp, chemical |

Worlds are sampled per seed; results are split by family.

### 2.5 Steps

| Step | What | Done when |
|---|---|---|
| L2.1 | Extend the tile catalog to four categories (+ decoys) | catalog tests pass; every category has train/test/decoy tiles |
| L2.2 | World families in config (`world_family: intuitive / counter`) | the env kills by family; tests cover both |
| L2.3 | Per-family evaluation and tables (split results) | `evaluate` reports both families |
| L2.4 | Classification benchmark for both families, all shields | precision/recall table per family |
| L2.5 | Novice evaluation, both families | main table with CIs |
| L2.6 | Experience-budget sweep per family (S3 vs S4) | plot |
| L2.7 | Lab: world-family switch | visible in the panel |
| L2.8 | Write `conclusion/level_2/` (paper + explained) | both written |
| L2.9 | Handover: expand Level 3 | tracker updated |

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
