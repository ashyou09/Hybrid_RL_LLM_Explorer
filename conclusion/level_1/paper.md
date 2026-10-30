# Level 1 — First Transfer

*Paper-format write-up. Every number cites the command that produced it.*
*Status: draft. Sections marked **[PENDING]** need the LLM (S3, S4) and embedding (S2) runs.*

## Abstract

Agents that learn safety only from their own failures generalise poorly to hazards
they have not met. We study whether a language model, reading a reinforcement-learning
agent's sparse failure log, can write rules that protect a *novice* agent against
**held-out** hazards while sparing **look-alike decoys**. In a MiniGrid room with a
natural-language tile layer, a tabular Q-learning agent trained on three hot tiles
reaches the goal in 90.7% of training episodes but dies in 41.1% [39.0, 43.6] of test
episodes containing only unseen hot tiles. An exact-match lookup of past killers gives
no protection (identical to no shield), while an oracle shield removes all deaths and
raises success from 57.4% to 92.6%. **[PENDING]** Results for embedding retrieval and
for LLM shields with and without experience.

## 1. Introduction

Reinforcement-learning agents learn from reward but share nothing interpretable;
language-model agents share readable advice but can be wrong. A middle path is to let
an RL agent experience failure, let an LLM *reflect* on that experience once (sparse
reflection), and let a new agent use the resulting rules as a **shield** that forbids
unsafe actions. The question is whether such rules carry information beyond the exact
cases experienced.

Contributions of this level:

1. A text-layer gridworld with hidden danger categories, held-out hazards and decoys,
   in which a lookup of past failures provably cannot succeed.
2. A paired evaluation protocol comparing six shields from "none" to "oracle".
3. Baselines bounding the achievable improvement (57.4% → 92.6% success).
4. **[PENDING]** The measured position of LLM- and embedding-based shields within that bound.

## 2. Related Work

*Only references marked ✅ in `docs/references.md` may be cited in a submission;
the items below are still being verified.*

- Shielding for safe RL (Alshiekh et al., AAAI 2018) ⚠ — our shield is learned from experience rather than specified.
- LLM guidance of RL agents (ELLM, Du et al., ICML 2023) ⚠.
- Verbal reflection on failure (Reflexion, Shinn et al., 2023) ⚠.
- LLMs as zero-shot planners and the grounding gap (Huang et al., ICML 2022) ⚠.

## 3. Method

### 3.1 Environment

A 7×7 MiniGrid room (5×5 interior), start top-left, goal bottom-right. Each episode
places `hazard_count = 3` deadly tiles and `decoy_count = 2` harmless tiles at random
positions, rejecting layouts without a safe path (BFS). Tiles come from a catalog with a
name, a description and a hidden category; a world declares which categories kill
(Level 1: `heat`). Rewards: −1 per step, −10 and termination on a deadly tile, +10 at the
goal; time-out after 100 steps.

| Split | Heat (deadly) | Harmless |
|---|---|---|
| Train | red lava, campfire, glowing hot coals | grass, wooden floor |
| Test | molten rock, burning log pile, white-hot metal plate | red carpet, red flower bed, painting of a fire, orange rug, cold stove |

Observation: the names of the front, left and right cells and a goal compass
(ahead/left/right/behind). No coordinates, map or category labels.

### 3.2 Learner (Agent A)

Tabular Q-learning (α = 0.2, γ = 0.95, ε linearly 1.0 → 0.05 over 70% of 6 000
episodes). State = (front cell name, kind of left/right cell ∈ {wall, open, tile}, goal
direction, previous action). The previous action removes left–right oscillation
(without it ~40% of episodes time out).

### 3.3 Rule synthesis (sparse reflection)

Training history is cut after `experience_budget = 5` deaths and aggregated by tile into
an experience log (`died_on`, `walked_safely_on`, `reached_goal`, with counts and ids).
One LLM call turns the log into rules `{id, if, then, confidence, evidence}` describing
tile properties; invalid rules and unknown evidence ids are dropped.

### 3.4 Shields

The novice uses Agent A's Q-table; a shield masks "move forward" when the front tile is
judged deadly. S0 none; S1 exact lookup of killer tiles; S2 cosine similarity of tile
names to killer names (all-MiniLM-L6-v2, threshold 0.65, as in the v0 prototype); S3 LLM
judge without experience; S4 LLM judge with the rules and log; S6 oracle on hidden
categories. LLM calls use temperature 0 and are cached by prompt hash.

## 4. Experimental Setup

Test worlds use the test split only. Each shield is evaluated greedily on 10 seeds ×
100 episodes; seeds are disjoint from training. The agent's tie-breaking RNG is reset per
evaluation, so all shields face identical layouts and draws (paired comparison). We
report means with 95% bootstrap intervals over seeds. Metrics: success, death,
hazard blocking (blocks / encounters of test hazards), decoy false-avoidance (blocks /
encounters of decoys).

## 5. Results

**Learner.** 90.7% [88.6, 92.6] success and 0.4% [0.0, 0.9] deaths in training worlds
(`python -m research train`).

**Novice in test worlds** (`python -m research evaluate`):

| Shield | Success | Death | Hazards blocked | Decoys avoided |
|---|---|---|---|---|
| S0 none | 57.4% [55.1, 59.4] | 41.1% [39.0, 43.6] | 0% | 0% |
| S1 lookup | 57.4% [55.1, 59.4] | 41.1% [39.0, 43.6] | 0% | 0% |
| S2 embedding | **[PENDING]** | | | |
| S3 LLM prior | **[PENDING]** | | | |
| S4 LLM + experience | **[PENDING]** | | | |
| S6 oracle | 92.6% [90.9, 94.4] | 0.0% [0.0, 0.0] | 100% | 0% |

**Difficulty sweep** (`python -m research sweep --knob hazard_count --values 1,2,4,6`):
deaths without a shield rise 15.9% → 28.8% → 54.5% → 74.6% for 1, 2, 4, 6 hazards; the
oracle remains at 0% with success 98.7% → 74.3%.

| Hypothesis | Status |
|---|---|
| L1-H1 Exact lookup gives no protection on unseen hazards | **Supported** (identical to S0) |
| L1-H2 The LLM shield with experience avoids most unseen hazards | **[PENDING]** |
| L1-H3 Embedding retrieval confuses decoys with hazards more than the LLM shield | **[PENDING]** |
| L1-H4 In this intuitive world, LLM prior ≈ LLM + experience | **[PENDING]** |

## 6. Discussion

The lookup result is by construction but important: it shows that the evaluation
cannot be passed by memorising failures, which the v0 prototype's design could not rule
out. The oracle gap (35 points of success, 41 points of deaths) is the room available to
any semantic shield. Even a perfect shield leaves 7.4% time-outs, because the learner's
policy was never trained with masked actions; a shield-aware learner is future work.

## 7. Limitations

- One danger category, intuitive by design; the LLM's pretraining knowledge ("lava is
  hot") may suffice without experience — this is what Level 2 tests.
- Tabular learner with a goal compass; local perception only.
- Decoys and categories are hand-designed.
- **[PENDING]** Model dependence: results for one or more LLMs.

## 8. Conclusion

**[PENDING]** — to be written once S2–S4 are measured.

## Reproducibility

```bash
pip install -r requirements-research.txt
```

```bash
python -m research train
```

```bash
python -m research evaluate --shields none,lookup,oracle
```

```bash
python -m research sweep --knob hazard_count --values 1,2,4,6 --shields none,lookup,oracle
```

LLM shields (needs a key in the environment or `.env`, and network access to the provider):

```bash
python -m research classify --provider groq
```

```bash
python -m research evaluate --provider groq
```

Embedding shield (needs `pip install sentence-transformers` and access to huggingface.co):

```bash
python -m research evaluate --shields embedding
```

## References

See `docs/references.md`; all entries are currently ⚠ (to verify).
