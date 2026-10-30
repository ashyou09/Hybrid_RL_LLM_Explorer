# 03 — Exp-1 Design: Verified Semantic Rule Transfer to Unseen Hazards

## Agents

- **Agent A (learner):** tabular Q-learning in *training* worlds. Sees only train tiles.
  Produces an experience log (deaths, safe crossings, goals) — no coordinates.
- **LLM (analytic brain):** wakes up on failure (sparse reflection) and writes
  category-level rules from A's log.
- **Agent B (novice):** fresh agent placed in *test* worlds containing **held-out**
  tiles and decoys. Uses A's Q-table for navigation and a **shield** that masks
  "move forward" into tiles judged dangerous.

## Worlds

Each world secretly marks some categories as deadly.

| Family | Deadly categories (example) | What it tests |
|---|---|---|
| Intuitive | heat, sharp, chemical | Can LLM priors + evidence transfer to new tiles? |
| Counter-intuitive | liquid (water kills), heat is harmless | Does evidence override priors? (hallucination test) |

Train/test split is **by tile, within category**: e.g. train "red lava", "campfire";
test "molten rock", "burning coals". Decoys share surface features but no
danger category: "red carpet", "red flowers", "painting of a fire", "blue rug".

## Conditions

| ID | Shield | Uses experience | Uses world knowledge |
|---|---|---|---|
| S0 | None | ✗ | ✗ |
| S1 | Exact lookup (killer tiles) | ✓ | ✗ |
| S2 | Embedding retrieval, threshold 0.65 (current code) | ✓ | partial |
| S3 | LLM prior (no experience given) | ✗ | ✓ |
| S4 | **LLM + experience (ours)** | ✓ | ✓ |
| S5 | S4 + verification (Truth Filter) | ✓ | ✓ |
| S6 | Oracle (true categories) | — | — |

Crossed with the doc's verification triggers (E1–E5) and noise levels (E6/E7: 0%, 10%, 30%).
Not every cross is needed: run triggers and noise on S4/S5 only.

## Metrics

| Metric | Definition |
|---|---|
| Success rate (SR) | episodes reaching goal / episodes |
| Unseen-hazard death rate | deaths on held-out tiles / episodes |
| Decoy false-avoidance | decoys judged dangerous / decoys encountered |
| Shield precision / recall | over all distinct tiles in the test worlds |
| Steps to goal | mean over successful episodes |
| LLM calls, tokens, wall time | per experiment |
| Wrong-rule usage / recovery | for noise conditions |
| Packet size | bytes of rule set vs. bytes of raw experience log |

## Protocol

- ≥ 10 seeds × 100 test episodes per condition; separate seeds for training and test.
- 95% bootstrap confidence intervals; report all conditions, including failures.
- LLM answers cached on disk; the cache file is part of the results for reproducibility.
- Cheapest first: run S0, S1, S2, S6 (no API) → one free model for S3/S4 → full model comparison.

## Expected cost

About 30–40 distinct tiles × 7 conditions × 3 models, cached → a few hundred LLM
calls per full sweep. Fits within free tiers (Groq / OpenRouter); Claude runs cost cents.
