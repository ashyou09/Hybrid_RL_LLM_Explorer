# Novice in test worlds: shields that need no LLM (S0, S1, S6)

Same trained Agent A as the baseline; test worlds contain only never-seen hot tiles
(molten rock, burning log pile, white-hot metal plate) and decoys. 10 seeds × 100
episodes per shield, paired (same layouts and random draws for every shield);
95% bootstrap CI over seeds.

| Shield | Success | Death (all unseen hot tiles) | Hazards blocked | Decoys wrongly avoided |
|---|---|---|---|---|
| S0 none | 57.4% [55.1%, 59.4%] | 41.1% [39.0%, 43.6%] | 0% | 0% |
| S1 lookup | 57.4% [55.1%, 59.4%] | 41.1% [39.0%, 43.6%] | 0% | 0% |
| S6 oracle | 92.6% [90.9%, 94.4%] | 0.0% [0.0%, 0.0%] | 100% | 0% |

- **S1 lookup = S0 none, exactly.** Remembering the exact tiles that killed the learner
  gives zero protection on tiles it never met (hypothesis L1-H1: supported).
- **S6 oracle** removes every death and lifts success from 57% to 93%: a shield with
  correct judgements is worth ~35 points of success. The LLM shields (S3, S4) will be
  placed between these two rows once they are run.
- No shield here avoided a decoy; decoy false-avoidance becomes meaningful for S2–S4.

Reproduce:

```bash
python -m research evaluate --shields none,lookup,oracle
```

## Difficulty sweep (hazards per room)

| hazards per room | S0 none death | S1 lookup death | S6 oracle death | S0 success | S6 success |
|---|---|---|---|---|---|
| 1 | 15.9% | 15.9% | 0.0% | 83.2% | 98.7% |
| 2 | 28.8% | 28.8% | 0.0% | 69.9% | 96.5% |
| 4 | 54.5% | 54.5% | 0.0% | 44.3% | 87.2% |
| 6 | 74.6% | 74.6% | 0.0% | 24.3% | 74.3% |

Deaths without a shield grow with the number of hot tiles; the oracle stays at 0%.

```bash
python -m research sweep --knob hazard_count --values 1,2,4,6 --shields none,lookup,oracle
```

Note: the `train` command's own evaluation (`baseline_no_shield.md`, 39.9% deaths) used the
training RNG for tie-breaks; `evaluate` resets it so shields are compared on identical draws
(41.1%). The two confidence intervals overlap.
