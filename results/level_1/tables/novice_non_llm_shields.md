# New robot in test worlds: shields that need no AI (S0, S1, S6)

Same trained Agent A as the baseline (10 000 episodes, stuck sense on). Test worlds contain
only NEW hot tiles (molten rock, burning log pile, white-hot metal plate) and harmless
look-alikes. 10 seeds × 100 episodes per shield, paired (same layouts and random draws
for every shield); 95% bootstrap CI over seeds.

| Shield | Reached goal | Died (on a new hot tile) | Got stuck (ran out of time) | Dangers blocked | False alarms (harmless tiles avoided) |
|---|---|---|---|---|---|
| S0 · No help | 57.5% [54.5%, 60.5%] | 42.4% [39.4%, 45.3%] | 0.1% [0.0%, 0.3%] | 0% | 0% |
| S1 · Memory of past killers | 57.5% [54.5%, 60.5%] | 42.4% [39.4%, 45.3%] | 0.1% [0.0%, 0.3%] | 0% | 0% |
| S6 · Perfect knowledge (cheat) | 97.8% [97.0%, 98.6%] | 0.0% [0.0%, 0.0%] | 2.2% [1.4%, 3.0%] | 100% | 0% |

- **Memory of past killers (S1) = No help (S0), exactly.** Remembering the exact tiles that
  killed the learner gives zero protection on tiles it never met (hypothesis L1-H1: supported).
- **Perfect knowledge (S6)** removes every death and lifts reaching the goal from 57.5% to
  97.8%: a shield with correct judgement is worth ~40 points. The AI shields (S3, S4) will
  land between these rows.
- Nobody avoided a look-alike here; false alarms become meaningful for S2–S4.

```bash
python -m research evaluate --shields none,lookup,oracle
```

## Difficulty sweep (hot tiles per room)

| Hot tiles per room | No help: died | Memory of past killers: died | Perfect knowledge: died | No help: reached goal | Perfect knowledge: reached goal |
|---|---|---|---|---|---|
| 1 | 13.8% | 13.8% | 0.0% | 86.1% | 99.9% |
| 2 | 30.2% | 30.2% | 0.0% | 69.7% | 99.4% |
| 4 | 53.9% | 53.9% | 0.0% | 45.7% | 93.4% |
| 6 | 74.7% | 74.7% | 0.0% | 24.9% | 80.4% |

```bash
python -m research sweep --knob hazard_count --values 1,2,4,6 --shields none,lookup,oracle
```
