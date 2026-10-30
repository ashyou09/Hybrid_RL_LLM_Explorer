# Baseline: Agent A with no shield (S0)

Agent A trained by tabular Q-learning for 10 000 episodes in training worlds (stuck sense
on), then evaluated greedily over 10 seeds × 100 fresh layouts per split (95% bootstrap
CI over seeds).

| Worlds | Reached goal | Died | Got stuck |
|---|---|---|---|
| Training worlds (known hot tiles + plain floor) | 94.1% [92.7%, 95.6%] | 0.7% [0.3%, 1.1%] | 5.2% [3.9%, 6.6%] |
| Test worlds (NEW hot tiles + look-alikes) | 57.5% [54.5%, 60.5%] | 42.4% [39.4%, 45.3%] | 0.1% [0.0%, 0.3%] |

Deaths in test worlds by tile (1 000 episodes): molten rock: 147, white-hot metal plate: 143, burning log pile: 134. No look-alike ever killed the agent
(they are harmless by design).

**Reading:** the learner avoids the hot tiles it has met, but walks into NEW hot tiles in
~42% of test episodes. This is the gap the AI rules must close.

Reproduce:

```bash
python -m research train
```

History: before stuck sense was switched on (6 000 episodes, one training run) the same
table read 90.7% / 39.9%; see `anti_stuck_5_training_seeds.md` for why it changed.
