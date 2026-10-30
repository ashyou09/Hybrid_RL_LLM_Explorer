# Baseline: Agent A with no shield (S0)

Agent A trained by tabular Q-learning for 6 000 episodes in training worlds, then
evaluated greedily over 10 seeds × 100 fresh layouts per split (95% bootstrap CI over seeds).

| Worlds | Success | Death | Time-out |
|---|---|---|---|
| Training worlds (seen hot tiles + plain ground) | 90.7% [88.6%, 92.6%] | 0.4% [0.0%, 0.9%] | 8.9% [7.2%, 10.8%] |
| Test worlds (never-seen hot tiles + decoys) | 59.1% [57.2%, 61.0%] | 39.9% [38.3%, 41.6%] | 1.0% [0.4%, 1.6%] |

Deaths in test worlds by tile (1 000 episodes): molten rock: 153, white-hot metal plate: 133, burning log pile: 113.
No decoy ever killed the agent (decoys are harmless by design).

**Reading:** the learner avoids hot tiles it has met (0.4% deaths), but walks into
never-seen hot tiles in ~40% of test episodes. This is the gap the LLM rules must close.

Reproduce (code from the L1.5 commit onward; deterministic for a given seed):

```bash
python -m research train --config configs/level1.yaml
```
