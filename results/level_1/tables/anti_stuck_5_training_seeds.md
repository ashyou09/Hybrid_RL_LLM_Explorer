# Anti-stuck test: 5 independent training runs per setting

The learner sometimes turned on the spot or walked back and forth until time ran out.
Each setting was trained 5 times (training seeds 0–4, 10 000 episodes each) and
evaluated on 5 seeds × 100 episodes; values are mean ± standard deviation over the
5 training runs. "Perfect knowledge" (S6) removes all deaths, so its failures are
purely getting stuck.

| Fix | Learner success (training rooms) | No help: reached goal | No help: died | Perfect knowledge: reached goal | Perfect knowledge: got stuck |
|---|---|---|---|---|---|
| current | 70.8% ± 16.5% | 46.0% ± 11.5% | 45.6% ± 3.9% | 76.6% ± 15.9% | 23.4% ± 15.9% |
| stuck sense | 90.1% ± 3.2% | 52.1% ± 4.1% | 46.5% ± 4.7% | 94.0% ± 3.2% | 6.0% ± 3.2% |
| revisit penalty | 75.6% ± 19.3% | 48.6% ± 12.3% | 49.0% ± 11.1% | 81.4% ± 17.7% | 18.6% ± 17.7% |
| all three | 88.7% ± 3.0% | 52.4% ± 2.2% | 45.5% ± 2.2% | 91.6% ± 2.3% | 8.4% ± 2.3% |

- **stuck sense** (the agent senses "I have not moved for 3 steps") is clearly best and
  the most reliable: stuck 6.0% ± 3.2% vs. 23.4% ± 15.9% without it. It is now **on by default**.
- **revisit penalty** (extra cost for walking over its own trail, the v0 idea) does not
  help on its own; combined with trail memory and stuck sense it is close to stuck sense alone.
- The previous setup was **unreliable**: one training run reached 90.7%, the average over
  five is 70.8% ± 16.5%. Results must therefore be reported over several training seeds.

Raw numbers: `anti_stuck_5_training_seeds.json`. Script settings: knobs `stuck_signal`,
`trail_memory`, `revisit_penalty` in `configs/level1.yaml`.
