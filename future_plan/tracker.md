# Tracker

Legend: ⬜ not started · 💬 discussing · 🟨 building · ✅ done

## Levels

| Level | Name | Status | Started | Finished | Conclusion |
|---|---|---|---|---|---|
| 1 | First Transfer | 🟨 building | 2026-10-30 | — | [conclusion/level_1](../conclusion/level_1/) |
| 2 | Experience vs. Common Sense | ⬜ | — | — | — |
| 3 | The Truth Filter | ⬜ | — | — | — |
| 4 | When to Think | ⬜ | — | — | — |
| 5 | Two Agents Share | ⬜ | — | — | — |
| 6 | The Village | ⬜ | — | — | — |

## Level 1 steps

| Step | What | Status | Commit / notes |
|---|---|---|---|
| L1.1 | `research/` package, seeds, tests, knob config | ✅ | 72b7e04 |
| L1.2 | Tile catalog | ✅ | 1c98460 |
| L1.3 | Text-layer env + layouts + local perception | ✅ | 2a693fb |
| L1.4 | Episode replay (GIF) | ✅ | d007414 |
| L1.5 | Agent A: tabular Q-learning | ✅ | 398aa33 — baseline: 39.9% deaths on unseen hot tiles |
| L1.6 | Experience log | ✅ | this commit |
| L1.7 | LLM adapter + cache | ⬜ | |
| L1.8 | Reflection → rules | ⬜ | |
| L1.9 | Shields (+ decisions in replay) | ⬜ | |
| L1.10 | Tile-classification benchmark (first result) | ⬜ | |
| L1.11 | Novice evaluation | ⬜ | |
| L1.12 | Sweeps: difficulty, strictness, experience | ⬜ | |
| L1.13 | Lab panel (Streamlit) | ⬜ | |
| L1.14 | Write conclusion (paper + explained) | ⬜ | |
| L1.15 | Handover to Level 2 | ⬜ | |

## Handover notes

*(Written at the end of each level: what was learned, what changes for the next level.)*

## Notes along the way

- **L1.5:** with state = (front tile, side kinds, goal compass) the learner got stuck
  turning back and forth (39% time-outs). Adding the previous action to the state
  fixed it (90.7% success in training worlds). Random actions at evaluation also
  break loops but cause deaths, so they were not used.
- **L1.5 baseline (no shield):** 0.4% deaths in training worlds vs. 39.9% deaths on
  never-seen hot tiles — the gap the LLM rules must close.
  See `results/level_1/tables/baseline_no_shield.md`.
- **Fix:** Hugging Face Space repaired (Gradio header, current pipeline in `app.py`,
  snapshot deploy). Needs the `HF_TOKEN` GitHub secret to deploy.
