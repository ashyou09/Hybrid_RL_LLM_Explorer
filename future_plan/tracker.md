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
| L1.6 | Experience log | ✅ | 9ead037 |
| L1.7 | LLM adapter + cache | ✅ | 863bbc8 |
| L1.8 | Reflection → rules | ✅ | 2707f91 |
| L1.9 | Shields (+ decisions in replay) | ✅ | 7e7ab93 |
| L1.10 | Tile-classification benchmark (first result) | 🟨 | ec1715f: built; LLM/embedding runs pending |
| L1.11 | Novice evaluation | 🟨 | ec1715f: S0/S1/S6 measured; S2-S4 pending |
| L1.12 | Sweeps: difficulty, strictness, experience | 🟨 | ec1715f: difficulty sweep (non-LLM) done |
| L1.13 | Lab panel (Streamlit) | ✅ | tested headless and in a browser |
| L1.14 | Write conclusion (paper + explained) | 🟨 | 41c6eca: draft; LLM sections pending |
| L1.15 | Handover to Level 2 | 🟨 | Level 2 expanded in future.md; waits for LLM runs |

## Handover notes

*(Written at the end of each level: what was learned, what changes for the next level.)*

**Level 1 to Level 2 (partial handover, 2026-10-30).** Everything is built and tested.
Measured: learner 90.7% success in training worlds; on never-seen hot tiles, no shield
and exact lookup both give 41.1% deaths; oracle 0% deaths, 92.6% success.
**Still open:** run S3/S4 (LLM) with Claude Haiku 5.5. This needs `ANTHROPIC_API_KEY` in
the environment settings and a **new session** (settings only reach new sessions); no
network change is needed. S2 (embedding) also needs `sentence-transformers` and access to
huggingface.co (blocked in the cloud session; run it on a laptop or Colab). Then fill the [PENDING] parts of `conclusion/level_1/`. Level 2 is expanded in
`future.md` and starts after those runs.

## Notes along the way

- **L1.5:** with state = (front tile, side kinds, goal compass) the learner got stuck
  turning back and forth (39% time-outs). Adding the previous action to the state
  fixed it (90.7% success in training worlds). Random actions at evaluation also
  break loops but cause deaths, so they were not used.
- **L1.5 baseline (no shield):** 0.4% deaths in training worlds vs. 39.9% deaths on
  never-seen hot tiles — the gap the LLM rules must close.
  See `results/level_1/tables/baseline_no_shield.md`.
- **L1.7:** this cloud session cannot reach api.groq.com, openrouter.ai or huggingface.co
  (network policy); api.anthropic.com is reachable. LLM runs need a key plus network access.
- **L1.11:** evaluation resets the agent's tie-breaking RNG, so shields are compared on
  identical draws; no-shield deaths are 41.1% (39.9% in the train command's own
  evaluation, which used the training RNG; the intervals overlap).
- **Fix:** Hugging Face Space repaired (Gradio header, current pipeline in `app.py`,
  snapshot deploy). Needs the `HF_TOKEN` GitHub secret to deploy.
