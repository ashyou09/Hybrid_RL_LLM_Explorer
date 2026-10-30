# Future Plan — Semantic Osmosis (Exp-1 → Paper)

This folder is the working plan for turning the Hybrid RL–LLM Explorer into a
publishable research project. It merges:

- the team research doc ("Semantic Osmosis: Scalable Multi-Agent Coordination via
  Grounded Neuro-Symbolic Rule Sharing"), and
- the design decisions agreed in discussion (zero-shot hazard transfer, mixed
  intuitive / counter-intuitive worlds, real Q-learning, MiniGrid + text layer,
  embedding vs LLM-judge comparison).

## Files

| File | What it contains |
|---|---|
| [future.md](future.md) | **The main plan:** 6 levels from very small to large; Level 1 in full detail |
| [tracker.md](tracker.md) | Status of every level and step, plus handover notes |
| [00_research_vision.md](00_research_vision.md) | The overall research story, hypotheses, mapping of the doc's RQs |
| [01_current_state_review.md](01_current_state_review.md) | Honest review of today's code and doc: what holds, what a reviewer will attack |
| [04_open_questions.md](04_open_questions.md) | Decisions still needed, plus a decision log |

Write-ups for each level live in [`../conclusion/`](../conclusion/): a paper-format
version and a plain-language version.

## Workflow

For every step: **discuss → build → push → review**. No step is built before we
have talked through its design.

## Ground rules

1. **Every number in the paper comes from a script in this repo**, run with fixed
   seeds, and the command is written next to the number.
2. **Expected results are hypotheses, not facts.** Claims like "99% bandwidth
   reduction" or "60–90% survival improvement" stay labelled *H#* until measured.
3. **The old demo (`app.py`, `run_experiment.py`, HF Space) keeps working.** New
   research code lives in a separate headless package (`research/`).
4. **Small steps.** Each step in a level should take roughly half a day to
   two days and end in a commit.
