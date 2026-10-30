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
| [00_research_vision.md](00_research_vision.md) | The research question, the claim, how the doc's RQs map onto experiments |
| [01_current_state_review.md](01_current_state_review.md) | Honest review of today's code and doc: what holds, what a reviewer will attack |
| [02_roadmap.md](02_roadmap.md) | Milestones M0–M7, each broken into small steps with a "done when" check |
| [03_exp1_design.md](03_exp1_design.md) | Exp-1 in detail: worlds, tiles, agents, conditions, metrics, statistics |
| [04_open_questions.md](04_open_questions.md) | Decisions still needed from the team, plus a decision log |

## Status board

| Milestone | Title | Status |
|---|---|---|
| M0 | Foundations (seeds, headless runs, tests) | ⬜ not started |
| M1 | Text-layer environment with hidden danger categories | ⬜ not started |
| M2 | Real RL body (tabular Q-learning + action-mask hook) | ⬜ not started |
| M3 | LLM brain: rule schema, provider adapter, shields | ⬜ not started |
| M4 | Truth Filter: verification loop + Verified/Rumor/Myth | ⬜ not started |
| M5 | Exp-1 evaluation harness and results | ⬜ not started |
| M6 | Exp-2: multi-agent Osmosis (2 → N agents, trust score) | ⬜ not started |
| M7 | Paper writing | ⬜ not started |

Legend: ⬜ not started · 🟨 in progress · ✅ done

## Ground rules

1. **Every number in the paper comes from a script in this repo**, run with fixed
   seeds, and the command is written next to the number.
2. **Expected results are hypotheses, not facts.** Claims like "99% bandwidth
   reduction" or "60–90% survival improvement" stay labelled *H#* until measured.
3. **The old demo (`app.py`, `run_experiment.py`, HF Space) keeps working.** New
   research code lives in a separate headless package (`research/`).
4. **Small steps.** Each step in the roadmap should take roughly half a day to
   two days and end in a commit.
