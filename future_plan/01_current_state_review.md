# 01 — Current State Review

An honest reviewer's pass over the code (`main` at `04a3a12`) and the research doc.

## What already works

- **Semantic, not positional, logging.** Deaths are logged by tile type ("red lava"),
  and Phase 3 uses fresh random layouts. This is the right core idea.
- **All-four-neighbour check** before moving (the "infinite hump" fix) is the right
  way to apply a rule as an action filter (shielding).
- **Sparse reflection** already exists: the LLM is called once after Phase 1, not per step.
- **A deterministic fallback** when the LLM fails keeps runs from crashing.

## Gaps in the code a reviewer will attack

| # | Issue | Where | Fix (level) |
|---|---|---|---|
| C1 | Phase 1 is not RL: the per-tile penalty table is cleared each episode, so nothing is learned across episodes | `src/core/rl_core.py` `reset_episode()` | Real tabular Q-learning (Level 1) |
| C2 | The LLM adds no measurable value: the log already names the killer tile, so `_fallback_rules()` gives identical behaviour | `src/agents/reflection_engine.py` | Held-out hazards + counter-intuitive worlds (Level 1) |
| C3 | Agents read the full map (`env.unwrapped.grid`), including the tile behind them | `rl_core.py`, `planner_agent.py` | Local perception only (Level 1) |
| C4 | No seeds, no baselines, about 2 test episodes: no statistics possible | `run_experiment.py` | Evaluation harness (Level 1) |
| C5 | Embedding threshold 0.65 untested: may block "red carpet" or miss "acid" | `planner_agent.py` | Measured as H2 (Level 1) |
| C6 | No verification step: LLM rules are used as soon as they're produced | `run_experiment.py` | Truth Filter (Level 3) |
| C7 | Rules only constrain "Move Forward"; other fields are unused | `reflection_engine.py` | Structured rule schema (Level 1) |

## Gaps in the research doc

| # | Issue | Suggested fix |
|---|---|---|
| D1 | E1–E5 are marked "Completed", but no code or results for them are in this repo | Locate the code/results, or re-run them inside the new harness (Level 4) |
| D2 | Numbers like "<10 ms", "~500 ms", "99% bandwidth", "60–90% survival", "3–5× faster" read as results | Rewrite as hypotheses H1–H7 until measured |
| D3 | 3 seeds is weak for a paper | Use ≥10 seeds × 100 episodes and report 95% CIs |
| D4 | EFF = SR / C is undefined for E1 (C = 0) and rewards doing nothing | Report SR and C side by side; plot the trade-off curve; use SR / (1 + C) if a single number is needed |
| D5 | "Verified if reward improves" is not precise | Define a test: e.g. paired comparison over N probe episodes, rule kept if mean return improves and death rate drops (Level 3) |
| D6 | Exact-coordinate rules ("avoid (5,5)") contradict the semantic, position-free goal | Rules are about tile *properties*, never coordinates |
| D7 | Citation inconsistency: LangGround is credited to "H. Li et al." in one place and "Zhang et al." in another | Check every reference against the actual paper (before writing any paper.md) |
| D8 | Some references could not be checked here (e.g. "Multi-Agent Memory from a Computer Architecture Perspective", SIGARCH 2026; ReAd venue) | Verify title, authors, venue, year before citing |
| D9 | Tech stack lists Ollama-only; team now plans free API tiers + Claude | Provider adapter supporting Groq, OpenRouter, Claude, Ollama (Level 1) |
