---
title: Hybrid RL-LLM Explorer
emoji: 🧠
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# Hybrid RL–LLM Explorer — Semantic Osmosis

**Research question:** can an agent learn from its own failures, have an LLM turn
those failures into human-readable rules, **verify** those rules against reality,
and pass them to a *new* agent that then survives dangers **nobody has seen before**?

We call this **Semantic Osmosis**: *one learns, all learn* — through verified,
interpretable rules instead of raw data or unchecked LLM advice.

> 🚧 **Status:** research in progress. Currently **Level 1 — First Transfer**,
> design stage. See the [tracker](future_plan/tracker.md).

---

## The idea in one picture

```
 Agent A (learner)          LLM (sparse reflection)          Agent B (novice)
 ─────────────────          ───────────────────────          ────────────────
 explores with RL   ──►     reads A's deaths, writes   ──►   meets NEW tiles,
 dies on red lava,          rules: "hot things kill"         asks the rules:
 campfire, coals            (+ evidence, confidence)         "molten rock?" → avoid
                                     │                        "red carpet?" → safe
                                     ▼
                       Truth Filter: rules are tested
                       against reward before trusted
                       (Verified → Rumor → Myth)
```

## Research plan: 6 levels, smallest first

| Level | Name | Question | Status |
|---|---|---|---|
| 1 | First Transfer | Do rules from a few deaths protect a novice on never-seen hazards, without fearing look-alikes? | 💬 designing |
| 2 | Experience vs. Common Sense | When the world breaks common sense, does experience override the LLM? | ⬜ |
| 3 | The Truth Filter | Does testing rules against reward stop wrong rules? | ⬜ |
| 4 | When to Think | How often to call the LLM; can a small model replace a big one? | ⬜ |
| 5 | Two Agents Share | Does verified sharing beat learning alone; does trust stop a liar? | ⬜ |
| 6 | The Village | Does it scale to many agents? | ⬜ |

Full plan: [future_plan/future.md](future_plan/future.md) ·
Why this design: [future_plan/00_research_vision.md](future_plan/00_research_vision.md)

## Repository map

| Path | What it is |
|---|---|
| [`future_plan/`](future_plan/) | **The plan:** levels, tracker, research vision, review of current state, decisions |
| [`conclusion/`](conclusion/) | Write-ups per level: `paper.md` (paper format) + `explained.md` (plain language) |
| [`docs/`](docs/) | Research idea summary, glossary, references, setup guide, legacy demo notes |
| `research/` | *(coming in Level 1)* new headless research code: environment, agents, shields, evaluation |
| `results/` | *(coming in Level 1)* configs, metrics, plots and replays of every run |
| [`src/`](src/), [`run_experiment.py`](run_experiment.py), [`app.py`](app.py) | **v0 demo** — the original prototype (kept working; see [docs/legacy_demo.md](docs/legacy_demo.md)) |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How we work: discuss → build → push → review |

## Tools

| Part | Choice |
|---|---|
| Environment | Gymnasium + MiniGrid, with a natural-language text layer |
| RL | Tabular Q-learning (no GPU needed) |
| LLM | Open-weight Llama models via Groq free tier · Ollama offline · OpenRouter backup · Claude as reference |
| Visuals | Episode replays (GIF) + Streamlit Lab panel |

Setup: [docs/setup.md](docs/setup.md)

## Running the v0 demo

```bash
pip install -r requirements.txt
```

```bash
python run_experiment.py
```

The v0 demo uses a local Ollama model (`llama3.2:3b`) if available, otherwise a
deterministic fallback rule engine.

## Team

Research idea and direction: the project team (see the team research doc).
Contributions follow [CONTRIBUTING.md](CONTRIBUTING.md).
