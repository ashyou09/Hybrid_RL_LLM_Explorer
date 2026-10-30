# 04 — Open Questions and Decision Log

## Decision log (agreed)

| Date | Decision |
|---|---|
| 2026-10-30 | Goal: paper / workshop submission. No fixed deadline yet. |
| 2026-10-30 | Core claim: zero-shot transfer to unseen hazards. |
| 2026-10-30 | Worlds: mix intuitive and counter-intuitive danger rules. |
| 2026-10-30 | Test-time decision: compare embedding retrieval vs. LLM judge. |
| 2026-10-30 | RL body: tabular Q-learning with action masking (shield). |
| 2026-10-30 | Environment: MiniGrid + natural-language text layer. |
| 2026-10-30 | Danger design: hidden categories, held-out tiles, decoys. |
| 2026-10-30 | LLM providers: Groq (free), OpenRouter (free), Claude API. |
| 2026-10-30 | Keep the old demo; build research code separately. |
| 2026-10-30 | Work in 6 levels, smallest first; Level 1 in full detail, later levels expanded at handover. |
| 2026-10-30 | Each level is written up in `conclusion/level_N/` as paper.md + explained.md. |
| 2026-10-30 | E1–E5 will be re-run in the new code (Level 4). |
| 2026-10-30 | Workflow: discuss each step before building it. |
| 2026-10-30 | Level 1 tile set (heat category) accepted. |
| 2026-10-30 | Add a visual Lab: knob config, episode replays, sweeps, Streamlit panel. |
| 2026-10-30 | Build order: foundations first (L1.1 → L1.15). |
| 2026-10-30 | LLM: open-weight Llama models via Groq free tier (Ollama for offline); OpenRouter backup; Claude as reference. |
| 2026-10-30 | Run in the cloud (cloud session / Colab / HF Space); laptop needs only a browser. |

| 2026-10-30 | Level 1 LLM runs use Claude Haiku 4.5 (`claude-haiku-4-5`), chosen by the team; Groq/OpenRouter remain supported. |

## Open questions

1. **Team split.** Who owns which milestone? (Doc colour code lists Sanchit.)
2. **Target venue.** Which workshop? This sets page limit and deadline.
3. **Verification definition.** How many probe episodes, what threshold? (Level 3.)
4. **Delayed-effect hazards** (poison that kills a few steps later) — stretch goal
   for doc RQ8, or out of scope?
