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

## Open questions

1. **E1–E5 results.** The research doc marks E1–E5 "Completed", but no code or
   results for them are in this repo. Where are they? Re-run inside the new harness?
2. **First paper scope.** Exp-1 only (learner → novice), with multi-agent trust
   (Exp-2) as future work — or must the first paper include multiple agents?
3. **Team split.** Who owns which milestone? (Doc colour code lists Sanchit.)
4. **Target venue.** Which workshop? This sets page limit and deadline.
5. **Verification definition.** How many probe episodes, what threshold? (Proposal in roadmap 4.1.)
6. **Delayed-effect hazards** (poison that kills a few steps later) — stretch goal
   for doc RQ8, or out of scope?
