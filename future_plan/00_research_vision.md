# 00 — Research Vision

## One-paragraph story

Agents that learn alone repeat each other's mistakes. Agents that share LLM-written
advice spread hallucinations. We study a middle path, **Semantic Osmosis**: one
agent learns from physical failure (RL), an LLM turns those failures into compact,
human-readable rules, the rules are **verified against reality** before they are
trusted, and a *new* agent uses them on its first try — including on hazards
**nobody has ever encountered**, by reasoning about what the rules mean.

## The first paper's question (Exp-1)

> Can safety rules that an LLM writes from a learner agent's few deaths let a
> **novice agent** avoid **never-seen hazards** zero-shot, while **not** avoiding
> harmless look-alikes — and does **verification against reward** stop wrong
> (hallucinated or noisy) rules from being used?

Why this question:

- A lookup table ("red lava = deadly") cannot answer it: the test hazards were never
  seen. This is where the LLM must earn its place.
- An LLM with no experience cannot answer it either, because some worlds
  **contradict common sense** (e.g. water deadly, fire harmless). The agent has to
  combine **LLM priors** with **its own evidence**. This is also the doc's planned
  "hallucination test": the LLM's plausible assumption is wrong, and verification
  must catch it.

## Hypotheses (to be measured, not assumed)

| ID | Hypothesis | Measured by |
|---|---|---|
| H1 | LLM-judge + experience beats exact lookup and embedding retrieval on unseen hazards | Hazard-death rate on held-out tiles |
| H2 | Embedding retrieval over-blocks look-alike decoys ("red carpet" ≈ "red lava") | False-avoidance rate on decoys |
| H3 | LLM without experience fails in counter-intuitive worlds; with experience it adapts | Death rate split by world type |
| H4 | Reward-based verification filters injected wrong rules (10% / 30% noise) | Wrong-rule usage, recovery rate |
| H5 | Sparse reflection (LLM on failure only) matches always-verify at a fraction of the LLM calls | Success rate vs. LLM calls (EFF) |
| H6 | A rule packet is orders of magnitude smaller than the raw trajectory log it came from | Bytes per packet vs. bytes per log |
| H7 | Smaller models (8B) approach large-model accuracy as the rule judge | Same metrics across model sizes |

## Mapping the doc's research questions to work

| Doc RQ | Where it is answered |
|---|---|
| RQ1 When to verify? | Exp-1 trigger conditions E1–E5 (none / always / on-failure / periodic K=5, K=10) |
| RQ2 Minimal effective verification | Shield comparison: lookup vs. embedding vs. LLM-judge vs. oracle |
| RQ3 Structured claim representation | Rule schema (IF/THEN + scope + confidence + evidence ids), Level 1 |
| RQ4 Error attribution | Each rule links to the evidence events that produced it; failures trace back to a rule id |
| RQ5 System-level reliability | Exp-1 main results table (Level 1) |
| RQ6 Can SLMs approximate verification? | Model-size comparison (H7) |
| RQ7 Verification vs. collaboration diversity | Deferred to Exp-2 (Level 5) |
| RQ8 Which errors are hardest to verify? | Partially: delayed-effect hazards as a stretch goal |

## Scope: what is in and out of the first paper

**In (Exp-1):** one learner agent (Agent A), one novice (Agent B), text-layer
MiniGrid, hidden danger categories, verification loop, Verified/Rumor/Myth states,
noisy-rule injection, 3+ LLMs.

**Out (Exp-2, later):** trust scores across many agents, the 20-agent Villa,
ShareScore bandwidth policy, 3-tier communication. These build on Exp-1 results;
doing them first would mean building a large system before the core claim is shown
to hold.
