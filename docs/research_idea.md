# Research Idea — Summary of the Team Research Doc

A summary of the team's research document ("Semantic Osmosis: Scalable Multi-Agent
Coordination via Grounded Neuro-Symbolic Rule Sharing"), kept in the repo so the
plan and the code stay connected. The full doc is the team's working document.

## The problem: the communication trilemma

- **Pure RL agents** learn from pain but are "mute": they share numbers (weights,
  trajectories) that other agents and humans cannot interpret.
- **Pure LLM agents** can talk, but **hallucinate**: they share plausible rules that
  are physically wrong.
- **Sharing everything** (full logs) floods the network and does not scale.

## The proposed solution

1. **RL body** — learns from real reward; failures are hard evidence.
2. **LLM brain with sparse reflection** — wakes up only on failure and writes a rule
   (IF condition THEN action, with scope and confidence).
3. **Truth Filter** — a rule is used only after it is **verified against reward**.
4. **Knowledge states** — Verified → Rumor → **Myth** (failed rules are archived, not
   deleted, so nobody wastes time testing them again).
5. **Trust-weighted sharing** — rules spread according to the source agent's trust
   score; trust speeds verification up but never skips it.
6. **Compact semantic packets** — share small rules, not raw logs.

## Research questions in the doc

| RQ | Question | Plan level |
|---|---|---|
| RQ1 | When should an agent verify? | Level 4 |
| RQ2 | What is the minimal effective verification? | Levels 1, 4 |
| RQ3 | How should claims be represented? | Level 1 (rule schema) |
| RQ4 | Can errors be traced to an agent / claim? | Levels 1, 3 (evidence links) |
| RQ5 | Does verification improve system reliability? | Levels 1–3 |
| RQ6 | Can small models do the verification? | Level 4 |
| RQ7 | Does verification reduce idea diversity? | Level 5 |
| RQ8 | Which errors are hardest to verify? | Stretch goal |

## Experiments named in the doc

| ID | Strategy | Plan level |
|---|---|---|
| E1 | No verification | Level 4 |
| E2 | Always verify | Level 4 |
| E3 | Failure-based (current method) | Level 4 |
| E4 | Periodic, every 5 steps | Level 4 |
| E5 | Periodic, every 10 steps | Level 4 |
| E6 | Noisy memory, 10% wrong rules | Level 3 |
| E7 | Noisy memory, 30% wrong rules | Level 3 |
| A1–A3 | Ablations: no memory, no action mask, limited rule count | Level 1 / 3 |
| Villa | 20 agents with roles in a 25×25 village | Level 6 |

## How the plan changes the doc's design

- **Unseen hazards and decoys** are added, so the LLM has to do something a lookup
  table cannot (otherwise the LLM adds nothing measurable).
- **Counter-intuitive worlds** make the doc's "hallucination test" concrete.
- **Expected numbers** in the doc (e.g. "99% bandwidth", "60–90% survival") are
  treated as hypotheses until measured.
- **More seeds** (≥ 10 instead of 3) and confidence intervals.
- **EFF = SR / C** is reported alongside SR and C, because it is undefined when C = 0.

See [future_plan/01_current_state_review.md](../future_plan/01_current_state_review.md)
for the full list.
