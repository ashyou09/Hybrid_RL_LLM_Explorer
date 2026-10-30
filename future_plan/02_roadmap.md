# 02 — Roadmap (small steps)

Each step should be about half a day to two days and end in one commit.
Tick a box when its "done when" check passes.

---

## M0 — Foundations

Goal: make every experiment reproducible and runnable without a screen.

- [ ] **0.1** Create the `research/` package next to `src/` (old demo stays untouched).
  *Done when:* `python -m research --help` prints usage.
- [ ] **0.2** One `seed_everything(seed)` helper (Python `random`, NumPy, env reset seed).
  *Done when:* two runs with the same seed produce identical trajectories (test).
- [ ] **0.3** Add `pytest` and a `tests/` folder; CI-friendly (no network, no GPU).
  *Done when:* `pytest` passes locally.
- [ ] **0.4** Split requirements: `requirements.txt` (demo) and `requirements-research.txt`.
  *Done when:* a fresh venv can install the research set and run the tests.
- [ ] **0.5** A `results/` folder convention: one JSON per run plus the exact command and git hash.
  *Done when:* a dummy run writes `results/<run_id>/config.json` and `metrics.json`.

## M1 — Text-layer environment

Goal: tiles with natural-language descriptions and hidden danger categories.

- [ ] **1.1** Tile catalog: `name`, `description`, `category`, `color` (for drawing), `split` (train / test / decoy).
  Starting categories: heat, liquid, sharp, chemical; plus harmless decoys.
  *Done when:* catalog has ≥ 4 categories × (≥ 3 train + ≥ 3 test tiles) + ≥ 8 decoys.
- [ ] **1.2** World spec: which categories are deadly in this world. Two families:
  *intuitive* (heat/sharp/chemical deadly) and *counter-intuitive* (e.g. liquid deadly, heat safe).
  *Done when:* `WorldSpec.sample(seed, family)` is deterministic and unit-tested.
- [ ] **1.3** MiniGrid env with a `DescribedTile` object (overlappable; kills only if its category is deadly in this world).
  *Done when:* stepping on a deadly tile terminates with −10; a safe tile does not.
- [ ] **1.4** Local perception: the agent observes only the descriptions of front / left / right tiles (no global map, no "behind").
  *Done when:* observation contains no coordinates and no hidden category labels (test).
- [ ] **1.5** Layout generator: random positions per episode, guaranteed path from start to goal.
  *Done when:* 1 000 generated layouts all have a reachable goal (BFS test).
- [ ] **1.6** Rendering hook so the old Pygame display can show the new worlds (optional, for demos).

## M2 — Real RL body (Agent A, the learner)

Goal: honest RL that learns across episodes, with a hook for safety masks.

- [ ] **2.1** Tabular Q-learning; state = (front tile token, relative goal direction); actions = left / right / forward.
  *Done when:* in a world with no hazards, success rate > 90% after training (test, fixed seed).
- [ ] **2.2** Action-mask hook: `act(obs, mask)` never picks a masked action.
  *Done when:* unit test with "forward" masked never moves forward.
- [ ] **2.3** Experience log: every episode stores structured events — `died_on`, `walked_safely_on`, `reached_goal` — with tile descriptions, never coordinates.
  *Done when:* log schema is validated by a test.
- [ ] **2.4** Training curve script: success / death rate vs. episodes.
  *Done when:* a PNG curve is saved to `results/`.

## M3 — LLM brain (Sparse Reflection) and shields

Goal: turn the experience log into structured rules, and judge new tiles.

- [ ] **3.1** Rule schema (answers doc RQ3): `id`, `if` (tile property), `then` (forbid action), `scope`, `confidence`, `evidence_ids`, `status` (verified / rumor / myth).
- [ ] **3.2** Provider adapter: one `complete(prompt)` interface for Groq, OpenRouter, Claude, Ollama; keys via environment variables.
  *Done when:* a fake provider passes tests; real providers work with a key.
- [ ] **3.3** Disk cache keyed by (provider, model, prompt hash), so each tile is asked once.
  *Done when:* the second run of the same experiment makes zero API calls.
- [ ] **3.4** Reflection prompt: experience log → rules (category-level, not tile-level).
- [ ] **3.5** Shields (all share one interface `is_dangerous(description) -> bool`):
  - none · exact lookup · embedding retrieval (current ChromaDB method) · LLM prior (no experience) · **LLM + experience (ours)** · oracle
  *Done when:* each shield passes a small labelled test set.
- [ ] **3.6** Tile-classification benchmark: precision / recall of each shield on held-out tiles and decoys, no RL needed.
  *Done when:* a table is printed for any provider (cheap first signal, before full runs).

## M4 — Truth Filter (verification)

Goal: rules must prove themselves against reward before being trusted.

- [ ] **4.1** Precise verification test: run N probe episodes with and without the rule; keep it if death rate drops and return does not fall.
- [ ] **4.2** Knowledge state machine: candidate → verified / rumor → myth, with thresholds as config.
- [ ] **4.3** Myth archive: an incoming rule that matches a myth is rejected without re-testing; count tests saved.
- [ ] **4.4** Noise injection (doc E6/E7): replace 10% / 30% of rules with wrong ones (flip deadly ↔ safe).
- [ ] **4.5** Verification triggers (doc E1–E5): none, always, on failure, periodic K=5, periodic K=10.
  *Done when:* each trigger logs how many LLM calls it made.

## M5 — Exp-1 evaluation and results

Goal: the main results table of the paper.

- [ ] **5.1** Harness: `python -m research.evaluate --conditions ... --seeds 10 --episodes 100`.
- [ ] **5.2** Metrics: success rate, hazard-death rate (seen / unseen), decoy false-avoidance, steps, LLM calls, tokens, wall time, bytes per rule packet vs. raw log (H6).
- [ ] **5.3** Statistics: 95% confidence intervals (bootstrap), effect sizes between conditions.
- [ ] **5.4** Run the cheap pilot (oracle, none, lookup, embedding) — no API cost.
- [ ] **5.5** Run LLM conditions on one free model; check the pipeline end to end.
- [ ] **5.6** Run the model-size comparison (small free model vs. large free model vs. Claude) (H7).
- [ ] **5.7** Plots: main bar chart, intuitive vs. counter-intuitive split, SR vs. LLM calls trade-off, noise robustness.
- [ ] **5.8** Ablations from the doc: A1 no memory, A2 no action mask, A3 limited rule count.

## M6 — Exp-2: Multi-agent Osmosis (after Exp-1 holds)

- [ ] **6.1** Two learners in different worlds share rules through the memory hub.
- [ ] **6.2** Trust score T (role accuracy, past accuracy ratio, experience depth, recency decay).
- [ ] **6.3** A corrupted agent that shares wrong rules; measure spread with and without verification.
- [ ] **6.4** Scale to N agents; measure redundant trap hits (silent vs. osmosis mode).
- [ ] **6.5** Villa environment (25×25, roles) — only if 6.1–6.4 show a clear effect.

## M7 — Paper

- [ ] **7.1** Verify every citation (title, authors, venue, year) and fix the LangGround attribution.
- [ ] **7.2** Related work: shielding (Alshiekh et al., AAAI 2018), LLM-guided RL (ELLM, Du et al., ICML 2023), Reflexion (Shinn et al., 2023), zero-shot planners (Huang et al., ICML 2022), Generative Agents (Park et al., 2023), LangGround (Li et al., NeurIPS 2024).
- [ ] **7.3** Pick a target workshop and its page limit / deadline.
- [ ] **7.4** Draft: intro → method → Exp-1 → limitations → future work (Exp-2).
- [ ] **7.5** Reproducibility appendix: commands, seeds, models, cost.
