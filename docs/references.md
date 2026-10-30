# References

Status: ✅ checked against the paper · ⚠️ needs checking before citing.
Only ✅ references go into a `paper.md`.

## Used in the research idea

| Reference | Used for | Status |
|---|---|---|
| Huang, Abbeel, Pathak, Mordatch — *Language Models as Zero-Shot Planners*, ICML 2022 | LLM plans are often not executable (grounding gap) | ⚠️ check executability numbers quoted in the doc |
| Park et al. — *Generative Agents: Interactive Simulacra of Human Behavior*, 2023 | Cost of calling the LLM every step; unverified shared "facts" | ⚠️ |
| *Language Grounded Multi-Agent Reinforcement Learning with Human-Interpretable Communication* (LangGround), NeurIPS 2024 | Interpretable communication between RL agents | ⚠️ doc credits "H. Li et al." and "Zhang et al." in different places — check authors |
| *Towards Efficient LLM Grounding for Embodied Multi-Agent Collaboration* (ReAd) | Feedback from RL to refine LLM plans | ⚠️ check venue and year |
| *Multi-Agent Memory from a Computer Architecture Perspective* (SIGARCH, 2026) | Memory-sharing bottleneck | ⚠️ could not be confirmed — check it exists as cited |

## To add for related work

| Reference | Why relevant | Status |
|---|---|---|
| Alshiekh et al. — *Safe Reinforcement Learning via Shielding*, AAAI 2018 | Our shield is a learned version of this idea | ⚠️ |
| Du et al. — *Guiding Pretraining in Reinforcement Learning with Large Language Models* (ELLM), ICML 2023 | LLMs guiding RL agents | ⚠️ |
| Shinn et al. — *Reflexion: Language Agents with Verbal Reinforcement Learning*, 2023 | Learning from failures through language reflection | ⚠️ |
| Chevalier-Boisvert et al. — MiniGrid / Minigrid & Miniworld | The environment we use | ⚠️ |

## How to check a reference

1. Find the paper on the venue site or arXiv.
2. Confirm exact title, all authors, venue and year.
3. Confirm the claim we cite actually appears in the paper (note the section or page).
4. Change ⚠️ to ✅ and add the link.
