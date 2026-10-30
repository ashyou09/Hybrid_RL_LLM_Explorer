# Glossary

Words we use, in plain language, with an example from this project.

## Project terms

| Word | Meaning | Example |
|---|---|---|
| **Semantic Osmosis** | Knowledge "soaks" from one agent to others through verified, readable rules | Agent A dies on lava; Agent B avoids molten rock on its first try |
| **Sparse reflection** | The expensive LLM is only called at important moments (e.g. after failures), not every step | One LLM call after A's training, not one per move |
| **Truth Filter** | A rule must prove itself against real reward before it is trusted | "Hot things kill" is kept only if the death rate drops when it's used |
| **Verified / Rumor / Myth** | Rule states: confirmed / doubtful / proven false but archived | A wrong rule "water kills" becomes a Myth so nobody re-tests it |
| **Shield** | A filter that blocks unsafe actions before the agent takes them | Blocks "move forward" into molten rock |
| **Learner (Agent A)** | The agent that explores and gathers experience | Trains on red lava, campfire, coals |
| **Novice (Agent B)** | A fresh agent that only uses the shared rules | Tested on tiles it has never seen |
| **Decoy** | A harmless tile that looks or sounds like a hazard | Red carpet, painting of a fire |
| **Held-out tile** | A tile kept out of training and used only for testing | Molten rock |
| **Intuitive / counter-intuitive world** | Dangers match / contradict common sense | Counter-intuitive: water kills, fire is safe |
| **Knob** | A setting we change on purpose in an experiment | `hazard_count = 4` |
| **Sweep** | Running the experiment over a range of one knob | `hazard_count` from 1 to 8 |

## Reinforcement learning

| Word | Meaning | Example |
|---|---|---|
| **Reinforcement learning (RL)** | Learning by trial and error from rewards | +10 for the goal, −10 for lava |
| **Q-learning** | RL method that keeps a table of how good each action is in each situation | Q["lava ahead", forward] becomes very negative |
| **Episode** | One attempt from start to end (goal, death, or time-out) | One walk through the room |
| **Policy** | The agent's way of choosing actions | "Pick the action with the highest Q-value" |
| **Epsilon-greedy** | Mostly pick the best action, sometimes a random one to explore | 10% random moves |
| **Action mask** | Making some actions impossible to choose | Shield masks "move forward" |

## Language models

| Word | Meaning | Example |
|---|---|---|
| **LLM / SLM** | Large / small language model | Llama 70B / Llama 8B |
| **Prior** | What the model "already believes" from training | "Fire is dangerous" |
| **Hallucination** | A confident but wrong statement | "Water is always safe" in a world where water kills |
| **Embedding** | A list of numbers representing the meaning of a text | "red lava" → 384 numbers |
| **Cosine similarity** | How close two embeddings are (1 = same meaning) | "red lava" vs "red carpet" may score high — a problem |
| **Prompt** | The text instruction sent to the LLM | "Given these deaths, write rules…" |
| **Cache** | Saved answers so the same question isn't asked twice | Re-running costs 0 API calls |

## Experiments and statistics

| Word | Meaning | Example |
|---|---|---|
| **Baseline** | A simple method to compare against | No shield, exact lookup |
| **Oracle** | A cheating method that knows the truth; the upper limit | Knows which tiles are deadly |
| **Ablation** | Removing one part to see how much it mattered | Run without the memory hub |
| **Seed** | Number that fixes the randomness so runs can be repeated | `seed = 7` gives the same layouts every time |
| **Independent / dependent variable** | What we change / what we measure | hazard count / death rate |
| **Precision** | Of the tiles we blocked, how many were really dangerous | Low if we block red carpets |
| **Recall** | Of the dangerous tiles, how many we blocked | Low if we miss molten rock |
| **Confidence interval (95% CI)** | Range where the true value likely lies | Success 82% ± 4% |
| **Hypothesis** | A claim we test, which may turn out false | L1-H2: the LLM shield avoids most unseen hot tiles |
