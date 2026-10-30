# Level 1 — First Transfer, Explained Simply

> **Status:** everything except the LLM runs is done and measured. The LLM shields (S3,
> S4) and the embedding shield (S2) are built and tested, but have not been run with a
> real model yet — section 5 says exactly what is measured and what is still open.

## 1. What question did we ask?

If an agent dies on a few hot tiles (red lava, hot coals), can an LLM turn that into a
rule that protects a **new** agent on hot tiles **nobody has ever seen** (molten rock,
burning logs, white-hot metal) — without making it afraid of harmless look-alikes
(red carpet, a painting of a fire)?

## 2. Why does it matter?

The old version of this project logged "red lava killed me" and later avoided "red
lava". That is a lookup table: it can never help with something new. If the LLM is
going to earn its place, it must handle things the agent has **not** experienced.

## 3. What did we build?

- **A small world** (7×7 room). Some floor tiles are hot and kill; others are harmless.
  Each tile has a name and a short description. The agent only sees the three tiles
  around it (front, left, right) and a compass pointing at the goal — no map, no
  coordinates, no "this is dangerous" label.
- **Agent A, the learner.** It learns by trial and error (Q-learning): it keeps a table
  of "in this situation, how good is each move?" and updates it after every step.
  Rewards: −1 per step, −10 for dying, +10 for the goal.
- **The experience log.** After 5 deaths, Agent A's experience is written in plain words,
  e.g. *"[E1] DIED after stepping on glowing hot coals — 3 times"*. No positions.
- **The LLM reflection.** The LLM reads the log once and writes rules such as
  *"IF the tile is hot or glowing THEN don't move forward onto it"*, citing the log
  lines it used.
- **Shields.** Before the new agent steps forward, a shield decides "is that tile
  deadly?" and if so, forbids the step. We compare six shields:

| Shield | How it decides |
|---|---|
| S0 none | never blocks |
| S1 lookup | "did this exact tile kill me before?" |
| S2 embedding | "does its name look like a killer's name?" (the old ChromaDB method) |
| S3 LLM prior | asks the LLM with no experience — common sense only |
| S4 LLM + experience | asks the LLM with the rules and log (**our method**) |
| S6 oracle | cheats: knows the truth. Shows the best possible result |

- **The Lab** (`streamlit run lab.py`): knobs, a train button, episode replays with the
  shield's reason written under each frame, comparisons and sweeps.

## 4. How did we test it fairly?

| What we did | Why |
|---|---|
| Test tiles were never seen in training | Otherwise a lookup table could pass the test |
| Decoys that share colours and words with hazards | Otherwise "block everything red" would look perfect |
| 10 seeds × 100 episodes for every result | One lucky run proves nothing |
| 95% confidence intervals | Shows how sure we are about each number |
| Same layouts and same random draws for every shield | Differences come from the shield, not from luck |
| Separate seeds for training and testing | The test rooms are truly new |
| Every number has the command that made it | Anyone can check it |

## 5. What happened?

**The learner learned.** After 6 000 training episodes it reaches the goal in 90.7% of
training rooms and dies in only 0.4%. It learned to avoid red lava and hot coals.

**But it walks into new hot tiles.** In test rooms with never-seen hot tiles it dies in
41.1% of episodes.

| Shield | Success | Deaths | Decoys wrongly avoided |
|---|---|---|---|
| S0 none | 57.4% | 41.1% | 0% |
| S1 lookup | 57.4% | 41.1% | 0% |
| S6 oracle | 92.6% | 0.0% | 0% |
| S2 embedding | *not run yet* | | |
| S3 LLM prior | *not run yet* | | |
| S4 LLM + experience | *not run yet* | | |

| What we expected | What happened | Why |
|---|---|---|
| L1-H1: lookup won't help on new tiles | **Confirmed.** Lookup = no shield, to the decimal | It only knows tiles that killed the learner, and none of those appear in tests |
| A good shield would help a lot | **Confirmed.** The oracle removes every death and adds 35 points of success | So the size of the prize for S3/S4 is known: somewhere between 57% and 93% success |
| L1-H2 to H4 (about the LLM and embeddings) | **Not measured yet** | Needs the Claude API key in a new session (then one command each) |

**Harder rooms:** with more hot tiles, deaths without a shield climb from 16% (1 hot tile)
to 75% (6 hot tiles); the oracle stays at 0% throughout.

## 6. What surprised us / what went wrong

- **The agent got stuck turning left-right-left-right.** With only "what's in front of
  me" it couldn't tell it had just turned, so ~40% of episodes timed out. Giving it a
  one-step memory (its last action) fixed it. Adding randomness also broke the loops but
  made it walk into lava, so we didn't use it.
- **Lookup knew only 2 of 3 training hazards.** With a budget of 5 deaths, the learner
  happened to die on lava and coals but never on a campfire. Even "seen" knowledge is
  partial — a realistic situation for the LLM to deal with.
- **A blocked agent can still get stuck.** With the oracle, 7.4% of episodes time out:
  when the way forward is blocked, the simple learner sometimes circles.
- **Network limits.** This cloud session cannot reach Groq, OpenRouter or Hugging Face.
  So we use Claude Haiku 5.5 (reachable, cheapest Claude model) for the LLM runs, and the
  embedding shield (needs Hugging Face) must run on a laptop or Colab.
- **The fake keyword LLM** (used only to test the plumbing) blocked the "painting of a
  fire" decoy. That shows why decoys matter: matching words is not understanding.

## 7. New words we learned

| Word | Meaning | Example here |
|---|---|---|
| Q-learning | Learning a table of "how good is this move here?" from rewards | Agent A |
| Held-out tile | Kept out of training, used only for testing | molten rock |
| Decoy | Harmless tile that looks dangerous | red carpet |
| Shield | Filter that forbids unsafe moves | blocks "move forward" into lava |
| Oracle | A cheating method that shows the best possible score | S6 |
| Paired comparison | Every method gets the exact same test rooms and luck | all shields use the same seeds |
| Confidence interval | Range where the true value probably is | 41.1% [39.0%, 43.6%] |
| Sparse reflection | Calling the LLM rarely (once, after 5 deaths) | one reflection call per run |

## 8. What this means for the next level

The setup works and the gap is real: 41% deaths that only a shield with good
judgement can remove. Two things are needed to finish Level 1: run S3/S4 with an LLM
and S2 with embeddings. Level 2 then asks the harder question: what if the world
breaks common sense (water kills, fire is safe)? There, an LLM without experience
(S3) should fail, and only S4 can adapt.
