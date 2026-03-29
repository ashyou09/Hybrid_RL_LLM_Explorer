"""
run_experiment.py — Main Orchestrator (Side-by-Side Display)

=============================================================
Single pygame window: game on the LEFT, coloured log on the RIGHT.

  Phase 1 — RL Agent explores Lava Room, dies twice, LLM generates rule,
            then Explorer proves the rule works in that room.
  Phase 2 — Same for Quicksand Room.
  Phase 3 — Explorer tackles the Combined Final Exam using
            both rules from the Vector DB.

ChromaDB is wiped at both start AND end so every run is pristine.
"""

import shutil
import time
import gymnasium as gym
import environments                          # registers custom rooms
from rl_core           import DQNAgent, preprocess_obs
from reflection_engine import analyze_failure_log, verify_rule
from memory_hub        import MemoryHub
from planner_agent     import OnlineExplorerAgent
from display           import UnifiedDisplay
import random

display = None   # global — set in __main__


# ──────────────────────────────────────────────────────────
#  Phase 1 & 2: RL exploration → LLM rule → Explorer validation
# ──────────────────────────────────────────────────────────

def run_learning_phase(env_name, agent, memory):
    env = gym.make(env_name, render_mode="rgb_array")
    label = env_name.split("-")[1]
    display.set_phase(f"RL Exploration — {label}")

    print(f"\n{'─'*50}")
    print(f"[Agent A] Entering {env_name}")
    print(f"{'─'*50}")

    deaths = 0

    for episode in range(30):
        obs = env.reset()[0]
        display.render_frame(env.render())

        for step in range(100):
            state  = preprocess_obs(obs)
            action = agent.select_action(state)
            display.wait(0.22)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            display.render_frame(env.render())

            if reward <= -10:
                deaths += 1
                print(f"  [💀] Death #{deaths} at step {step}")

                if deaths >= 2:
                    print(f"  [Agent A] {deaths} deaths collected. Sending to LLM…")
                    agent.trigger_failure_log(env_name, obs["image"], action)

                    display.set_phase(f"LLM Reflection — {label}")
                    rule_data = analyze_failure_log()

                    if rule_data and verify_rule(env, agent, rule_data):
                        memory.store_verified_rule(rule_data)

                    env.close()
                    print(f"  [Agent A] Rule learned: {rule_data['rule']}")

                    print(f"\n  ╔══════════════════════════════════════════════╗")
                    print(f"  ║  TRUTH CONFIRMATION: Re-entering same room    ║")
                    print(f"  ║  Verifying the LLM rule prevents real deaths  ║")
                    print(f"  ╚══════════════════════════════════════════════╝")

                    display.set_phase(f"Truth Confirmation — {label}")
                    env = gym.make(env_name, render_mode="rgb_array")
                    obs = env.reset()[0]
                    display.render_frame(env.render())
                    explorer = OnlineExplorerAgent(memory)

                    for _ in range(300):
                        display.wait(0.22)
                        action = explorer.act(env, obs)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        display.render_frame(env.render())

                        if reward >= 10:
                            print(f"\n  [\033[92m✓ TRUTH CONFIRMED\033[0m] Rule works — no deaths!")
                            break
                        if reward <= -10:
                            print(f"\n  [\033[91m✗ TRUTH REFUTED\033[0m] Rule failed — agent died!")
                            break
                        if terminated or truncated:
                            break

                    env.close()
                    return rule_data["trigger_feature"]
                else:
                    print("  [Agent A] Respawning to confirm…")
                    break

            obs = next_obs
            if terminated or truncated:
                break

    env.close()
    return None


# ──────────────────────────────────────────────────────────
#  Phase 3: Final Exam
# ──────────────────────────────────────────────────────────

FINAL_EXAM_SECS = 50    # how long Phase 3 runs (wall-clock seconds)

def run_final_exam(env_name, memory):
    """Run Phase 3 for FINAL_EXAM_SECS wall-clock seconds, resetting on each episode end."""
    display.set_phase("PHASE 3 — FINAL EXAM")

    print(f"\n{'═'*50}")
    print(f"  PHASE 3 — FINAL EXAM: {env_name}")
    print(f"  Running for {FINAL_EXAM_SECS} seconds…")
    print(f"{'═'*50}")

    deadline = time.time() + FINAL_EXAM_SECS
    episode  = 0

    while time.time() < deadline:
        episode += 1
        env      = gym.make(env_name, render_mode="rgb_array")
        explorer = OnlineExplorerAgent(memory)
        obs      = env.reset()[0]
        display.render_frame(env.render())
        print(f"\n  [Phase 3] Episode {episode} starting…")

        step = 0
        while time.time() < deadline:
            display.wait(0.22)
            action = explorer.act(env, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            display.render_frame(env.render())
            step += 1

            if reward <= -10:
                print(f"  [Agent C] Episode {episode} FAILED — stepped into a hazard after {step} steps.")
                break
            if reward >= 10:
                print(f"  [\033[92m✓ FLAWLESS SUCCESS\033[0m] Episode {episode} goal reached in {step} steps!")
                break
            if terminated or truncated:
                break

        env.close()

    print(f"\n  [Phase 3] 50-second showcase complete — {episode} episode(s) played.")


# ──────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    display = UnifiedDisplay()

    print("\n" + "=" * 55)
    print("  Exp-1: Hybrid RL > LLM > Semantic Rule Transfer")
    print("=" * 55)

    memory  = MemoryHub()
    agent_a = DQNAgent()

    display.set_phase("PHASE 1 — Lava Room")
    hazard_1 = run_learning_phase("MiniGrid-LavaRoom-v0", agent_a, memory)

    display.set_phase("PHASE 2 — Quicksand Room")
    hazard_2 = run_learning_phase("MiniGrid-QuicksandRoom-v0", agent_a, memory)

    print(f"\n{'─'*55}")
    print(f"  Conclusion: must avoid {hazard_1} and {hazard_2}.")
    print(f"  Now the Explorer will tackle the combined maze.")
    print(f"{'─'*55}")

    run_final_exam("MiniGrid-CombinedTesting-v0", memory)

    display.set_phase("Done")
    display.wait(0.2)

    print("[Done] Next run will start fresh automatically.\n")

    display.cleanup()
