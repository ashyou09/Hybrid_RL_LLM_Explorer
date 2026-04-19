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

import time
import gymnasium as gym
import src.env.environments as environments                          # registers custom rooms
from src.core.rl_core           import DQNAgent, preprocess_obs, parse_local_observation
from src.agents.reflection_engine import analyze_failure_log, verify_rule
from src.core.memory_hub        import MemoryHub
from src.agents.planner_agent     import OnlineExplorerAgent
from src.ui.display           import UnifiedDisplay
from dataclasses import dataclass

display = None   # global — set in __main__

@dataclass(frozen=True)
class ExperimentTiming:
    step_delay_secs: float = 0.18          # visual speed per env step
    learning_phase_secs: int = 3600        # basically infinite time
    final_exam_secs: int = 3600            # basically infinite time
    final_exam_max_episodes: int = 1       # only need one successful run now

TIMING = ExperimentTiming()


def _require_display():
    if display is None:
        raise RuntimeError("Display is not initialized.")


# ──────────────────────────────────────────────────────────
#  Phase 1 & 2: RL exploration → LLM rule → Explorer validation
#  Each phase always runs for LEARNING_PHASE_SECS wall-clock seconds.
# ──────────────────────────────────────────────────────────

def run_learning_phase(env_name, agent, memory):
    _require_display()
    label = env_name.split("-")[1]
    display.set_phase(f"RL Exploration — {label}")

    print(f"\n{'─'*50}")
    print(f"[Agent A] Entering {env_name}  ({TIMING.learning_phase_secs}s)")
    print(f"{'─'*50}")

    deadline = time.time() + TIMING.learning_phase_secs
    deaths = 0
    rule_data = None
    trigger   = None

    # ── SUB-PHASE A: RL exploration until 2 deaths ──────────
    env = gym.make(env_name, render_mode="rgb_array")
    episode = 0

    while deaths < 2 and time.time() < deadline:
        episode += 1
        obs = env.reset()[0]
        
        from collections import defaultdict
        revisit_counts = defaultdict(int)
        
        last_pos = None
        grid_size = env.unwrapped.grid.width
        display.render_frame(env.render(), overlay_visits=revisit_counts, grid_size=grid_size)

        for step in range(100):
            if time.time() >= deadline:
                break
            state  = preprocess_obs(obs)
            
            # 🧠 LLM Knowledge Injection (Action Masking)
            mask = [1.0, 1.0, 1.0]
            local_text = parse_local_observation(obs["image"])
            front_block = local_text.split(".")[0].replace("Front: ", "")
            
            if memory:
                # Lowered threshold to 0.60 to be more aggressive with safety rules
                rule = memory.query_local_context(front_block, threshold=0.60, silent=True)
                if rule and rule.get("forbidden_action") == "Move Forward":
                    print(f"\n  [🧠 LLM KNOWLEDGE] !!! DANGER: {front_block} !!! — applying HEAVY PENALTY.")
                    mask = [1.0, 1.0, 0.0]  # Mask 'Move Forward'

            action = agent.select_action(state, apply_mask=mask)
            display.wait(TIMING.step_delay_secs)

            pos = tuple(env.unwrapped.agent_pos)
            if pos != last_pos: # Only append penalty if we successfully moved!
                if pos in revisit_counts:
                    revisit_counts[pos] += 1
                    count = revisit_counts[pos]
                    print(f"  [⚠ Revisit] {pos} x{count} | penalty=-3.0")
                    if count > 100:
                        print(f"  [🛑 GAME OVER] hallucinate")
                        break
                else:
                    revisit_counts[pos] = 1
            last_pos = pos

            next_obs, reward, terminated, truncated, _ = env.step(action)
            display.render_frame(env.render(), overlay_visits=revisit_counts, grid_size=grid_size)

            if reward <= -10:
                deaths += 1
                print(f"  [💀] Death #{deaths} at step {step}")

                if deaths >= 2:
                    print(f"  [Agent A] {deaths} deaths collected. Sending to LLM…")
                    agent.trigger_failure_log(env_name, obs["image"], action)

                    display.set_phase(f"LLM Reflection — {label}")
                    from src.agents.reflection_engine import analyze_failure_log, verify_rule
                    rule_data = analyze_failure_log()

                    if rule_data and verify_rule(rule_data):
                        memory.store_verified_rule(rule_data)

                    if rule_data:
                        print(f"  [Agent A] Rule learned: {rule_data['rule']}")
                        trigger = rule_data["trigger_feature"]
                    break
                else:
                    print("  [Agent A] Respawning to confirm…")
                    break

            obs = next_obs
            if terminated or truncated:
                break

    env.close()

    # ── SUB-PHASE B: Truth Confirmation (loop until deadline) ──
    if rule_data and time.time() < deadline:
        print(f"\n  ╔══════════════════════════════════════════════╗")
        print(f"  ║  TRUTH CONFIRMATION: Re-entering same room    ║")
        print(f"  ║  Verifying the LLM rule prevents real deaths  ║")
        print(f"  ╚══════════════════════════════════════════════╝")

        display.set_phase(f"Truth Confirmation — {label}")
        ep = 0

        while time.time() < deadline and ep < 1:
            ep += 1
            env = gym.make(env_name, render_mode="rgb_array")
            obs = env.reset()[0]
            grid_size = env.unwrapped.grid.width
            explorer = OnlineExplorerAgent(memory, env_name=env_name)
            display.render_frame(env.render(), overlay_visits=explorer.revisit_counts, grid_size=grid_size)
            print(f"  [Truth] Episode {ep} starting…")

            while time.time() < deadline:
                display.wait(TIMING.step_delay_secs)
                action = explorer.act(env, obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                display.render_frame(env.render(), overlay_visits=explorer.revisit_counts, grid_size=grid_size)

                if explorer.force_stop:
                    print("  [🛑] Revisit penalty limit reached — ending episode.")
                    break

                if reward >= 10:
                    print(f"\n  [\033[92m✓ TRUTH CONFIRMED\033[0m] Rule works — no deaths!")
                    break
                if reward <= -10:
                    print(f"\n  [\033[91m✗ TRUTH REFUTED\033[0m] Rule failed — agent died!")
                    break
                if terminated or truncated:
                    break

            explorer.episode_summary()
            env.close()

    # If we haven't learned a rule yet but ran out of time, just
    # keep exploring to fill the remaining seconds visually
    elif not rule_data and time.time() < deadline:
        env = gym.make(env_name, render_mode="rgb_array")
        obs = env.reset()[0]
        display.render_frame(env.render())
        print("  [Agent A] Continuing exploration…")
        while time.time() < deadline:
            state  = preprocess_obs(obs)
            action = agent.select_action(state)
            display.wait(TIMING.step_delay_secs)
            obs, reward, terminated, truncated, _ = env.step(action)
            display.render_frame(env.render())
            if terminated or truncated:
                obs = env.reset()[0]
                display.render_frame(env.render())
        env.close()

    elapsed = TIMING.learning_phase_secs - max(0, deadline - time.time())
    print(f"  [{label}] Phase complete ({elapsed:.0f}s elapsed)")
    return trigger



# ──────────────────────────────────────────────────────────
#  Phase 3: Final Exam
# ──────────────────────────────────────────────────────────

def run_final_exam(env_name, memory):
    """Run Phase 3 for up to TIMING.final_exam_secs wall-clock seconds.
    
    Research Feature: Multi-death episodic analysis.
    If explorer dies 3+ times, collect all death paths and send to LLM
    for strategic pattern analysis."""
    _require_display()
    display.set_phase("PHASE 3 — FINAL EXAM")

    print(f"\n{'═'*50}")
    print(f"  PHASE 3 — FINAL EXAM: {env_name}")
    print(f"  Running for up to {TIMING.final_exam_secs} seconds (max {TIMING.final_exam_max_episodes} episodes)…")
    print(f"{'═'*50}")

    deadline = time.time() + TIMING.final_exam_secs
    episode  = 0

    # User requested to keep running until it wins, so we drop the max episode limit
    while True:
        episode += 1
        env      = gym.make(env_name, render_mode="rgb_array")
        explorer = OnlineExplorerAgent(memory, env_name=env_name)
        obs      = env.reset()[0]
        grid_size = env.unwrapped.grid.width
        display.render_frame(env.render(), overlay_visits=explorer.revisit_counts, grid_size=grid_size)
        print(f"\n  [Phase 3] Episode {episode} starting…")

        step = 0
        while True:
            display.wait(TIMING.step_delay_secs)
            action = explorer.act(env, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            display.render_frame(env.render(), overlay_visits=explorer.revisit_counts, grid_size=grid_size)
            step += 1

            if explorer.force_stop:
                print("  [🛑] Revisit penalty limit reached — ending episode.")
                break

            if reward <= -10:
                # Agent died — record the death for multi-death analysis
                failure_pos = tuple(env.unwrapped.agent_pos)
                explorer.record_death(failure_pos)
                print(f"  [💀] Episode {episode}: FAILED — stepped into hazard at {failure_pos} after {step} steps.")
                
                # Check if we've accumulated 3 deaths for meta-analysis
                if explorer.episode_deaths >= 3:
                    print("\n  [📊] Three deaths recorded! Requesting LLM meta-analysis…")
                    explorer.analyze_multiple_deaths()
                
                break

            if reward >= 10:
                print(f"  [\033[92m✓ FLAWLESS SUCCESS\033[0m] Episode {episode} goal reached in {step} steps!")
                explorer.episode_summary()
                env.close()
                print(f"\n  [Phase 3] Game WON! Showcase complete.")
                return

            if terminated or truncated:
                break

        explorer.episode_summary()
        env.close()


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
    display.wait(TIMING.step_delay_secs)

    print("[Done] Next run will start fresh automatically.\n")

    display.cleanup()
