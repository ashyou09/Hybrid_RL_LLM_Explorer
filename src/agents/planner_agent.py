"""
planner_agent.py — Online DFS Explorer (Real‑Time Pathfinding)
===============================================================
This agent only sees ONE block ahead—exactly like a human.

How it works:
  1. Stand on a tile → paint it GREEN on the GUI (breadcrumb trail).
  2. Look at the tile directly in front.
     - Wall or already‑visited?  → Turn right, try another direction.
     - Vector DB says "danger"?  → Mark as visited, turn right.
     - Safe & unvisited?         → Step forward.
  3. If all 4 directions are blocked → backtrack along the green trail
     to the previous intersection and try a new branch.

New additions (Adaptive LLM Consultation + Revisit Penalty):
  - Revisit penalty:  After the first 10 steps, re-entering a cell incurs
    a growing internal penalty (−1 × revisit_count). Logged to console.
  - Trajectory buffer: Rolling log of the last 20 steps (pos, front label,
    action, revisit flag) sent periodically to the LLM for navigation hints.
  - Periodic LLM consult: Every 10–15 steps (randomised), if ≥10 unique
    tiles have been visited, the trajectory is sent to llama3.2:3b (via
    reflection_engine.get_navigation_hint). The returned hint is stored in
    MemoryHub as a soft navigation rule.
"""

import random
from collections import defaultdict

from minigrid.core.world_object import Floor
from src.core.rl_core import parse_local_observation


# ─── Trajectory entry type ───────────────────────────────────
_TRAJ_MAX = 30          # how many recent steps to keep
_REVISIT_GRACE = 10     # steps before revisit penalty kicks in
_LLM_INTERVAL_MIN = 10  # consult no sooner than every N steps
_LLM_INTERVAL_MAX = 15  # consult no later than every M steps
_MAX_DEATHS_TRACKED = 3 # store paths for first 3 deaths


class OnlineExplorerAgent:
    """Real‑time depth‑first maze explorer with LLM rule masking,
    revisit penalties, and periodic LLM navigation consultation.
    
    Research feature: Multi-death episodic learning via LLM analysis.
    Collects paths from up to 3 deaths to build strategic meta-rules."""

    def __init__(self, memory_hub=None, env_name="unknown"):
        self.memory         = memory_hub    # ChromaDB Vector Store
        self.env_name       = env_name

        # ── Core DFS state ──────────────────
        self.visited        = set()         # (x, y) already explored
        self.path_stack     = []            # positions for backtracking
        self.backtracking   = False
        self.turn_count     = 0

        # ── Revisit tracking ────────────────
        self.revisit_counts = defaultdict(int)   # pos → total entries
        self._step_count    = 0                  # steps taken this episode
        self._episode_penalty = 0.0              # accumulated revisit penalty
        self._last_pos      = None               # detect movement
        self.force_stop     = False              # signal to end episode

        # ── Trajectory buffer ───────────────
        self.trajectory_log: list[dict] = []     # rolling last-N steps

        # ── LLM consultation schedule ───────
        self._next_llm_at   = random.randint(_LLM_INTERVAL_MIN, _LLM_INTERVAL_MAX)
        self._llm_calls     = 0                  # how many hints requested
        
        # ── Multi-death episodic learning ───
        self.episode_deaths         = 0          # deaths so far this episode
        self.death_paths: list[dict] = []        # store up to _MAX_DEATHS_TRACKED
        self.current_episode_path   = []         # track path for current death

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _record_step(self, pos, front_label, action_name, revisit_hit):
        """Append an entry to the rolling trajectory buffer."""
        entry = {
            "step": self._step_count,
            "pos": pos,
            "front": front_label,
            "action": action_name,
            "revisit": revisit_hit,
        }
        self.trajectory_log.append(entry)
        if len(self.trajectory_log) > _TRAJ_MAX:
            self.trajectory_log.pop(0)

    def _apply_revisit_penalty(self, pos):
        """After the grace period, penalise re-entering a known cell.
        Returns the penalty value (≤0) and logs it to console."""
        
        # Do not apply penalty if the agent is performing a calculated backtrack out of a dead end.
        # ALSO, do not apply penalty if the agent just rotated in place (pos == last_pos)
        if self.backtracking or pos == self._last_pos:
            return 0.0, False

        # Penalize re-entering in a visited cell
        if self._step_count > _REVISIT_GRACE and pos in self.visited:
            self.revisit_counts[pos] += 1
            count = self.revisit_counts[pos]    
            penalty = -1.0 * count  # Penalty = -1 per revisit (cumulative)
            self._episode_penalty += penalty
            
            # Reduce log noise: only print every 5 revisits or on first revisit
            if count <= 1 or count % 5 == 0 or self._episode_penalty < -500:
                print(
                    f"\n  [⚠ Revisit #{count}] pos={pos} | "
                    f"penalty={penalty:.1f}, cumulative={self._episode_penalty:.1f}"
                )
            
            if count > 100:
                print(f"\n  [🛑 GAME OVER] Tile {pos} visited over 100 times. Stopping stuck agent.")
                self.force_stop = True
                
            if self._episode_penalty <= -500:
                print(f"\n  [🛑 CRITICAL] Revisit penalty exceeded -500. Force-stopping game.")
                self.force_stop = True
                
            return penalty, True
        return 0.0, False

    def _should_consult_llm(self):
        """True when the step counter hits the scheduled threshold AND
        we have seen at least 10 unique tiles (enough context for the LLM)."""
        return (
            self._step_count >= self._next_llm_at
            and len(self.visited) >= 10
        )

    def record_death(self, failure_pos):
        """Called when agent dies. Stores path data for multi-death analysis.
        Tracks up to _MAX_DEATHS_TRACKED deaths."""
        self.episode_deaths += 1
        
        death_record = {
            "death_number": self.episode_deaths,
            "step_count": self._step_count,
            "unique_tiles": len(self.visited),
            "revisit_penalty": self._episode_penalty,
            "failure_location": failure_pos,
            "revisit_counts": dict(self.revisit_counts),
            "path": self.trajectory_log.copy(),
        }
        
        if len(self.death_paths) < _MAX_DEATHS_TRACKED:
            self.death_paths.append(death_record)
            print(
                f"\n  [📊 Death Recording] Death #{self.episode_deaths} recorded. "
                f"(Stored {len(self.death_paths)}/{_MAX_DEATHS_TRACKED})"
            )
        
        # Reset for next attempt
        self.visited.clear()
        self.revisit_counts.clear()
        self.trajectory_log.clear()
        self._step_count = 0
        self._episode_penalty = 0.0
        self.turn_count = 0
        self.path_stack = []
        self.backtracking = False

    def _consult_llm(self):
        """Build a trajectory payload and request a navigation hint from
        the LLM. If a hint is returned, store it in MemoryHub."""
        self._llm_calls += 1
        # Schedule the next consultation
        self._next_llm_at = (
            self._step_count
            + random.randint(_LLM_INTERVAL_MIN, _LLM_INTERVAL_MAX)
        )

        print(
            f"\n  [🧠 LLM Consult #{self._llm_calls}] "
            f"step={self._step_count}, unique_cells={len(self.visited)}, "
            f"penalty_so_far={self._episode_penalty:.1f}"
        )

        # Build summary string (last N trajectory entries)
        traj_lines = [
            f"  step {e['step']}: pos={e['pos']} front='{e['front']}' "
            f"action={e['action']} revisit={e['revisit']}"
            for e in self.trajectory_log
        ]
        top_revisited = sorted(
            self.revisit_counts.items(), key=lambda kv: kv[1], reverse=True
        )[:3]

        payload = {
            "environment": self.env_name,
            "total_steps": self._step_count,
            "unique_tiles_visited": len(self.visited),
            "accumulated_revisit_penalty": self._episode_penalty,
            "top_revisited_cells": top_revisited,
            "trajectory": traj_lines,
        }

        try:
            from src.agents.reflection_engine import get_navigation_hint
            hint = get_navigation_hint(payload)
        except Exception as exc:
            print(f"  [LLM Consult] Error calling reflection engine: {exc}")
            hint = None

        if hint and self.memory:
            print(f"  [LLM Hint] Rule received: {hint.get('rule', '?')}")
            self.memory.store_verified_rule(hint)
        elif hint is None:
            print("  [LLM Consult] No actionable hint returned.")

    def analyze_multiple_deaths(self):
        """Called when episode_deaths >= 3. Sends all collected death paths
        to LLM for meta-analysis and receives strategic guidance."""
        if len(self.death_paths) < 2:
            print("  [Multi-Death Analysis] Not enough deaths recorded (need ≥2).")
            return None
        
        print(
            f"\n  ╔════════════════════════════════════════════════╗"
            f"\n  ║  MULTI-DEATH EPISODIC ANALYSIS (Deaths: {len(self.death_paths)})  ║"
            f"\n  ╚════════════════════════════════════════════════╝"
        )
        
        # Build comprehensive death summary for LLM
        death_summaries = []
        for i, death_record in enumerate(self.death_paths, 1):
            top_rev_cells = sorted(
                death_record["revisit_counts"].items(),
                key=lambda kv: kv[1],
                reverse=True
            )[:2]
            
            summary = (
                f"Death #{i}: "
                f"step={death_record['step_count']}, "
                f"unique={death_record['unique_tiles']}, "
                f"penalty={death_record['revisit_penalty']:.1f}, "
                f"failure_pos={death_record['failure_location']}, "
                f"top_revisits={top_rev_cells}"
            )
            death_summaries.append(summary)
            print(f"    {summary}")
        
        payload = {
            "environment": self.env_name,
            "analysis_type": "multi_death_episodic",
            "death_count": len(self.death_paths),
            "death_summaries": death_summaries,
            "all_death_records": self.death_paths,
        }
        
        try:
            from src.agents.reflection_engine import analyze_multi_death_pattern
            strategic_hint = analyze_multi_death_pattern(payload)
        except Exception as exc:
            print(f"  [Multi-Death] Error: {exc}")
            strategic_hint = None
        
        if strategic_hint and self.memory:
            print(f"\n  [🧠 Strategic Hint] {strategic_hint.get('rule', '?')}")
            self.memory.store_verified_rule(strategic_hint)
            return strategic_hint
        
        return None

    # ─────────────────────────────────────────────
    # Main decision loop (called every tick)
    # ─────────────────────────────────────────────

    def act(self, env, obs):
        """Returns an action: 0=Turn Left, 1=Turn Right, 2=Move Forward."""
        self._step_count += 1
        pos     = tuple(int(c) for c in env.unwrapped.agent_pos)
        dir_idx = env.unwrapped.agent_dir

        # ── Track revisits & paint breadcrumb ──────────────────
        env.unwrapped.grid.set(pos[0], pos[1], Floor("green"))

        penalty, revisit_hit = self._apply_revisit_penalty(pos)
        self.visited.add(pos)
        self._last_pos = pos

        if not self.path_stack:
            self.path_stack.append(pos)

        # Direction vector the agent is currently facing
        dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][dir_idx]
        front  = (pos[0] + dx, pos[1] + dy)

        # ── Periodic LLM consultation ───────────────────────────
        if self._should_consult_llm():
            self._consult_llm()

        # ═══════════════════════════════════════
        # A. BACKTRACKING MODE
        # ═══════════════════════════════════════
        if self.backtracking:
            if len(self.path_stack) <= 1:
                self._record_step(pos, "trapped", "TurnLeft", revisit_hit)
                return 0

            target = self.path_stack[-2]
            tdx = target[0] - pos[0]
            tdy = target[1] - pos[1]

            if (tdx, tdy) == (dx, dy):
                print(f"\n  [←] Backtracking to {target}")
                self.backtracking = False
                self.turn_count   = 0
                self.path_stack.pop()
                self._record_step(pos, "backtrack", "MoveForward", revisit_hit)
                return 2  # Move Forward

            action = 0 if (tdx, tdy) == (dy, -dx) else 1
            self._record_step(pos, "backtrack", "TurnLeft" if action == 0 else "TurnRight", revisit_hit)
            return action

        # ═══════════════════════════════════════
        # B. EXPLORATION MODE
        # ═══════════════════════════════════════
        local_text  = parse_local_observation(obs["image"])
        front_block = local_text.split(".")[0].replace("Front: ", "")

        # B1. Check LLM rules for danger first
        is_front_dangerous = False
        if self.memory:
            rule = self.memory.query_local_context(
                front_block, threshold=0.75, silent=True  # Ensure high threshold so "empty space" doesn't match "revisited tile"
            )
            if rule and rule.get("forbidden_action") == "Move Forward":
                print(
                    f"\n  [\033[91m🧠 LLM KNOWLEDGE\033[0m] {front_block} ahead "
                    f"— Applying HEAVY PENALTY based on rule: {rule['rule']}"
                )
                is_front_dangerous = True

        # B2. Evaluate Front vs All Options if we are stuck or facing blocked paths
        if front_block == "wall" or front in self.visited or is_front_dangerous:
            self.turn_count += 1
            
            # Find the best valid visited tile as a fallback, we prefer paths with LOWEST revisit_counts
            if front in self.visited and not is_front_dangerous:
                best_vis_pos = front
                min_count = self.revisit_counts.get(front, float('inf'))
                
                # Check surrounding visited tiles (only if they aren't walls)
                for i in range(4):
                    adx, ady = [(1, 0), (0, 1), (-1, 0), (0, -1)][i]
                    adj = (pos[0] + adx, pos[1] + ady)
                    # We only consider adjacent cells that are visited (so we know they aren't walls)
                    if adj in self.visited:
                        cnt = self.revisit_counts.get(adj, 0)
                        if cnt < min_count:
                            min_count = cnt
                            best_vis_pos = adj
                
                # If there is a BETTER visited tile around us, we should turn to face it!
                if best_vis_pos != front:
                    self._record_step(pos, "seeking less penality", "TurnRight", revisit_hit)
                    return 1 # Keep turning until we face the best one
            
            if self.turn_count >= 4:
                # If we've looked everywhere and they are all visited, blocked, or dangerous:
                if len(self.path_stack) > 1:
                    print(f"\n  [↺] Dead end/trapped at {pos} — backtracking")
                    self.backtracking = True
                else:
                    # Nowhere to backtrack! Just move into the least penalized visited tile if we can.
                    # Or keep turning.
                    print(f"\n  [↺] Completely trapped at {pos}")
                    
            if is_front_dangerous:
                self.visited.add(front) # Mark dangerous tile as visited so we don't try it again
                self._record_step(pos, front_block, "TurnRight(danger)", revisit_hit)
            else:
                self._record_step(pos, front_block, "TurnRight", revisit_hit)
            
            return 1

        # B3. Safe & unvisited → move forward
        print(
            f"\r  [\033[92m✓\033[0m] Safe — stepping into {front}        ",
            end="", flush=True,
        )
        self.path_stack.append(front)
        self.turn_count = 0
        self._record_step(pos, front_block, "MoveForward", revisit_hit)
        return 2  # Move Forward

    # ─────────────────────────────────────────────
    # Episode summary
    # ─────────────────────────────────────────────

    def episode_summary(self):
        """Print a summary of revisit penalties and LLM consultations."""
        print(
            f"\n  ── Episode Stats ──────────────────────────────────\n"
            f"     Steps taken    : {self._step_count}\n"
            f"     Unique tiles   : {len(self.visited)}\n"
            f"     Revisit penalty: {self._episode_penalty:.1f}\n"
            f"     LLM consults   : {self._llm_calls}\n"
            f"     Deaths tracked : {self.episode_deaths}/{_MAX_DEATHS_TRACKED}\n"
            f"  ───────────────────────────────────────────────────"
        )
