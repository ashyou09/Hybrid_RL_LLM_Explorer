"""
planner_agent.py — Phase 3 Smart Explorer
==========================================
Used in Phase 3 only. The agent explores the Lava Room using:
  1. MemoryHub LLM rules (from Phase 1 learning) — if 'red lava' is
     directly ahead the rule fires and the agent TURNS AWAY instead of dying.
  2. Pure RL Exploration (TilePenaltyTracker) — it greedily seeks the maximum penalty
     (closest to 0) to avoid already-visited tiles, exactly like Phase 1.

Lava tiles are at NEW random positions every episode so the agent
cannot rely on position memory — it must use the tile-type rule.
"""

import random
from src.core.rl_core import parse_local_observation, TilePenaltyTracker


class OnlineExplorerAgent:
    """Smart Explorer that navigates using greedy RL (penalty avoidance)
    BUT stops to query the MemoryHub (LLM rules) before stepping forward.
    If a rule marks a tile as dangerous, it applies a massive artificial penalty
    to that direction, forcing the RL logic to turn away."""

    def __init__(self, memory_hub=None, env_name="unknown"):
        self.memory         = memory_hub    # ChromaDB Vector Store
        self.env_name       = env_name
        
        self.tracker        = TilePenaltyTracker()
        self.revisit_counts = self.tracker.tile_penalty # Expose for UI display overlay
        
        self._step_count    = 0
        self._last_pos      = None
        self._stay_count    = 0
        self.force_stop     = False

    def episode_summary(self):
        """Print a summary of episode stats."""
        print(
            f"\n  ── Episode Stats ──────────────────────────────────\n"
            f"     Steps taken    : {self.tracker.total_steps}\n"
            f"     Total penalty  : {self.tracker.total_penalty:.1f}\n"
            f"  ───────────────────────────────────────────────────"
        )

    # ─────────────────────────────────────────────
    # Main decision loop (called every tick)
    # ─────────────────────────────────────────────

    def act(self, env, obs):
        """Returns an action: 0=Turn Left, 1=Turn Right, 2=Move Forward."""
        self._step_count += 1
        pos    = tuple(int(c) for c in env.unwrapped.agent_pos)
        dir_   = env.unwrapped.agent_dir

        # Record step penalty
        if pos == self._last_pos:
            self._stay_count += 1
        else:
            self._stay_count = 0
            self._last_pos = pos

        mult = 1.0 + (self._stay_count // 3) * 2.0
        cost, auto_kill = self.tracker.record_step(pos)

        if mult > 1.0:
            extra = (mult - 1.0) * self.tracker.STEP_COST
            self.tracker.tile_penalty[pos] += extra
            self.tracker.total_penalty += extra

        if auto_kill:
            print(f"\n  [🛑 GAME OVER] Tile {pos} accumulated 60 penalty. Stopping stuck agent.")
            self.force_stop = True

        # Direction vectors for dir 0-3: Right, Down, Left, Up (MiniGrid convention)
        DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dx, dy = DIRS[dir_]
        front  = (pos[0] + dx, pos[1] + dy)

        left_dir   = (dir_ - 1) % 4
        right_dir  = (dir_ + 1) % 4
        behind_dir = (dir_ + 2) % 4
        
        ldx, ldy = DIRS[left_dir]
        rdx, rdy = DIRS[right_dir]
        bdx, bdy = DIRS[behind_dir]
        
        left_pos   = (pos[0] + ldx, pos[1] + ldy)
        right_pos  = (pos[0] + rdx, pos[1] + rdy)
        behind_pos = (pos[0] + bdx, pos[1] + bdy)

        _mem_cache = {}
        def get_pen(p):
            w = env.unwrapped.grid.width
            h = env.unwrapped.grid.height
            if p[0] < 0 or p[0] >= w or p[1] < 0 or p[1] >= h: 
                return -99999.0
                
            cell = env.unwrapped.grid.get(*p)
            # Empty spaces in MiniGrid are None!
            if cell is None: 
                return self.tracker.penalty_at(p)
                
            if getattr(cell, "type", "unknown") == "wall": 
                return -99999.0
            
            # Reconstruct description for memory lookup
            col = getattr(cell, "color", "unknown")
            obj = getattr(cell, "type", "unknown")
            desc = f"{col} {obj}"
            if obj == "lava" and col == "yellow": desc = "sand"
            
            # Query memory rules to see if this tile type is forbidden
            if desc not in _mem_cache:
                blocked = False
                if self.memory:
                    rule = self.memory.query_local_context(desc, threshold=0.65, silent=True)
                    if rule and rule.get("forbidden_action") == "Move Forward":
                        # Print blocked message only once per tile appearance
                        print(f"\n  [\033[91m⛔ RULE BLOCKED\033[0m] '{desc}' nearby — "
                              f"rule says: \"{rule['rule']}\" → turning away.")
                        blocked = True
                _mem_cache[desc] = blocked
                
            if _mem_cache[desc]:
                return -99999.0
                
            return self.tracker.penalty_at(p)

        front_pen  = get_pen(front)
        left_pen   = get_pen(left_pos)
        right_pen  = get_pen(right_pos)
        behind_pen = get_pen(behind_pos)

        # --- Greedy: pick the direction with the minimum pain ---
        # Penalties are negative or 0. The least painful path is the MAXIMUM value (e.g. 0 > -30)
        options = {
            "front": front_pen,
            "left": left_pen,
            "right": right_pen,
            "behind": behind_pen
        }
        best_dir = max(options, key=options.get)

        if best_dir == "front":
            print(f"\r  [\033[92m✓\033[0m] Safe — stepping into {front}        ", end="", flush=True)
            return 2   # Move Forward
        elif best_dir == "left":
            return 0   # Turn Left
        elif best_dir == "right":
            return 1   # Turn Right
        elif best_dir == "behind":
            return random.choice([0, 1])  # Turn either way to start turning around
