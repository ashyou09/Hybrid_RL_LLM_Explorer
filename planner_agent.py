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
"""

from minigrid.core.world_object import Floor
from rl_core import parse_local_observation


class OnlineExplorerAgent:
    """Real‑time depth‑first maze explorer with LLM rule masking."""

    def __init__(self, memory_hub=None):
        self.memory     = memory_hub    # The ChromaDB Vector Store
        self.visited    = set()         # Set of (x, y) tuples already explored
        self.path_stack = []            # Stack of positions for backtracking
        self.backtracking   = False     # Currently retreating?
        self.turn_count     = 0         # How many 90° turns at current tile

    # ────────────────────────────────────────────
    # Main decision loop (called every tick)
    # ────────────────────────────────────────────

    def act(self, env, obs):
        """Returns an action: 0=Turn Left, 1=Turn Right, 2=Move Forward."""
        pos = tuple(int(c) for c in env.unwrapped.agent_pos)   # clean ints
        dir_idx = env.unwrapped.agent_dir

        # ── Paint green breadcrumb on the tile we are standing on ──
        env.unwrapped.grid.set(pos[0], pos[1], Floor("green"))
        self.visited.add(pos)

        if not self.path_stack:
            self.path_stack.append(pos)

        # Direction vector the agent is currently facing
        dx, dy = [(1,0), (0,1), (-1,0), (0,-1)][dir_idx]
        front = (pos[0] + dx, pos[1] + dy)

        # ═══════════════════════════════════════
        # A. BACKTRACKING MODE
        # ═══════════════════════════════════════
        if self.backtracking:
            if len(self.path_stack) <= 1:
                # Fully trapped — nowhere to go
                return 0

            target = self.path_stack[-2]         # one step back on the trail
            tdx = target[0] - pos[0]
            tdy = target[1] - pos[1]

            if (tdx, tdy) == (dx, dy):
                # Facing the target — step forward to retreat
                print(f"\n  [←] Backtracking to {target}")
                self.backtracking = False
                self.turn_count = 0
                self.path_stack.pop()
                return 2  # Move Forward

            # Not facing target yet — rotate towards it
            if (tdx, tdy) == (dy, -dx):  return 0   # Turn Left
            if (tdx, tdy) == (-dy, dx):  return 1   # Turn Right
            return 1  # 180° needs two rights

        # ═══════════════════════════════════════
        # B. EXPLORATION MODE
        # ═══════════════════════════════════════
        local_text  = parse_local_observation(obs["image"])
        front_block = local_text.split(".")[0].replace("Front: ", "")

        # B1. Wall or already visited → turn right
        if front_block == "wall" or front in self.visited:
            self.turn_count += 1
            if self.turn_count >= 4:
                print(f"\n  [↺] Dead end at {pos} — backtracking")
                self.backtracking = True
            return 1

        # B2. Vector DB says it's dangerous → treat as impassable
        if self.memory:
            rule = self.memory.query_local_context(front_block, threshold=0.70, silent=True)
            if rule and rule.get("forbidden_action") == "Move Forward":
                print(f"\n  [\033[91m✗\033[0m] {front_block} ahead — Rule: {rule['rule']}")
                self.visited.add(front)
                self.turn_count += 1
                if self.turn_count >= 4:
                    print(f"  [↺] All routes blocked at {pos} — backtracking")
                    self.backtracking = True
                return 1

        # B3. Safe & unvisited → move forward and mark green
        print(f"\r  [\033[92m✓\033[0m] Safe — stepping into {front}        ", end="", flush=True)
        self.path_stack.append(front)
        self.turn_count = 0
        return 2  # Move Forward
