"""
rl_core.py — RL Agent: Pure Penalty-Driven Exploration
=======================================================
How it works:
  - Every step costs -3 (step penalty). Score starts at 0, decreases.
  - Agent ALWAYS moves to the neighbour tile with the LOWEST cumulative
    penalty (greedy toward less pain = smarter exploration over time).
  - If the agent steps on red lava:   reward = -10, terminated (death).
  - If the agent reaches the goal:    reward = +10, terminated (success).
  - Auto-death: if any tile accumulates ≥ 60 total penalty (visited 20+
    times), the agent is force-killed and that event is logged.
  - On each terminal event a *semantic* fact (tile type, not coordinates)
    is appended to learning_log.json for later LLM synthesis.
"""

import json
import os
import random
import numpy as np

from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

# Reverse look-ups: numeric index → human-readable label
IDX_TO_OBJ   = {v: k for k, v in OBJECT_TO_IDX.items()}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}

LEARNING_LOG  = "learning_log.json"
AUTO_DEATH_THRESHOLD = 60   # cumulative penalty per tile before forced kill


# ─────────────────────────────────────────────────────────────
#  Observation helpers
# ─────────────────────────────────────────────────────────────

def parse_local_observation(obs_image) -> str:
    """Translate the egocentric 7×7 image into a 3-word text description.
    The agent always faces 'up'. Front = (3,5), Left = (2,6), Right = (4,6)."""

    def cell_text(x, y) -> str:
        obj = IDX_TO_OBJ.get(obs_image[x, y, 0], "unknown")
        col = IDX_TO_COLOR.get(obs_image[x, y, 1], "unknown")
        if obj == "empty": return "empty space"
        if obj == "wall":  return "wall"
        if obj == "lava" and col == "yellow": return "sand"
        return f"{col} {obj}"

    return f"Front: {cell_text(3,5)}. Left: {cell_text(2,6)}. Right: {cell_text(4,6)}."


def preprocess_obs(obs) -> np.ndarray:
    """Flatten 7×7×3 MiniGrid image → 147-D float vector."""
    return obs["image"].flatten().astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  Penalty tracker (per-tile cumulative)
# ─────────────────────────────────────────────────────────────

class TilePenaltyTracker:
    """Tracks cumulative step penalty accumulated on each grid tile.
    Each time the agent stands on a tile it costs STEP_COST = -3.
    When any tile's total reaches AUTO_DEATH_THRESHOLD the agent is killed."""

    STEP_COST = -3.0

    def __init__(self):
        self.tile_penalty: dict[tuple, float] = {}   # pos → cumulative penalty
        self.total_steps = 0
        self.total_penalty = 0.0
        self.auto_death_tile: tuple | None = None    # set when 60-rule fires

    def record_step(self, pos: tuple) -> tuple[float, bool]:
        """Call every step.  Returns (step_cost, auto_death_triggered)."""
        self.total_steps += 1
        self.total_penalty += self.STEP_COST
        self.tile_penalty[pos] = self.tile_penalty.get(pos, 0.0) + self.STEP_COST

        cumulative = self.tile_penalty[pos]
        if cumulative <= -AUTO_DEATH_THRESHOLD:
            self.auto_death_tile = pos
            return self.STEP_COST, True     # auto-death

        return self.STEP_COST, False

    def penalty_at(self, pos: tuple) -> float:
        """Return the cumulative penalty already recorded for a tile (≤ 0)."""
        return self.tile_penalty.get(pos, 0.0)

    def reset(self):
        self.tile_penalty.clear()
        self.total_steps = 0
        self.total_penalty = 0.0
        self.auto_death_tile = None


# ─────────────────────────────────────────────────────────────
#  RL Agent: greedy least-penalty explorer
# ─────────────────────────────────────────────────────────────

class RLExplorer:
    """Pure penalty-driven explorer.

    Action space (MiniGrid):  0 = Turn Left, 1 = Turn Right, 2 = Move Forward
    """

    ACTIONS = {0: "Turn Left", 1: "Turn Right", 2: "Move Forward"}

    def __init__(self, eps_start=0.4, eps_min=0.05, eps_decay=0.99):
        self.epsilon   = eps_start
        self.eps_min   = eps_min
        self.eps_decay = eps_decay
        self.tracker   = TilePenaltyTracker()
        self._last_pos = None     # track if agent actually moved
        self._stay_count = 0      # how many steps on same tile

    # ------------------------------------------------------------------
    def select_action(self, env) -> int:
        """Choose action: greedily prefer the neighbour with least pain."""
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        pos    = tuple(int(c) for c in env.unwrapped.agent_pos)
        dir_   = env.unwrapped.agent_dir
        self._last_dir = dir_

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

        def get_pen(p):
            w = env.unwrapped.grid.width
            h = env.unwrapped.grid.height
            if p[0] < 0 or p[0] >= w or p[1] < 0 or p[1] >= h: 
                return -99999.0
            cell = env.unwrapped.grid.get(*p)
            if cell is None:
                return self.tracker.penalty_at(p)
            if getattr(cell, "type", "unknown") == "wall":
                return -99999.0
            return self.tracker.penalty_at(p)

        front_pen  = get_pen(front)
        left_pen   = get_pen(left_pos)
        right_pen  = get_pen(right_pos)
        behind_pen = get_pen(behind_pos)

        # --- Random exploration (epsilon-greedy) ---
        if random.random() < self.epsilon:
            return random.randrange(3)

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
            return 2   # Move Forward
        elif best_dir == "left":
            return 0   # Turn Left
        elif best_dir == "right":
            return 1   # Turn Right
        elif best_dir == "behind":
            return random.choice([0, 1])  # Turn either way to start turning around

    # ------------------------------------------------------------------
    def step_done(self, pos: tuple) -> tuple[float, bool]:
        """Record penalty for the tile the agent just landed on.
        If the agent has stayed on the same tile (due to turns or wall hits),
        the penalty increases more sharply to prevent 'stuck' behaviors.
        """
        if pos == self._last_pos:
            self._stay_count += 1
        else:
            self._stay_count = 0
            self._last_pos = pos

        # If stuck or spinning for too long, penalize this state exponentially
        mult = 1.0 + (self._stay_count // 3) * 2.0  # multiplier grows
        cost, auto = self.tracker.record_step(pos)
        
        # Apply extra penalty if multiplier > 1
        if mult > 1.0:
            extra = (mult - 1.0) * self.tracker.STEP_COST
            self.tracker.tile_penalty[pos] += extra
            self.tracker.total_penalty += extra

        return cost, auto

    # ------------------------------------------------------------------
    def reset_episode(self):
        """Reset penalty state for a new episode."""
        self.tracker.reset()


# ─────────────────────────────────────────────────────────────
#  Learning log helpers
# ─────────────────────────────────────────────────────────────

def _load_log() -> list:
    if os.path.exists(LEARNING_LOG):
        with open(LEARNING_LOG) as f:
            return json.load(f)
    return []


def _save_log(entries: list):
    with open(LEARNING_LOG, "w") as f:
        json.dump(entries, f, indent=2)


def log_death_event(
    phase: int,
    env_name: str,
    tile_description: str,
    auto_death: bool,
    total_steps: int,
    total_penalty: float,
    auto_death_tile_type: str = "none",
):
    """Append a semantic death/event fact to learning_log.json.
    Coordinates are intentionally omitted — only tile *type* is stored."""
    entry = {
        "phase":              phase,
        "environment":        env_name,
        "event":              "auto_death" if auto_death else "tile_death",
        "tile_that_killed":   tile_description,        # e.g. "red lava"
        "auto_death":         auto_death,
        "high_penalty_tile":  auto_death_tile_type,    # tile type at 60-limit, if auto
        "total_steps":        total_steps,
        "total_penalty":      total_penalty,
    }
    entries = _load_log()
    entries.append(entry)
    _save_log(entries)
    print(f"  [📝 Learning Log] Event recorded: {entry['event']} — '{tile_description}'")


def log_success_event(
    phase: int,
    env_name: str,
    total_steps: int,
    total_penalty: float,
):
    """Append a goal-reached fact to learning_log.json."""
    entry = {
        "phase":         phase,
        "environment":   env_name,
        "event":         "success",
        "tile_that_killed": "none",
        "auto_death":    False,
        "total_steps":   total_steps,
        "total_penalty": total_penalty,
    }
    entries = _load_log()
    entries.append(entry)
    _save_log(entries)
    print(f"  [📝 Learning Log] SUCCESS recorded after {total_steps} steps.")


def clear_learning_log():
    """Delete the log so each full experiment run starts fresh."""
    if os.path.exists(LEARNING_LOG):
        os.remove(LEARNING_LOG)
    print("[Learning Log] Cleared for new experiment run.")
