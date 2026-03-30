"""
environments.py — Custom MiniGrid Rooms for the Experiment
===========================================================
Room 1 (LavaRoom): Red Lava barrier. Agent must learn "lava kills".
Room 2 (QuicksandRoom): Sand barrier. Agent must learn "sand kills".
Room 3 (CombinedTesting): Both hazards combined. The final exam.
"""

import gymnasium as gym
from gymnasium.envs.registration import register
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Wall, WorldObj
from minigrid.minigrid_env import MiniGridEnv
import random


# ---------- Custom Tile: Quicksand (Yellow Lava) ----------

class Quicksand(WorldObj):
    """Sand tile that kills on contact. Inherits 'lava' type so MiniGrid
    handles collision, but uses yellow colour so the LLM must generate
    a *different* semantic rule from red lava."""

    def __init__(self):
        super().__init__("lava", "yellow")

    def can_overlap(self):
        return True

    def render(self, img):
        from minigrid.core.constants import COLORS
        img[:, :, 0] = COLORS["yellow"][0]
        img[:, :, 1] = COLORS["yellow"][1]
        img[:, :, 2] = COLORS["yellow"][2]


# ---------- Room 1: Lava Room ----------

class LavaRoomEnv(MiniGridEnv):
    """7×7 room with a horizontal red‑lava barrier and one gap."""

    def __init__(self, size=7, **kwargs):
        mission_space = MissionSpace(mission_func=lambda: "Avoid red lava.")
        super().__init__(mission_space=mission_space, grid_size=size,
                         max_steps=4 * size ** 2, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(), width - 2, height - 2)
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        # Horizontal lava barrier with one gap at column 3
        for i in range(1, width - 1):
            if i%2 != 0:
                self.grid.set(i, width // 2, Lava())
                self.grid.set(width // 2, i, Lava())

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        cell = self.grid.get(*self.agent_pos)
        reward = -0.1  # time penalty
        if cell and cell.type == "lava" and cell.color == "red":
            reward, terminated = -10.0, True
        if cell and cell.type == "goal":
            reward = 10.0
        return obs, reward, terminated, truncated, info


# ---------- Room 2: Quicksand Room ----------

class QuicksandRoomEnv(MiniGridEnv):
    """7×7 room with a vertical sand barrier and one gap."""

    def __init__(self, size=7, **kwargs):
        mission_space = MissionSpace(mission_func=lambda: "Avoid quicksand.")
        super().__init__(mission_space=mission_space, grid_size=size,
                         max_steps=4 * size ** 2, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(), width - 2, height - 2)
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        # Vertical quicksand wall with a gap at row 4
        for j in range(1, height - 1):
            if j%2 != 0:
                self.grid.set(width // 2-1, j, Quicksand())
                self.grid.set(width // 1 - 3, j, Quicksand())


    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        cell = self.grid.get(*self.agent_pos)
        reward = -0.1  # time penalty
        if cell and cell.type == "lava" and cell.color == "yellow":
            reward, terminated = -10.0, True
        if cell and cell.type == "goal":
            reward = 10.0
        return obs, reward, terminated, truncated, info


# ---------- Room 3: Combined Final Exam ----------

class CombinedTestingEnv(MiniGridEnv):
    """9×9 room with both red lava and sand scattered."""

    def __init__(self, size=9, **kwargs):
        mission_space = MissionSpace(mission_func=lambda: "Avoid all hazards.")
        super().__init__(mission_space=mission_space, grid_size=size,
                         max_steps=4 * size ** 2, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(), width - 2, height - 2)
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        # Lava cluster
        l = random.randint(1, 7)
        k = random.randint(1, 7)

        self.grid.set(l, 5, Lava())
        self.grid.set(k, 2, Lava())
        self.grid.set(3, k, Lava())
        self.grid.set(4, l, Lava())

        

        # Quicksand cluster
        
        self.grid.set(k, 5, Quicksand())
        self.grid.set(6, k, Quicksand())
        self.grid.set(l, 4, Quicksand())
        self.grid.set(7, 4, Quicksand())

        

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        cell = self.grid.get(*self.agent_pos)
        reward = -0.1  # time penalty
        if cell and cell.type == "lava":
            reward, terminated = -10.0, True
        if cell and cell.type == "goal":
            reward = 10.0
        return obs, reward, terminated, truncated, info


# ---------- Register with Gymnasium ----------

try:
    register(id="MiniGrid-LavaRoom-v0",        entry_point="environments:LavaRoomEnv")
    register(id="MiniGrid-QuicksandRoom-v0",    entry_point="environments:QuicksandRoomEnv")
    register(id="MiniGrid-CombinedTesting-v0",  entry_point="environments:CombinedTestingEnv")
except Exception:
    pass  # Already registered during interactive reload
