"""Text-layer MiniGrid world.

A walled room with a start (top-left), a goal (bottom-right) and special tiles from
the catalog placed at random each episode. Each special tile has a natural-language
name; whether it kills depends on the world's hidden ``deadly_categories``.

The agent perceives only:
  front / left / right – names of the three neighbouring cells ("wall",
                         "empty floor", "goal" or a tile name)
  goal_dir             – a compass: goal is "ahead", "left", "right" or "behind"
No coordinates, no map, no danger labels.

Rewards: -1 per step, -10 and episode end on a deadly tile, +10 at the goal.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from minigrid.core.constants import COLORS as RGB
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_rect

from research import tiles as tile_catalog
from research.config import Config
from research.tiles import Tile

STEP_REWARD = -1.0
DEATH_REWARD = -10.0
GOAL_REWARD = 10.0

ACTIONS = ("turn left", "turn right", "move forward")   # MiniGrid actions 0, 1, 2
LEFT, RIGHT, FORWARD = 0, 1, 2

EMPTY, WALL, GOAL = "empty floor", "wall", "goal"


class DescribedTile(WorldObj):
    """A walkable tile carrying a catalog entry. Drawn as a coloured square;
    its danger is never encoded in what the agent sees."""

    def __init__(self, tile: Tile):
        super().__init__("floor", tile.color)
        self.tile = tile

    def can_overlap(self) -> bool:
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), RGB[self.color] * 0.75)


class TextGridEnv(MiniGridEnv):
    """MiniGrid room with described tiles. ``split`` picks train or test tiles."""

    def __init__(self, cfg: Config, split: str = "train", render_mode: str | None = None,
                 max_layout_tries: int = 200):
        self.cfg = cfg
        self.split = split
        self.deadly = set(cfg.deadly_categories)
        pool = tile_catalog.tiles(split)
        self.hazard_pool = [t for t in pool if t.category in self.deadly]
        self.safe_pool = [t for t in pool if t.category not in self.deadly]
        if cfg.hazard_count and not self.hazard_pool:
            raise ValueError(f"No deadly tiles in split {split!r} for {sorted(self.deadly)}")
        self.max_layout_tries = max_layout_tries
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: "reach the goal safely"),
            grid_size=cfg.grid_size, max_steps=cfg.max_steps,
            see_through_walls=True, render_mode=render_mode,
        )

    # ── layout ─────────────────────────────────────────────────
    def _gen_grid(self, width: int, height: int) -> None:
        start, goal = (1, 1), (width - 2, height - 2)
        interior = [(x, y) for x in range(1, width - 1) for y in range(1, height - 1)
                    if (x, y) not in (start, goal)]
        for _ in range(self.max_layout_tries):
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)
            self.put_obj(Goal(), *goal)
            n = min(self.cfg.hazard_count + self.cfg.decoy_count, len(interior))
            cells = [interior[i] for i in self.np_random.permutation(len(interior))[:n]]
            for i, cell in enumerate(cells):
                is_hazard = i < self.cfg.hazard_count
                pool = self.hazard_pool if is_hazard else self.safe_pool
                if pool:
                    tile = pool[int(self.np_random.integers(len(pool)))]
                    self.grid.set(*cell, DescribedTile(tile))
            if self._safe_path_exists(start, goal):
                break
        else:
            raise RuntimeError("Could not generate a solvable layout; lower hazard_count")
        self.agent_pos = start
        self.agent_dir = 0
        self.goal_pos = goal

    def _deadly_at(self, cell) -> bool:
        obj = self.grid.get(*cell)
        return isinstance(obj, DescribedTile) and obj.tile.category in self.deadly

    def _safe_path_exists(self, start, goal) -> bool:
        seen, queue = {start}, deque([start])
        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                obj = self.grid.get(nx, ny)
                if (nx, ny) in seen or (obj is not None and obj.type == "wall") \
                        or self._deadly_at((nx, ny)):
                    continue
                seen.add((nx, ny))
                queue.append((nx, ny))
        return False

    # ── perception ─────────────────────────────────────────────
    def cell_name(self, cell) -> str:
        obj = self.grid.get(*cell)
        if obj is None:
            return EMPTY
        if isinstance(obj, DescribedTile):
            return obj.tile.name
        return GOAL if obj.type == "goal" else WALL if obj.type == "wall" else obj.type

    def goal_direction(self) -> str:
        g = np.asarray(self.goal_pos) - np.asarray(self.agent_pos)
        fwd, right = int(g @ self.dir_vec), int(g @ self.right_vec)
        if fwd == right == 0:
            return "here"
        if abs(fwd) >= abs(right):
            return "ahead" if fwd > 0 else "behind"
        return "right" if right > 0 else "left"

    def text_obs(self) -> dict:
        pos, right = np.asarray(self.agent_pos), np.asarray(self.right_vec)
        return {
            "front": self.cell_name(tuple(self.front_pos)),
            "left": self.cell_name(tuple(pos - right)),
            "right": self.cell_name(tuple(pos + right)),
            "goal_dir": self.goal_direction(),
        }

    # ── dynamics ───────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info.update(text=self.text_obs(), event=None, tile=None)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        reward, event, tile = STEP_REWARD, None, None
        here = self.grid.get(*self.agent_pos)
        if isinstance(here, DescribedTile):
            tile = here.tile.name
            if here.tile.category in self.deadly:
                reward, terminated, event = DEATH_REWARD, True, "died"
            else:
                event = "walked_on"
        elif here is not None and here.type == "goal":
            reward, terminated, event = GOAL_REWARD, True, "goal"
        if truncated and not terminated:
            event = "timeout"
        info.update(text=self.text_obs(), event=event, tile=tile)
        return obs, reward, terminated, truncated, info

    def tiles_on_grid(self) -> list[Tile]:
        return [obj.tile for obj in self.grid.grid if isinstance(obj, DescribedTile)]
