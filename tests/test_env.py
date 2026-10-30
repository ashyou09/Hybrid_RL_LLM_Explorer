import re

import pytest

from research import tiles
from research.config import Config
from research.env import (DEATH_REWARD, FORWARD, GOAL_REWARD, LEFT, STEP_REWARD,
                          DescribedTile, TextGridEnv)


def make(split="train", **knobs):
    return TextGridEnv(Config(**knobs), split=split)


def place(env, name, cell):
    env.grid.set(*cell, DescribedTile(tiles.get(name)))


def test_deadly_tile_ends_episode():
    env = make(hazard_count=0, decoy_count=0)
    env.reset(seed=0)
    place(env, "red lava", (2, 1))           # agent at (1,1) facing right
    _, reward, terminated, _, info = env.step(FORWARD)
    assert terminated and reward == DEATH_REWARD
    assert info["event"] == "died" and info["tile"] == "red lava"


def test_safe_tile_does_not_kill():
    env = make(split="test", hazard_count=0, decoy_count=0)
    env.reset(seed=0)
    place(env, "red carpet", (2, 1))
    _, reward, terminated, _, info = env.step(FORWARD)
    assert not terminated and reward == STEP_REWARD and info["event"] == "walked_on"


def test_danger_depends_on_world_not_tile():
    env = make(hazard_count=0, decoy_count=0, deadly_categories=["liquid"])
    env.reset(seed=0)
    place(env, "red lava", (2, 1))
    _, _, terminated, _, _ = env.step(FORWARD)
    assert not terminated


def test_goal_reward():
    env = make(hazard_count=0, decoy_count=0, grid_size=4)   # goal at (2,2)
    env.reset(seed=0)
    env.step(FORWARD)                        # (2,1)
    env.step(1)                              # turn right -> facing down
    _, reward, terminated, _, info = env.step(FORWARD)
    assert terminated and reward == GOAL_REWARD and info["event"] == "goal"


@pytest.mark.parametrize("split", ["train", "test"])
def test_1000_layouts_all_solvable_and_reproducible(split):
    env = make(split=split, hazard_count=6, decoy_count=3)
    for seed in range(1000):
        env.reset(seed=seed)
        assert env._safe_path_exists((1, 1), env.goal_pos)
    env.reset(seed=42)
    first = [(t.name) for t in env.tiles_on_grid()]
    env.reset(seed=42)
    assert [t.name for t in env.tiles_on_grid()] == first


def test_splits_use_only_their_tiles():
    for split in ("train", "test"):
        env = make(split=split, hazard_count=3, decoy_count=3)
        names = set()
        for seed in range(200):
            env.reset(seed=seed)
            names |= {t.name for t in env.tiles_on_grid()}
        assert names <= {t.name for t in tiles.tiles(split)}


def test_observation_has_no_coordinates_or_danger_labels():
    env = make(split="test", hazard_count=4, decoy_count=4)
    _, info = env.reset(seed=3)
    for _ in range(30):
        text = info["text"]
        core = {"front", "left", "right", "goal_dir"}
        assert core <= set(text) <= core | {"stuck", "ahead_visited"}
        flat = " ".join(text.values())
        assert not re.search(r"\d", flat)
        assert "heat" not in flat and "deadly" not in flat
        _, _, term, trunc, info = env.step(LEFT if text["front"] == "wall" else FORWARD)
        if term or trunc:
            _, info = env.reset(seed=4)


def test_compass():
    env = make(hazard_count=0, decoy_count=0)
    env.reset(seed=0)            # (1,1) facing right, goal (5,5): equal distance -> ahead
    assert env.goal_direction() == "ahead"
    env.step(LEFT)               # facing up: goal is behind
    assert env.goal_direction() == "behind"


def test_stuck_sense_turns_on_after_three_steps_in_place():
    env = make(hazard_count=0, decoy_count=0, stuck_signal=True)
    _, info = env.reset(seed=0)
    assert info["text"]["stuck"] == "no"
    for _ in range(3):
        _, _, _, _, info = env.step(LEFT)
    assert info["text"]["stuck"] == "yes"
    _, _, _, _, info = env.step(FORWARD)  # three lefts: facing down, the cell below is open
    assert info["text"]["stuck"] == "no"
