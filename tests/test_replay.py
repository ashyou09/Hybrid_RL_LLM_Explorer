from PIL import Image

from research.config import Config
from research.env import FORWARD, LEFT, TextGridEnv
from research.replay import record_episode, save_gif


def wall_avoider(view):
    return (LEFT, "wall ahead") if view["front"] == "wall" else (FORWARD, "")


def test_replay_is_saved_as_gif(tmp_path):
    env = TextGridEnv(Config(hazard_count=2, decoy_count=2, max_steps=20), split="test")
    frames, summary = record_episode(env, wall_avoider, seed=1)
    assert len(frames) == summary["steps"] + 1
    path = save_gif(frames, tmp_path / "ep.gif")
    with Image.open(path) as gif:
        assert gif.n_frames == len(frames)
