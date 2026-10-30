"""Episode replays: draw each step with tile labels and a caption, save as a GIF.

Each special tile shows its short label (e.g. "MR" for molten rock); the caption
shows the step, the agent's view, its action and why (e.g. a shield decision).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from research.env import ACTIONS, DescribedTile, TextGridEnv

# A policy maps the text observation to (action, note); the note explains the choice.
Policy = Callable[[dict], tuple[int, str]]

TILE_PX = 48
FRAME_W = 560          # wide enough for readable captions
CAPTION_LINES = 6
WRAP = 78


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:          # Pillow < 10.1
        return ImageFont.load_default()


def draw_frame(env: TextGridEnv, caption: list[str]) -> Image.Image:
    grid = Image.fromarray(env.get_frame(highlight=False, tile_size=TILE_PX))
    draw = ImageDraw.Draw(grid)
    font = _font(TILE_PX // 3)
    for x in range(env.width):
        for y in range(env.height):
            obj = env.grid.get(x, y)
            if isinstance(obj, DescribedTile):
                cx, cy = (x + 0.5) * TILE_PX, (y + 0.5) * TILE_PX
                draw.text((cx, cy), obj.tile.short, fill="white", font=font,
                          anchor="mm", stroke_width=2, stroke_fill="black")

    lines = [w for line in caption for w in textwrap.wrap(line, WRAP) or [""]][:CAPTION_LINES]
    line_h, width = 18, max(grid.width, FRAME_W)
    frame = Image.new("RGB", (width, grid.height + line_h * CAPTION_LINES + 8), "white")
    frame.paste(grid, ((width - grid.width) // 2, 0))
    draw = ImageDraw.Draw(frame)
    for i, line in enumerate(lines):
        draw.text((8, grid.height + 4 + i * line_h), line, fill="black", font=_font(13))
    return frame


def legend(env: TextGridEnv) -> str:
    seen = {t.short: t.name for t in env.tiles_on_grid()}
    return "  ".join(f"{k}={v}" for k, v in sorted(seen.items()))


def record_episode(env: TextGridEnv, policy: Policy, seed: int) -> tuple[list[Image.Image], dict]:
    """Play one episode and return its frames and a summary."""
    _, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    frames = [draw_frame(env, ["step 0 — start", legend(env)])]
    total, steps, event = 0.0, 0, None
    while True:
        view = info["text"]
        action, note = policy(view)
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        event = info["event"]
        outcome = {"died": f"DIED on {info['tile']}", "goal": "REACHED GOAL",
                   "timeout": "time-out"}.get(event, "")
        frames.append(draw_frame(env, [
            f"step {steps}  return {total:+.0f}  {outcome}",
            f"saw: front={view['front']}  left={view['left']}  right={view['right']}",
            f"goal {view['goal_dir']} -> {ACTIONS[action]}" + (f"  ({note})" if note else ""),
            legend(env),
        ]))
        if terminated or truncated:
            break
    return frames, {"seed": seed, "steps": steps, "return": total, "outcome": event}


def save_gif(frames: list[Image.Image], path: Path, ms_per_frame: int = 400) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # hold the last frame longer so the outcome is readable
    durations = [ms_per_frame] * (len(frames) - 1) + [ms_per_frame * 6]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    return path
