"""Tile catalog: every special tile has a name, a natural-language description and a
hidden danger category.

The agent only ever sees ``name`` / ``description``. Whether a tile kills depends on
the world's deadly categories — the agent never sees ``category``.

Splits:
  train – tiles the learner (Agent A) meets
  test  – tiles only the novice (Agent B) meets: held-out hazards and decoys
"""

from __future__ import annotations

from dataclasses import dataclass

# MiniGrid can only draw these colours.
COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
SPLITS = ("train", "test")


@dataclass(frozen=True)
class Tile:
    name: str          # what the agent "sees", e.g. "molten rock"
    description: str   # longer text, used in LLM prompts
    category: str      # hidden: "heat", or "none" for harmless tiles
    split: str         # "train" or "test"
    color: str         # drawing colour (look-alikes share colours on purpose)
    short: str         # 2–3 letter label drawn on the tile in replays

    @property
    def is_decoy(self) -> bool:
        """A harmless test tile that looks or sounds like a hazard."""
        return self.category == "none" and self.split == "test"


CATALOG: tuple[Tile, ...] = (
    # ── heat, train: the learner dies on these ─────────────────
    Tile("red lava", "a pool of glowing red lava", "heat", "train", "red", "LV"),
    Tile("campfire", "a crackling campfire with open flames", "heat", "train", "red", "CF"),
    Tile("glowing hot coals", "a bed of glowing hot coals", "heat", "train", "red", "HC"),
    # ── heat, test: never seen in training ─────────────────────
    Tile("molten rock", "a slab of molten rock oozing heat", "heat", "test", "red", "MR"),
    Tile("burning log pile", "a pile of logs engulfed in fire", "heat", "test", "red", "BL"),
    Tile("white-hot metal plate", "a metal plate heated until it glows white", "heat", "test", "grey", "WM"),
    # ── harmless, train: plain ground the learner walks on ─────
    Tile("grass", "a patch of short green grass", "none", "train", "green", "GR"),
    Tile("wooden floor", "plain wooden floorboards", "none", "train", "yellow", "WF"),
    # ── harmless, test: decoys that look like danger ───────────
    Tile("red carpet", "a soft red carpet", "none", "test", "red", "RC"),
    Tile("red flower bed", "a bed of red flowers", "none", "test", "red", "RF"),
    Tile("painting of a fire", "a flat painting of a fire on the floor", "none", "test", "red", "PF"),
    Tile("orange rug", "a woven orange rug", "none", "test", "yellow", "OR"),
    Tile("cold stove", "an old iron stove that has been off for years", "none", "test", "grey", "CS"),
)

_BY_NAME = {t.name: t for t in CATALOG}


def get(name: str) -> Tile:
    return _BY_NAME[name]


def tiles(split: str | None = None, category: str | None = None) -> list[Tile]:
    """Tiles filtered by split and/or hidden category."""
    return [t for t in CATALOG
            if (split is None or t.split == split)
            and (category is None or t.category == category)]


def is_deadly(tile: Tile, deadly_categories) -> bool:
    return tile.category in set(deadly_categories)
