from research import tiles
from research.tiles import CATALOG, COLORS, SPLITS


def test_names_and_labels_are_unique():
    assert len({t.name for t in CATALOG}) == len(CATALOG)
    assert len({t.short for t in CATALOG}) == len(CATALOG)


def test_fields_are_valid():
    for t in CATALOG:
        assert t.color in COLORS
        assert t.split in SPLITS
        assert t.description and 2 <= len(t.short) <= 3


def test_level1_split():
    train_heat = tiles.tiles("train", "heat")
    test_heat = tiles.tiles("test", "heat")
    decoys = [t for t in CATALOG if t.is_decoy]
    assert len(train_heat) >= 3 and len(test_heat) >= 3 and len(decoys) >= 5
    # held-out: no test hazard name appears in training
    assert not {t.name for t in train_heat} & {t.name for t in test_heat}


def test_decoys_look_like_hazards():
    # at least some decoys share the hazards' colour, so colour alone can't separate them
    hazard_colors = {t.color for t in tiles.tiles(category="heat")}
    assert sum(t.color in hazard_colors for t in CATALOG if t.is_decoy) >= 3


def test_deadliness_comes_from_world_categories():
    lava = tiles.get("red lava")
    assert tiles.is_deadly(lava, ["heat"])
    assert not tiles.is_deadly(lava, ["liquid"])
    assert not tiles.is_deadly(tiles.get("red carpet"), ["heat"])
