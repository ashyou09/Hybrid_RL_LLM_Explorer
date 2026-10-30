import json
import random

import numpy as np
import pytest

from research.config import Config
from research.results import new_run, save_metrics
from research.seeding import seed_everything


def test_same_seed_same_numbers():
    rng_a = seed_everything(7)
    a = (random.random(), np.random.rand(), rng_a.random())
    rng_b = seed_everything(7)
    b = (random.random(), np.random.rand(), rng_b.random())
    assert a == b


def test_config_loads_file_and_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: test\nhazard_count: 5\n")
    cfg = Config.load(path, ["decoy_count=4", "deadly_categories=[heat, sharp]"])
    assert (cfg.name, cfg.hazard_count, cfg.decoy_count) == ("test", 5, 4)
    assert cfg.deadly_categories == ["heat", "sharp"]


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("not_a_knob: 1\n")
    with pytest.raises(ValueError):
        Config.load(path)
    with pytest.raises(ValueError):
        Config().set("typo=3")


def test_level1_config_file_is_valid():
    cfg = Config.load("configs/level1.yaml")
    assert cfg.deadly_categories == ["heat"]


def test_run_folder_records_config_and_metrics(tmp_path):
    cfg = Config(name="unit")
    run_dir = new_run(cfg, root=tmp_path)
    save_metrics(run_dir, {"success_rate": 1.0})
    assert json.loads((run_dir / "config.json").read_text())["name"] == "unit"
    assert "git_commit" in json.loads((run_dir / "meta.json").read_text())
    assert json.loads((run_dir / "metrics.json").read_text())["success_rate"] == 1.0
