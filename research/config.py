"""Experiment configuration: every knob of a run lives in one ``Config``.

A run is fully described by its config file plus the git commit, so any number in a
write-up can be reproduced with ``python -m research <command> --config <file>``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    # ── identity ──────────────────────────────────────────────
    name: str = "level1"
    seed: int = 0

    # ── world knobs ───────────────────────────────────────────
    grid_size: int = 7                 # room size including walls
    max_steps: int = 100               # episode time-out
    hazard_count: int = 3              # deadly tiles per episode (difficulty)
    decoy_count: int = 2               # harmless tiles per episode (plain in training, look-alikes in testing)
    deadly_categories: list[str] = field(default_factory=lambda: ["heat"])

    # ── learner (Agent A) knobs ───────────────────────────────
    train_episodes: int = 6000
    alpha: float = 0.2                 # Q-learning step size
    gamma: float = 0.95                # discount
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.7  # share of training spent decaying epsilon
    epsilon_eval: float = 0.0          # exploration during evaluation (breaks loops if > 0)

    # ── evaluation knobs ──────────────────────────────────────
    eval_seeds: int = 10
    eval_episodes: int = 100

    # ── rule / shield knobs (used from step L1.7 on) ──────────
    experience_budget: int = 5         # learner deaths given to the LLM
    shield: str = "none"
    llm_provider: str = "claude"
    llm_model: str = "claude-haiku-5-5"
    embed_threshold: float = 0.65

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None, overrides: list[str] | None = None) -> "Config":
        """Load a YAML file (optional) and apply ``key=value`` overrides."""
        data = {}
        if path:
            data = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = cls(**data)
        for item in overrides or []:
            cfg.set(item)
        return cfg

    def set(self, item: str) -> None:
        """Apply one ``key=value`` override, parsing the value as YAML."""
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        if key not in {f.name for f in fields(self)}:
            raise ValueError(f"Unknown config key: {key!r}")
        setattr(self, key, yaml.safe_load(raw))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
