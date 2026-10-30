"""One place to fix all randomness, so the same seed always gives the same run."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python's and NumPy's global RNGs and return a fresh NumPy generator.

    Environments are seeded separately through ``env.reset(seed=...)``.
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
