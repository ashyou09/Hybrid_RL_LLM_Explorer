"""
rl_core.py — System 1: The RL Agent (PyTorch DQN)
===================================================
Responsibilities:
  - parse_local_observation(): Converts the 7×7 MiniGrid image → text description.
  - QNetwork: 3-layer fully-connected Q-value estimator.
  - DQNAgent: Epsilon-greedy policy with action masking + failure logging.
  - preprocess_obs(): Flattens observation dict → 1-D numpy array for the network.
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

# Reverse look-up: numeric index → human-readable name
IDX_TO_OBJ  = {v: k for k, v in OBJECT_TO_IDX.items()}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}


def parse_local_observation(obs_image):
    """Translate the egocentric 7×7 image into a 3-word text description.
    The agent always faces 'up'. Front = (3,5), Left = (2,6), Right = (4,6)."""

    def cell_text(x, y):
        obj = IDX_TO_OBJ.get(obs_image[x, y, 0], "unknown")
        col = IDX_TO_COLOR.get(obs_image[x, y, 1], "unknown")
        if obj == "empty": return "empty space"
        if obj == "wall":  return "wall"
        if obj == "lava" and col == "yellow": return "sand"
        return f"{col} {obj}"

    return f"Front: {cell_text(3,5)}. Left: {cell_text(2,6)}. Right: {cell_text(4,6)}."


class QNetwork(nn.Module):
    """Simple 3-layer MLP: 147 inputs (7×7×3 flattened) → 3 action Q-values."""

    def __init__(self, action_dim=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(147, 128), nn.ReLU(),
            nn.Linear(128,  64), nn.ReLU(),
            nn.Linear( 64, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    """Epsilon-greedy DQN agent. Used here only for blind exploration and
    failure logging — the training loop (update/replay) is not needed."""

    def __init__(self, action_dim=3, lr=1e-3, eps_start=1.0,
                 eps_end=0.05, eps_decay=1000):
        self.action_dim = action_dim
        self.epsilon    = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.steps_done = 0

        self.q_net     = QNetwork(action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def select_action(self, state, apply_mask=None):
        """Pick an action via epsilon-greedy.
        apply_mask: list like [1,1,0] — actions with 0 are forbidden."""
        self.steps_done += 1
        self.epsilon = max(self.eps_end,
            self.eps_end + (1.0 - self.eps_end) * np.exp(-self.steps_done / self.eps_decay))

        # Random exploration
        if random.random() < self.epsilon:
            if apply_mask is None:
                return random.randrange(self.action_dim)
            allowed = [i for i, v in enumerate(apply_mask) if v == 1.0]
            return random.choice(allowed) if allowed else random.randrange(self.action_dim)

        # Greedy exploitation
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(state).unsqueeze(0)).squeeze(0)
            if apply_mask is not None:
                q[torch.FloatTensor(apply_mask) == 0.0] = -1e9
            return int(q.argmax().item())

    def trigger_failure_log(self, env_name, obs_image, action_taken):
        """Write a JSON snapshot of the fatal moment for the LLM to reflect on."""
        action_names = {0: "Turn Left", 1: "Turn Right", 2: "Move Forward"}
        log = {
            "environment":   env_name,
            "fatal_action":  action_names.get(action_taken, "Unknown"),
            "state_context": parse_local_observation(obs_image),
        }
        with open("failure_log.json", "w") as f:
            json.dump(log, f, indent=4)


def preprocess_obs(obs):
    """Flatten the 7×7×3 MiniGrid observation image → 147-D float vector."""
    return obs["image"].flatten().astype(np.float32)
