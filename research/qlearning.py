"""Agent A: tabular Q-learning on the text observation.

State = (front cell name, left kind, right kind, goal direction, previous action),
where a side "kind" is "wall", "open" (empty floor or goal) or "tile" (any special
tile). The previous action is a one-step memory that stops the agent from turning
back and forth forever (without it ~40% of episodes time out). The front cell
keeps its full name, so the agent can learn "red lava ahead → don't move forward".
A tile it has never seen (e.g. "molten rock") is an unseen state with no knowledge —
exactly the gap the rules and shields of later steps must fill.

``act`` accepts an action mask: masked actions are never chosen (used by shields).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.env import EMPTY, GOAL, WALL

PREV = ("L", "R", "F", "B")   # previous action as part of the state; "start" at reset


def _kind(name: str) -> str:
    if name == WALL:
        return "wall"
    return "open" if name in (EMPTY, GOAL) else "tile"


def state_of(view: dict, prev_action: int | None = None) -> str:
    prev = "start" if prev_action is None else PREV[prev_action]
    parts = [view["front"], _kind(view["left"]), _kind(view["right"]), view["goal_dir"], prev]
    if "stuck" in view:                       # only when the stuck_signal knob is on
        parts.append("stuck" if view["stuck"] == "yes" else "moving")
    if "ahead_visited" in view:               # only when the trail_memory knob is on
        parts.append("trail" if view["ahead_visited"] == "yes" else "new")
    return "|".join(parts)


class QLearner:
    def __init__(self, alpha: float = 0.2, gamma: float = 0.95,
                 rng: np.random.Generator | None = None, n_actions: int = 3):
        self.alpha, self.gamma, self.n_actions = alpha, gamma, n_actions
        self.rng = rng or np.random.default_rng()
        self.q: dict[str, np.ndarray] = {}

    def values(self, state: str) -> np.ndarray:
        return self.q.get(state, np.zeros(self.n_actions))

    def knows(self, state: str) -> bool:
        return state in self.q

    def act(self, state: str, epsilon: float = 0.0, mask=None) -> int:
        allowed = [a for a in range(self.n_actions) if mask is None or mask[a]]
        if not allowed:
            raise ValueError("All actions are masked")
        if self.rng.random() < epsilon:
            return int(self.rng.choice(allowed))
        vals = self.values(state)[allowed]
        best = np.flatnonzero(vals == vals.max())
        return int(allowed[int(self.rng.choice(best))])

    def update(self, s: str, a: int, r: float, s2: str, done: bool) -> None:
        q = self.q.setdefault(s, np.zeros(self.n_actions))
        target = r if done else r + self.gamma * self.values(s2).max()
        q[a] += self.alpha * (target - q[a])

    # ── persistence ────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        data = {"alpha": self.alpha, "gamma": self.gamma, "n_actions": self.n_actions,
                "q": {s: v.round(4).tolist() for s, v in sorted(self.q.items())}}
        Path(path).write_text(json.dumps(data, indent=1))

    @classmethod
    def load(cls, path: str | Path, rng: np.random.Generator | None = None) -> "QLearner":
        data = json.loads(Path(path).read_text())
        agent = cls(data["alpha"], data["gamma"], rng, data.get("n_actions", 3))
        agent.q = {s: np.asarray(v, dtype=float) for s, v in data["q"].items()}
        return agent


class AgentPolicy:
    """Wraps a QLearner as a replay policy, tracking the previous action itself.

    ``mask_fn(view) -> (mask, note)`` optionally lets a shield forbid actions.
    """

    def __init__(self, agent: QLearner, epsilon: float = 0.0, mask_fn=None):
        self.agent, self.epsilon, self.mask_fn = agent, epsilon, mask_fn
        self.prev = None

    def reset(self) -> None:
        self.prev = None

    def __call__(self, view: dict) -> tuple[int, str]:
        state = state_of(view, self.prev)
        mask, note = self.mask_fn(view) if self.mask_fn else (None, "")
        if not note and not self.agent.knows(state):
            note = "never seen this situation"
        self.prev = self.agent.act(state, self.epsilon, mask)
        return self.prev, note
