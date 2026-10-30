"""Training and evaluating Agent A (no rules yet — this is shield S0, "none")."""

from __future__ import annotations

from collections import Counter

import numpy as np

from research.config import Config
from research.env import TextGridEnv
from research.qlearning import QLearner, state_of

# Disjoint seed ranges so training and evaluation never share a layout.
TRAIN_SEED_BASE = 0
EVAL_SEED_BASE = 10_000_000


def epsilon_at(cfg: Config, episode: int) -> float:
    decay = max(1, int(cfg.train_episodes * cfg.epsilon_decay_fraction))
    frac = min(1.0, episode / decay)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def train(cfg: Config, rng: np.random.Generator) -> tuple[QLearner, list[dict]]:
    """Train Agent A in training worlds. Returns the agent and one record per episode."""
    env = TextGridEnv(cfg, split="train")
    agent = QLearner(cfg.alpha, cfg.gamma, rng)
    history = []
    for ep in range(cfg.train_episodes):
        eps = epsilon_at(cfg, ep)
        _, info = env.reset(seed=TRAIN_SEED_BASE + cfg.seed * 1_000_000 + ep)
        s = state_of(info["text"])
        total, steps, events = 0.0, 0, []
        while True:
            a = agent.act(s, eps)
            _, r, term, trunc, info = env.step(a)
            s2 = state_of(info["text"], a)
            agent.update(s, a, r, s2, term)
            s, total, steps = s2, total + r, steps + 1
            if info["event"] in ("died", "walked_on"):
                events.append({"event": info["event"], "tile": info["tile"]})
            if term or trunc:
                break
        history.append({"episode": ep, "epsilon": round(eps, 4), "outcome": info["event"],
                        "steps": steps, "return": total, "events": events})
    return agent, history


def evaluate(agent: QLearner, cfg: Config, split: str, mask_fn=None) -> dict:
    """Greedy evaluation over ``eval_seeds`` × ``eval_episodes`` fresh layouts.

    ``mask_fn(view) -> mask`` or ``-> (mask, note)`` lets a shield forbid actions
    (None = no shield). The agent's tie-breaking RNG is reset first, so every shield
    is evaluated on the same layouts with the same random draws (a paired comparison).
    """
    env = TextGridEnv(cfg, split=split)
    agent.rng = np.random.default_rng(EVAL_SEED_BASE + cfg.seed)
    per_seed, deaths_by_tile, unseen_states = [], Counter(), 0
    for s_idx in range(cfg.eval_seeds):
        outcomes = Counter()
        for ep in range(cfg.eval_episodes):
            _, info = env.reset(seed=EVAL_SEED_BASE + s_idx * 100_000 + ep)
            prev = None
            while True:
                view = info["text"]
                state = state_of(view, prev)
                unseen_states += not agent.knows(state)
                mask = mask_fn(view) if mask_fn else None
                if isinstance(mask, tuple):
                    mask = mask[0]
                prev = agent.act(state, cfg.epsilon_eval, mask)
                _, _, term, trunc, info = env.step(prev)
                if term or trunc:
                    break
            outcomes[info["event"]] += 1
            if info["event"] == "died":
                deaths_by_tile[info["tile"]] += 1
        n = cfg.eval_episodes
        per_seed.append({k: outcomes[k] / n for k in ("goal", "died", "timeout")})

    def summary(key):
        vals = np.array([p[key] for p in per_seed])
        return {"mean": float(vals.mean()), "ci95": bootstrap_ci(vals)}

    return {
        "split": split,
        "success_rate": summary("goal"),
        "death_rate": summary("died"),
        "timeout_rate": summary("timeout"),
        "deaths_by_tile": dict(deaths_by_tile.most_common()),
        "unseen_state_visits": unseen_states,
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """95% bootstrap confidence interval of the mean over seeds."""
    if len(values) < 2:
        return [float(values.mean())] * 2
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (n_boot, len(values))).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def plot_curve(history: list[dict], path, window: int = 100) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rolling(key):
        hits = np.array([h["outcome"] == key for h in history], dtype=float)
        w = min(window, len(hits))
        return np.convolve(hits, np.ones(w) / w, mode="valid")

    x = np.arange(len(rolling("goal"))) + min(window, len(history))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, rolling("goal"), label="reached goal")
    ax.plot(x, rolling("died"), label="died")
    ax.plot(x, rolling("timeout"), label="timed out")
    ax.set(xlabel="training episode", ylabel=f"rate (rolling {window})", ylim=(0, 1),
           title="Agent A learning curve (training worlds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
