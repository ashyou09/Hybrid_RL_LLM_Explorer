"""Level 1 experiments on a trained run: classify (L1.10), evaluate (L1.11), sweep (L1.12).

All three start from a training run folder (qtable.json + history.json + config.json),
rebuild the experience log and the LLM rules, then build the requested shields.
Shields that cannot run (no API key, no network, missing package) are skipped with
the reason recorded — never replaced by something else.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from research import experience, rules as rules_mod, tiles
from research.config import Config
from research.llm import LLM, LLMUnavailable
from research.qlearning import QLearner
from research.shields import LABELS, LLM_SHIELDS, SHIELDS, make_shield
from research.train import evaluate

RUNS = Path("results/level_1/runs")


# ── loading ────────────────────────────────────────────────────
def latest_run(root: Path = RUNS) -> Path:
    runs = sorted(p for p in root.glob("*") if (p / "qtable.json").exists())
    if not runs:
        raise FileNotFoundError("No trained run found. Run: python -m research train")
    return runs[-1]


def load_run(run_dir: Path, overrides: list[str] | None = None):
    cfg = Config(**json.loads((run_dir / "config.json").read_text()))
    for item in overrides or []:
        cfg.set(item)
    agent = QLearner.load(run_dir / "qtable.json")
    history = json.loads((run_dir / "history.json").read_text())
    return cfg, agent, history


def build(cfg: Config, history: list[dict], names: list[str], provider: str,
          model: str | None = None):
    """Experience log, rules and shields for one configuration.

    Returns (log, rules, shields, skipped, llm); ``skipped`` maps shield -> reason.
    """
    log = experience.build_log(history, cfg.experience_budget)
    llm = LLM(provider, model or cfg.llm_model or None)
    rules, skipped, shields = [], {}, {}
    if "llm_rules" in names:
        try:
            rules = rules_mod.reflect(log, llm)
        except LLMUnavailable as e:
            skipped["llm_rules"] = str(e)
    for name in names:
        if name in skipped:
            continue
        try:
            shield = make_shield(name, cfg, log, rules, llm if name in LLM_SHIELDS else None)
            if name in LLM_SHIELDS:
                shield.judge(tiles.tiles("test")[0].name)   # fail early if the LLM is unreachable
            shields[name] = shield
        except (LLMUnavailable, RuntimeError) as e:
            skipped[name] = str(e)
    return log, rules, shields, skipped, llm


def out_dir(run_dir: Path, kind: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    d = run_dir / f"{kind}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── L1.10: tile classification ─────────────────────────────────
GROUPS = {
    "seen hazards": lambda t: t.split == "train" and t.category != "none",
    "unseen hazards": lambda t: t.split == "test" and t.category != "none",
    "plain (train)": lambda t: t.split == "train" and t.category == "none",
    "decoys": lambda t: t.is_decoy,
}


def classify(shields: dict, deadly) -> dict:
    """Judge every catalog tile with every shield; no RL involved."""
    out = {}
    for name, shield in shields.items():
        per_tile = {t.name: {"dangerous": (j := shield.judge(t.name)).dangerous,
                             "truth": tiles.is_deadly(t, deadly), "reason": j.reason}
                    for t in tiles.CATALOG}
        groups = {g: float(np.mean([per_tile[t.name]["dangerous"] for t in tiles.CATALOG if f(t)]))
                  for g, f in GROUPS.items()}
        test = [t for t in tiles.CATALOG if t.split == "test"]
        tp = sum(per_tile[t.name]["dangerous"] and per_tile[t.name]["truth"] for t in test)
        fp = sum(per_tile[t.name]["dangerous"] and not per_tile[t.name]["truth"] for t in test)
        fn = sum(not per_tile[t.name]["dangerous"] and per_tile[t.name]["truth"] for t in test)
        out[name] = {"blocked_share": groups,
                     "test_precision": tp / (tp + fp) if tp + fp else None,
                     "test_recall": tp / (tp + fn) if tp + fn else None,
                     "per_tile": per_tile}
    return out


def classify_table(result: dict) -> str:
    head = "| Shield | " + " | ".join(f"blocks {g}" for g in GROUPS) + " | precision (test) | recall (test) |"
    rows = [head, "|" + "---|" * (len(GROUPS) + 3)]
    for name, r in result.items():
        cells = [f"{r['blocked_share'][g]:.0%}" for g in GROUPS]
        p, rc = r["test_precision"], r["test_recall"]
        rows.append(f"| {LABELS[name]} | " + " | ".join(cells) +
                    f" | {'—' if p is None else f'{p:.0%}'} | {'—' if rc is None else f'{rc:.0%}'} |")
    return "\n".join(rows)


# ── L1.11: novice evaluation ───────────────────────────────────
def evaluate_shields(agent: QLearner, cfg: Config, shields: dict) -> dict:
    out = {}
    for name, shield in shields.items():
        m = evaluate(agent, cfg, "test", mask_fn=shield.mask)
        decoys = [t.name for t in tiles.CATALOG if t.is_decoy]
        hazards = [t.name for t in tiles.tiles("test") if tiles.is_deadly(t, cfg.deadly_categories)]
        seen_d = sum(shield.seen[d] for d in decoys)
        seen_h = sum(shield.seen[h] for h in hazards)
        m["decoy_false_avoidance"] = sum(shield.blocked[d] for d in decoys) / seen_d if seen_d else None
        m["hazard_blocking"] = sum(shield.blocked[h] for h in hazards) / seen_h if seen_h else None
        out[name] = m
    return out


def evaluate_table(result: dict) -> str:
    def ci(m):
        return f"{m['mean']:.1%} [{m['ci95'][0]:.1%}, {m['ci95'][1]:.1%}]"

    def pct(x):
        return "—" if x is None else f"{x:.0%}"

    rows = ["| Shield | Success | Death (all unseen hot tiles) | Hazards blocked | Decoys wrongly avoided |",
            "|---|---|---|---|---|"]
    for name, m in result.items():
        rows.append(f"| {LABELS[name]} | {ci(m['success_rate'])} | {ci(m['death_rate'])} | "
                    f"{pct(m['hazard_blocking'])} | {pct(m['decoy_false_avoidance'])} |")
    return "\n".join(rows)


def plot_evaluation(result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (key, label) in enumerate((("success_rate", "success"), ("death_rate", "death"))):
        means = [result[n][key]["mean"] for n in names]
        err = np.array([[result[n][key]["mean"] - result[n][key]["ci95"][0],
                         result[n][key]["ci95"][1] - result[n][key]["mean"]] for n in names]).T
        ax.bar(x + (i - 0.5) * 0.38, means, 0.38, yerr=err, capsize=3, label=label)
    ax.set_xticks(x, [LABELS[n] for n in names], rotation=15)
    ax.set(ylim=(0, 1), ylabel="rate (95% CI)", title="Novice in test worlds (never-seen hot tiles + decoys)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── L1.12: sweeps ──────────────────────────────────────────────
def sweep(agent, base_cfg: Config, history, knob: str, values: list, names: list[str],
          provider: str, model: str | None = None) -> dict:
    """Change one knob, keep everything else fixed, evaluate every shield."""
    points, skipped = [], {}
    for v in values:
        cfg = Config(**base_cfg.to_dict())
        cfg.set(f"{knob}={v}")
        _, _, shields, skip, _ = build(cfg, history, names, provider, model)
        skipped.update(skip)
        res = evaluate_shields(agent, cfg, shields)
        points.append({"value": v, "results": {n: {
            "success": r["success_rate"]["mean"], "death": r["death_rate"]["mean"],
            "decoy_false_avoidance": r["decoy_false_avoidance"]} for n, r in res.items()}})
    return {"knob": knob, "values": values, "points": points, "skipped": skipped}


def plot_sweep(result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted({n for p in result["points"] for n in p["results"]}, key=SHIELDS.index)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in ((axes[0], "death", "death rate"), (axes[1], "success", "success rate")):
        for i, n in enumerate(names):
            xs = [p["value"] for p in result["points"] if n in p["results"]]
            ys = [p["results"][n][key] for p in result["points"] if n in p["results"]]
            # distinct line styles keep identical curves (e.g. none vs lookup) visible
            ax.plot(xs, ys, marker="os^Dv*"[i % 6], markersize=9 - i, linewidth=2.5 - 0.3 * i,
                    linestyle=("-", "--", ":", "-.")[i % 4], label=LABELS[n])
        ax.set(xlabel=result["knob"], ylabel=title, ylim=(0, 1), title=f"{title} vs {result['knob']}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
