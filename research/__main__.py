"""Command-line entry point: ``python -m research <command> [options]``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.config import Config


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/level1.yaml", help="YAML config file")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="override one knob (repeatable)")


def _load(args) -> Config:
    return Config.load(args.config, args.overrides)


def cmd_config(args: argparse.Namespace) -> None:
    print(_load(args).to_json())


def cmd_train(args: argparse.Namespace) -> None:
    """Train Agent A, evaluate it with no shield on train and test worlds, save everything."""
    from research import experience
    from research.env import TextGridEnv
    from research.qlearning import AgentPolicy
    from research.replay import record_episode, save_gif
    from research.results import new_run, save_metrics
    from research.seeding import seed_everything
    from research.train import evaluate, plot_curve, train

    cfg = _load(args)
    run_dir = new_run(cfg)
    print(f"Run folder: {run_dir}")
    agent, history = train(cfg, seed_everything(cfg.seed))
    agent.save(run_dir / "qtable.json")
    (run_dir / "history.json").write_text(json.dumps(history))
    plot_curve(history, run_dir / "learning_curve.png")
    log = experience.build_log(history, cfg.experience_budget)
    experience.save(log, run_dir / "experience_log.json")
    print(f"Experience log ({cfg.experience_budget} deaths):\n{experience.to_text(log)}")

    metrics = {"train": evaluate(agent, cfg, "train"), "test": evaluate(agent, cfg, "test"),
               "q_states": len(agent.q)}
    save_metrics(run_dir, metrics)
    for split in ("train", "test"):
        env = TextGridEnv(cfg, split=split)
        frames, _ = record_episode(env, AgentPolicy(agent), seed=1)
        save_gif(frames, run_dir / f"replay_{split}.gif")
        m = metrics[split]
        print(f"{split:5s}  success {m['success_rate']['mean']:.1%}  "
              f"death {m['death_rate']['mean']:.1%}  timeout {m['timeout_rate']['mean']:.1%}"
              f"  deaths by tile: {m['deaths_by_tile']}")


def cmd_replay(args: argparse.Namespace) -> None:
    """Save a GIF of one episode, played by a trained agent or a random one."""
    from research.env import TextGridEnv
    from research.qlearning import AgentPolicy, QLearner
    from research.replay import record_episode, save_gif

    cfg = _load(args)
    rng = np.random.default_rng(args.seed)
    if args.qtable:
        policy = AgentPolicy(QLearner.load(args.qtable, rng), cfg.epsilon_eval)
    else:
        policy = lambda view: (int(rng.integers(3)), "random")   # noqa: E731
    frames, summary = record_episode(TextGridEnv(cfg, split=args.split), policy, args.seed)
    print(f"Saved {save_gif(frames, Path(args.out))}: {summary}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m research",
                                     description="Semantic Osmosis research experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("config", help="print the merged config of a run")
    _add_config_args(p)
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("train", help="train Agent A and evaluate it without a shield")
    _add_config_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("replay", help="save one episode as a GIF")
    _add_config_args(p)
    p.add_argument("--qtable", help="trained agent (qtable.json); random agent if omitted")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="replay.gif")
    p.set_defaults(func=cmd_replay)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
