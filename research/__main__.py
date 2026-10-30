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


def _add_run_args(p: argparse.ArgumentParser) -> None:
    from research.shields import SHIELDS
    p.add_argument("--run", help="training run folder (default: latest in results/level_1/runs)")
    p.add_argument("--shields", default=",".join(SHIELDS),
                   help=f"comma-separated, from: {','.join(SHIELDS)}")
    p.add_argument("--provider", help="LLM provider: groq, openrouter, claude, ollama, fake")
    p.add_argument("--model", help="LLM model (default: the provider's default)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="override one knob of the run's config")


def _prepare(args):
    from research import experiments as ex
    run_dir = Path(args.run) if args.run else ex.latest_run()
    cfg, agent, history = ex.load_run(run_dir, args.overrides)
    names = [n.strip() for n in args.shields.split(",") if n.strip()]
    provider = args.provider or cfg.llm_provider
    return ex, run_dir, cfg, agent, history, names, provider


def _report_skipped(skipped: dict) -> None:
    for name, why in skipped.items():
        print(f"  skipped {name}: {why}")


def _write(out, name: str, data: dict, table: str | None = None) -> None:
    (out / f"{name}.json").write_text(json.dumps(data, indent=2, default=str))
    if table:
        (out / f"{name}.md").write_text(table + "\n")


def cmd_classify(args: argparse.Namespace) -> None:
    """L1.10: every shield judges every tile (no RL) -> precision / recall table."""
    ex, run_dir, cfg, _, history, names, provider = _prepare(args)
    log, rules, shields, skipped, llm = ex.build(cfg, history, names, provider, args.model)
    result = ex.classify(shields, cfg.deadly_categories)
    out = ex.out_dir(run_dir, "classify")
    table = ex.classify_table(result)
    _write(out, "classify", {"llm": llm.label, "rules": rules, "skipped": skipped,
                             "result": result}, table)
    print(f"Rules ({llm.label}):\n{__import__('research.rules', fromlist=['x']).to_text(rules)}\n")
    print(table)
    _report_skipped(skipped)
    print(f"\nSaved to {out}  (LLM: {llm.api_calls} API calls, {llm.cache_hits} cached)")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """L1.11: the novice plays test worlds with each shield."""
    ex, run_dir, cfg, agent, history, names, provider = _prepare(args)
    log, rules, shields, skipped, llm = ex.build(cfg, history, names, provider, args.model)
    result = ex.evaluate_shields(agent, cfg, shields)
    out = ex.out_dir(run_dir, "evaluate")
    table = ex.evaluate_table(result)
    _write(out, "evaluate", {"llm": llm.label, "rules": rules, "skipped": skipped,
                             "llm_api_calls": llm.api_calls, "result": result}, table)
    if result:
        ex.plot_evaluation(result, out / "evaluate.png")
        from research.env import TextGridEnv
        from research.qlearning import AgentPolicy
        from research.replay import record_episode, save_gif
        for name, shield in shields.items():
            frames, _ = record_episode(TextGridEnv(cfg, split="test"),
                                       AgentPolicy(agent, mask_fn=shield.mask), seed=1)
            save_gif(frames, out / f"replay_{name}.gif")
    print(table)
    _report_skipped(skipped)
    print(f"\nSaved to {out}  (LLM: {llm.api_calls} API calls, {llm.cache_hits} cached)")


def cmd_sweep(args: argparse.Namespace) -> None:
    """L1.12: change one knob over a range; evaluate every shield at each value."""
    import yaml
    ex, run_dir, cfg, agent, history, names, provider = _prepare(args)
    values = [yaml.safe_load(v) for v in args.values.split(",")]
    result = ex.sweep(agent, cfg, history, args.knob, values, names, provider, args.model)
    out = ex.out_dir(run_dir, f"sweep_{args.knob}")
    _write(out, "sweep", result)
    ex.plot_sweep(result, out / f"sweep_{args.knob}.png")
    for p in result["points"]:
        print(f"{args.knob}={p['value']}: " + "  ".join(
            f"{n} death {r['death']:.0%}" for n, r in p["results"].items()))
    _report_skipped(result["skipped"])
    print(f"\nSaved to {out}")


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

    p = sub.add_parser("classify", help="L1.10: each shield judges every tile (no RL)")
    _add_run_args(p)
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("evaluate", help="L1.11: the novice plays test worlds with each shield")
    _add_run_args(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("sweep", help="L1.12: change one knob over a range of values")
    _add_run_args(p)
    p.add_argument("--knob", required=True, help="e.g. hazard_count, experience_budget, embed_threshold")
    p.add_argument("--values", required=True, help="comma-separated, e.g. 1,2,4,6")
    p.set_defaults(func=cmd_sweep)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
