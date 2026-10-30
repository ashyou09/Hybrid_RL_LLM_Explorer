"""Command-line entry point: ``python -m research <command> [options]``."""

from __future__ import annotations

import argparse

from research.config import Config


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/level1.yaml", help="YAML config file")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="override one knob (repeatable)")


def cmd_config(args: argparse.Namespace) -> None:
    print(Config.load(args.config, args.overrides).to_json())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m research",
                                     description="Semantic Osmosis research experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("config", help="print the merged config of a run")
    _add_config_args(p)
    p.set_defaults(func=cmd_config)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
