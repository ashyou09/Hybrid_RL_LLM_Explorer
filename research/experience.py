"""Experience log: what Agent A went through, in words — the only thing the LLM sees.

Built from the learner's training history, cut off once it has died
``experience_budget`` times (sparse reflection: the LLM wakes up after N failures).
Entries are aggregated by tile and never contain coordinates:

    {"id": "E1", "event": "died_on",          "tile": "red lava",  "description": "...", "count": 2}
    {"id": "E2", "event": "walked_safely_on", "tile": "grass",     "description": "...", "count": 31}
    {"id": "E3", "event": "reached_goal",     "tile": null,        "description": null,  "count": 4}

Each rule the LLM writes later cites the ids of the entries it is based on.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from research import tiles

EVENTS = ("died_on", "walked_safely_on", "reached_goal")
FIELDS = {"id", "event", "tile", "description", "count"}


def build_log(history: list[dict], experience_budget: int) -> list[dict]:
    """Aggregate training history up to and including the N-th death."""
    counts: Counter = Counter()
    deaths = 0
    for ep in history:
        for e in ep["events"]:
            counts[("died_on" if e["event"] == "died" else "walked_safely_on", e["tile"])] += 1
        if ep["outcome"] == "goal":
            counts[("reached_goal", None)] += 1
        if ep["outcome"] == "died":
            deaths += 1
            if deaths >= experience_budget:
                break

    order = {e: i for i, e in enumerate(EVENTS)}
    entries = sorted(counts.items(), key=lambda kv: (order[kv[0][0]], -kv[1], kv[0][1] or ""))
    return [{"id": f"E{i}", "event": event, "tile": tile,
             "description": tiles.get(tile).description if tile else None, "count": n}
            for i, ((event, tile), n) in enumerate(entries, 1)]


def validate(log: list[dict]) -> None:
    """Raise ValueError if the log breaks the schema (or leaks positions)."""
    for e in log:
        if set(e) != FIELDS:
            raise ValueError(f"bad fields in {e}")
        if e["event"] not in EVENTS:
            raise ValueError(f"bad event in {e}")
        if (e["tile"] is None) != (e["event"] == "reached_goal"):
            raise ValueError(f"tile must be set exactly for tile events: {e}")
        if any(ch.isdigit() for ch in json.dumps({k: e[k] for k in ("tile", "description")})):
            raise ValueError(f"log must not contain coordinates: {e}")
        if not isinstance(e["count"], int) or e["count"] < 1:
            raise ValueError(f"bad count in {e}")


def to_text(log: list[dict]) -> str:
    """Plain-English lines for an LLM prompt."""
    lines = []
    for e in log:
        times = f"{e['count']} time" + ("s" if e["count"] != 1 else "")
        if e["event"] == "died_on":
            lines.append(f"[{e['id']}] DIED after stepping on {e['tile']} ({e['description']}) — {times}.")
        elif e["event"] == "walked_safely_on":
            lines.append(f"[{e['id']}] Walked safely over {e['tile']} ({e['description']}) — {times}.")
        else:
            lines.append(f"[{e['id']}] Reached the goal — {times}.")
    return "\n".join(lines)


def save(log: list[dict], path: str | Path) -> None:
    validate(log)
    Path(path).write_text(json.dumps(log, indent=2))
