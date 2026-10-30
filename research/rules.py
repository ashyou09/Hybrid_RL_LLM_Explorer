"""Sparse reflection: the LLM reads Agent A's experience log once and writes rules.

Rule schema (structured claim, research doc RQ3):
    {"id": "R1",
     "if": "the tile is hot, burning or glowing",     # a tile *property*, never a position
     "then": "do not move forward onto it",
     "confidence": 0.9,                                # 0..1, the LLM's own estimate
     "evidence": ["E1", "E2"]}                        # experience-log ids it is based on

Evidence ids make every rule traceable to the experiences behind it (RQ4).
"""

from __future__ import annotations

import json
from pathlib import Path

from research import experience
from research.llm import LLM, extract_json

REFLECT_SYSTEM = (
    "You are the reflection module of a learning agent in a grid world. "
    "You turn the agent's experiences into short, general safety rules. "
    "Reply with JSON only."
)

REFLECT_PROMPT = """\
The agent walks on floor tiles. Some tiles killed it; others it crossed safely.
Its experiences so far (ids in brackets):

{experience}

Write general rules that will help the agent decide about tiles it has NEVER seen.
- Describe the PROPERTY of tiles that kills (what they have in common), not exact
  tile names and never positions.
- Only make claims supported by the experiences; cite the ids you used.
- If something was crossed safely, do not call it dangerous.

Reply with JSON only, in this format:
{{"rules": [{{"id": "R1", "if": "<property of the tile>", "then": "do not move forward onto it",
            "confidence": <0..1>, "evidence": ["E1"]}}]}}
"""


def reflect(log: list[dict], llm: LLM) -> list[dict]:
    """One LLM call: experience log -> validated rules."""
    reply = llm.complete(REFLECT_PROMPT.format(experience=experience.to_text(log)),
                         system=REFLECT_SYSTEM)
    return validate(extract_json(reply).get("rules", []), log)


def validate(raw_rules, log: list[dict]) -> list[dict]:
    """Keep well-formed rules; drop evidence ids that are not in the log."""
    known = {e["id"] for e in log}
    rules = []
    for i, r in enumerate(raw_rules if isinstance(raw_rules, list) else [], 1):
        if not isinstance(r, dict) or not str(r.get("if", "")).strip():
            continue
        try:
            conf = min(1.0, max(0.0, float(r.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        evidence = [e for e in r.get("evidence", []) if e in known] \
            if isinstance(r.get("evidence"), list) else []
        rules.append({"id": f"R{len(rules) + 1}", "if": str(r["if"]).strip(),
                      "then": str(r.get("then") or "do not move forward onto it").strip(),
                      "confidence": round(conf, 3), "evidence": evidence})
    return rules


def to_text(rules: list[dict]) -> str:
    if not rules:
        return "(no rules)"
    return "\n".join(f"[{r['id']}] IF {r['if']} THEN {r['then']} "
                     f"(confidence {r['confidence']}, evidence {', '.join(r['evidence']) or '-'})"
                     for r in rules)


def save(rules: list[dict], path: str | Path) -> None:
    Path(path).write_text(json.dumps(rules, indent=2))
