"""
reflection_engine.py — LLM Rule Synthesis from Learning Log
=============================================================
Single LLM call after Phase 1. Reads learning_log.json and generates
plain-language navigation rules the agent can follow in Phase 3.

Key fix: the prompt now includes a concrete JSON example so the LLM
always returns the exact forbidden_action strings needed.
"""

import json
import os

LEARNING_LOG  = "learning_log.json"
_VALID_ACTIONS = {"Move Forward", "Turn Left", "Turn Right"}


# ─────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────

def synthesize_rules_from_log(log_path: str = LEARNING_LOG) -> list[dict]:
    """Read the learning log -> call LLM once -> return validated rule list.

    Each returned rule dict:
        {
            "rule":             str,   # plain-language sentence
            "forbidden_action": str,   # "Move Forward" | "Turn Left" | "Turn Right"
            "trigger_feature":  str,   # 1-3 word tile label, e.g. "red lava"
        }
    """
    if not os.path.exists(log_path):
        print("[Reflection Engine] No learning log found — nothing to synthesize.")
        return []

    with open(log_path) as f:
        log_entries = json.load(f)

    if not log_entries:
        print("[Reflection Engine] Learning log is empty.")
        return []

    print(f"\n[Reflection Engine] Synthesizing rules from {len(log_entries)} log entries...")
    _print_log_summary(log_entries)

    prompt = _build_prompt(log_entries)
    rules  = _call_llm(prompt)
    valid  = [_normalise(r) for r in rules if _normalise(r) is not None]

    if valid:
        print(f"\n[Reflection Engine] {len(valid)} valid rule(s) from LLM:")
        for i, r in enumerate(valid, 1):
            print(f"  Rule {i}: [{r['trigger_feature']}] {r['rule']}")
    else:
        print("[Reflection Engine] LLM gave no usable rules — using deterministic fallback.")
        valid = _fallback_rules(log_entries)

    return valid


# ─────────────────────────────────────────────────────────────
#  Prompt construction
# ─────────────────────────────────────────────────────────────

def _print_log_summary(entries: list):
    deaths    = sum(1 for e in entries if e["event"] == "tile_death")
    auto_kill = sum(1 for e in entries if e["event"] == "auto_death")
    success   = sum(1 for e in entries if e["event"] == "success")
    print(f"  Log: {deaths} tile-death(s), {auto_kill} auto-death(s), {success} success(es).")


def _build_prompt(entries: list) -> str:
    lines = []
    for e in entries:
        ev   = e["event"]
        tile = e.get("tile_that_killed", "unknown")
        steps = e.get("total_steps", "?")
        pen   = e.get("total_penalty", "?")

        if ev == "tile_death":
            lines.append(
                f"- Stepped into '{tile}' and DIED instantly "
                f"(after {steps} steps, total penalty {pen})."
            )
        elif ev == "auto_death":
            hp = e.get("high_penalty_tile", tile)
            lines.append(
                f"- Was FORCE-KILLED after revisiting the same '{hp}' tile too many times "
                f"(cumulative penalty on that tile reached -60, happened after {steps} steps)."
            )
        elif ev == "success":
            lines.append(f"- Successfully reached the GREEN GOAL in {steps} steps.")

    experience = "\n".join(lines) or "No experience logged."

    # Use a very explicit prompt with a filled-in example so the LLM
    # knows EXACTLY what format to produce — especially forbidden_action.
    prompt = f"""\
You are a safety-rule generator for a grid-world navigation agent.
The agent explored a room and had these experiences:

{experience}

Game rules you MUST reflect in your output:
- Every step costs -3 penalty. Score starts at 0, decreases each step.
- Stepping onto 'red lava' deals -10 and kills the agent instantly.
- Revisiting the same tile too many times eventually force-kills the agent (-60 threshold).
- The green goal tile gives +10 and wins the game.
- The agent can only do: Turn Left, Turn Right, Move Forward.

Your task: output a JSON object with a "rules" array.
Each rule must have exactly these three fields:
  - "rule"             : a clear plain-English sentence explaining what to avoid and why
  - "forbidden_action" : MUST be exactly one of -> "Move Forward"  "Turn Left"  "Turn Right"
  - "trigger_feature"  : a short 1-3 word label for what the agent sees (e.g. "red lava")

EXAMPLE (use this exact structure):
{{
  "rules": [
    {{
      "rule": "Never move forward into red lava — it causes instant death.",
      "forbidden_action": "Move Forward",
      "trigger_feature": "red lava"
    }},
    {{
      "rule": "Avoid revisiting the same tile repeatedly — too many revisits force-kills the agent.",
      "forbidden_action": "Move Forward",
      "trigger_feature": "revisited tile"
    }}
  ]
}}

Now generate rules based on the agent's actual experiences above.
Output ONLY valid JSON — no explanation, no markdown fences, no extra text.
"""
    return prompt


# ─────────────────────────────────────────────────────────────
#  LLM call
# ─────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> list[dict]:
    """Try Ollama llama3.2:3b. Returns list of raw rule dicts."""
    try:
        import ollama
        print("  [LLM] Calling llama3.2:3b...")
        resp = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp["message"]["content"].strip()

        # Strip markdown fences if LLM wraps in ```json ... ```
        if "```" in raw:
            lines = [l for l in raw.splitlines() if not l.strip().startswith("```")]
            raw = "\n".join(lines).strip()

        # Find the JSON object (sometimes LLM adds preamble)
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]

        data  = json.loads(raw)
        rules = data.get("rules", [])
        print(f"  [LLM] Received {len(rules)} candidate rule(s).")
        return rules

    except json.JSONDecodeError as exc:
        print(f"  [LLM] JSON parse error: {exc}")
    except Exception as exc:
        print(f"  [LLM] Unavailable ({exc.__class__.__name__}: {exc}). Using fallback.")

    return []


# ─────────────────────────────────────────────────────────────
#  Normalise + validate a single rule dict
# ─────────────────────────────────────────────────────────────

def _normalise(r) -> dict | None:
    """Title-case forbidden_action and validate. Returns None if invalid."""
    if not isinstance(r, dict):
        return None
    if not r.get("rule") or not r.get("trigger_feature"):
        return None

    # Normalise capitalisation ("move forward" -> "Move Forward")
    raw_action  = str(r.get("forbidden_action", "")).strip()
    normalised  = " ".join(w.capitalize() for w in raw_action.split())
    if normalised not in _VALID_ACTIONS:
        # Try common LLM mistakes: "forward" -> "Move Forward"
        if "forward" in raw_action.lower():
            normalised = "Move Forward"
        elif "left" in raw_action.lower():
            normalised = "Turn Left"
        elif "right" in raw_action.lower():
            normalised = "Turn Right"
        else:
            print(f"  [Validation] Dropping rule — bad forbidden_action: {raw_action!r}")
            return None

    return {
        "rule":             str(r["rule"]).strip(),
        "forbidden_action": normalised,
        "trigger_feature":  str(r["trigger_feature"]).strip().lower(),
    }


# ─────────────────────────────────────────────────────────────
#  Deterministic fallback rules
# ─────────────────────────────────────────────────────────────

def _fallback_rules(entries: list) -> list[dict]:
    """Build basic rules from log facts without needing the LLM."""
    rules    = []
    seen_tiles = set()

    for e in entries:
        tile = e.get("tile_that_killed", "")
        ev   = e.get("event", "")

        if ev == "tile_death" and tile and tile not in seen_tiles:
            seen_tiles.add(tile)
            rules.append({
                "rule":             f"Never move forward into {tile} — it causes instant death.",
                "forbidden_action": "Move Forward",
                "trigger_feature":  tile.lower(),
            })

        if ev == "auto_death" and "auto_revisit" not in seen_tiles:
            seen_tiles.add("auto_revisit")
            hp = e.get("high_penalty_tile", "this tile")
            rules.append({
                "rule": (
                    f"Avoid staying on the same tile repeatedly — "
                    f"after enough revisits the game force-kills the agent. "
                    f"Always move on to a different tile."
                ),
                "forbidden_action": "Move Forward",
                "trigger_feature":  "revisited tile",
            })

    if not rules:
        rules.append({
            "rule":             "Never move forward into red lava — it causes instant death.",
            "forbidden_action": "Move Forward",
            "trigger_feature":  "red lava",
        })

    return rules


# ─────────────────────────────────────────────────────────────
#  Legacy shim
# ─────────────────────────────────────────────────────────────

def verify_rule(rule_data, trials=1) -> bool:
    if not isinstance(rule_data, dict):
        return False
    if rule_data.get("forbidden_action") not in _VALID_ACTIONS:
        return False
    if not rule_data.get("trigger_feature"):
        return False
    return True
