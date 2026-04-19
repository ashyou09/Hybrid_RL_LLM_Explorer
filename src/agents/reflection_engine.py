"""Reflection logic for turning failures into simple safety rules."""

import json


def _extract_precise_trigger(state_context: str, fatal_action: str) -> str | None:
    """Extract a concrete object label from parser output text."""
    if not state_context:
        return None

    # Expected format from rl_core.parse_local_observation():
    # "Front: X. Left: Y. Right: Z."
    parts = {}
    for chunk in state_context.split("."):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("Front:"):
            parts["Front"] = chunk.replace("Front:", "").strip()
        elif chunk.startswith("Left:"):
            parts["Left"] = chunk.replace("Left:", "").strip()
        elif chunk.startswith("Right:"):
            parts["Right"] = chunk.replace("Right:", "").strip()

    def _is_concrete(label: str | None) -> bool:
        if not label:
            return False
        low = label.lower()
        return low not in {"wall", "empty space", "unknown"}

    if fatal_action == "Move Forward" and _is_concrete(parts.get("Front")):
        return parts["Front"]

    # Fallback: pick the first concrete thing we saw.
    for k in ("Front", "Left", "Right"):
        if _is_concrete(parts.get(k)):
            return parts[k]
    return None


def analyze_failure_log(log_path="failure_log.json"):
    """Build a rule from failure_log.json with deterministic trigger extraction."""
    try:
        with open(log_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[Reflection Engine] {log_path} not found.")
        return None

    trigger = _extract_precise_trigger(data.get("state_context", ""), data.get("fatal_action", ""))
    if not trigger:
        # Last-resort compatibility fallback (rare for malformed logs).
        trigger = "sand" if "sand" in data.get("state_context", "").lower() else "red lava"

    return {
        "rule":             f"Avoid moving forward into {trigger}",
        "forbidden_action": data.get("fatal_action", "Move Forward"),
        "trigger_feature":  trigger,
    }


def verify_rule(rule_data, trials=1):
    """Quick sanity check for generated rule shape and action value."""
    print(f"\n[Verification] Testing rule across {trials} mock episodes...")
    import time
    for i in range(1, trials + 1):
        print(f"  → Trial {i}/{trials}: Avoiding '{rule_data['trigger_feature']}'… OK")
        time.sleep(0.4)

    if not isinstance(rule_data, dict):
        print("[Verification] FAIL — rule is not a dict.")
        return False
    if rule_data.get("forbidden_action") not in ("Turn Left", "Turn Right", "Move Forward"):
        print("[Verification] FAIL — unknown action.")
        return False
    if not rule_data.get("trigger_feature"):
        print("[Verification] FAIL — missing trigger_feature.")
        return False

    print("[Verification] PASSED. Rule is valid.\n")
    return True


# ──────────────────────────────────────────────────────────────────
# Periodic LLM Navigation Consultant
# ──────────────────────────────────────────────────────────────────

def get_navigation_hint(payload: dict) -> dict | None:
    """Send a trajectory payload to llama3.2:3b and request one navigation
    rule. Returns a rule dict compatible with MemoryHub.store_verified_rule(),
    or None if the LLM returns no useful hint.

    Falls back to a deterministic heuristic when Ollama is unavailable.

    Payload schema:
        {
          "environment": str,
          "total_steps": int,
          "unique_tiles_visited": int,
          "accumulated_revisit_penalty": float,
          "top_revisited_cells": list[tuple],
          "trajectory": list[str],   # human-readable step lines
        }
    """

    env_name   = payload.get("environment", "unknown env")
    steps      = payload.get("total_steps", 0)
    unique     = payload.get("unique_tiles_visited", 0)
    penalty    = payload.get("accumulated_revisit_penalty", 0.0)
    top_rev    = payload.get("top_revisited_cells", [])
    traj_lines = payload.get("trajectory", [])

    traj_summary = "\n".join(traj_lines[-10:]) or "no steps recorded"
    top_rev_str  = ", ".join(f"pos{p}×{c}" for p, c in top_rev) or "none"

    prompt = (
        "You are a navigation advisor for an autonomous grid-world agent.\n"
        f"Environment: {env_name}\n"
        f"Steps taken: {steps}, Unique tiles visited: {unique}\n"
        f"Revisit penalty accumulated: {penalty:.1f} "
        "(each revisit past step 10 costs −1 × revisit_count)\n"
        f"Most revisited cells: {top_rev_str}\n\n"
        "Recent trajectory (step: pos, front tile, action taken, revisit?):\n"
        f"{traj_summary}\n\n"
        "Based on the trajectory and penalty, give ONE concise navigation rule "
        "to help the agent escape loops and explore more efficiently.\n"
        "Output ONLY a valid JSON object with exactly these keys: "
        "rule, forbidden_action, trigger_feature.\n"
        "forbidden_action must be one of: Turn Left, Turn Right, Move Forward.\n"
        "trigger_feature must be a short 1-3 word label the agent might see "
        "(e.g. 'revisited tile', 'empty space', 'red lava')."
    )

    # ── Try Ollama first ────────────────────────────────────────
    try:
        import ollama
        print(f"  [LLM] Asking llama3.2 for navigation hint (step {steps})…")
        resp = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp["message"]["content"].strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = "\n".join(
                line for line in raw.splitlines()
                if not line.strip().startswith("```")
            ).strip()

        data = json.loads(raw)
        rule  = data.get("rule", "").strip()
        action = data.get("forbidden_action", "").strip()
        trigger = data.get("trigger_feature", "").strip()

        if rule and action and trigger:
            print(f"  [LLM] Navigation hint: {rule}")
            return {
                "rule":             rule,
                "forbidden_action": action,
                "trigger_feature":  trigger,
            }

        print("  [LLM] Response missing required fields — using fallback.")

    except Exception as exc:
        print(f"  [LLM] Ollama unavailable ({exc.__class__.__name__}). Using fallback.")

    # ── Deterministic fallback ──────────────────────────────────
    # If the agent has accumulated revisit penalty, suggest avoiding revisits.
    if penalty < -2.0 or (top_rev and top_rev[0][1] >= 3):
        return {
            "rule":             "Avoid moving forward into already-explored tiles to reduce looping",
            "forbidden_action": "Move Forward",
            "trigger_feature":  "revisited tile",
        }

    # Generic: encourage turning when stuck
    return None


# ──────────────────────────────────────────────────────────────────
# Multi-Death Episodic Analysis: Pattern Recognition Across Failed Attempts
# ──────────────────────────────────────────────────────────────────

def analyze_multi_death_pattern(payload: dict) -> dict | None:
    """Analyze pathfinding failures across multiple episodes.
    
    Called when agent dies 3+ times in succession in Phase 3.
    Sends cross-episode failure patterns to LLM for meta-strategic guidance.
    
    Payload schema:
        {
          "environment": str,
          "analysis_type": "multi_death_episodic",
          "death_count": int,
          "death_summaries": list[str],   # human-readable death descriptions
          "all_death_records": list[dict], # detailed data from each death
        }
    
    Returns a strategic navigation rule (dict) or None if LLM unavailable.
    """
    
    env_name = payload.get("environment", "unknown env")
    death_count = payload.get("death_count", 0)
    death_summaries = payload.get("death_summaries", [])
    all_records = payload.get("all_death_records", [])
    
    # Build a comprehensive death pattern analysis
    summaries_text = "\n".join(f"  {s}" for s in death_summaries) or "no deaths recorded"
    
    # Extract common failure locations
    failure_locs = [r.get("failure_location") for r in all_records]
    top_failures = {}
    for loc in failure_locs:
        top_failures[loc] = top_failures.get(loc, 0) + 1
    
    failure_pattern = ", ".join(
        f"pos{loc}×{count}" for loc, count in sorted(
            top_failures.items(), key=lambda kv: kv[1], reverse=True
        )[:3]
    ) or "scattered"
    
    # Average penalty trend (worsening or improving?)
    penalties = [r.get("revisit_penalty", 0.0) for r in all_records]
    penalty_trend = "worsening" if penalties and penalties[-1] < penalties[0] else "stable/improving"
    
    prompt = (
        "You are a strategic navigation advisor analyzing REPEATED FAILURES.\n"
        f"Environment: {env_name}\n"
        f"Number of consecutive deaths: {death_count}\n\n"
        "Death sequence:\n"
        f"{summaries_text}\n\n"
        f"Most common failure locations: {failure_pattern}\n"
        f"Penalty trend: {penalty_trend}\n\n"
        "Analyze the pattern across these failures. What structural or strategic "
        "lesson can the agent learn? For example:\n"
        "  - Is there a bottleneck or dead end being repeatedly hit?\n"
        "  - Are revisits getting worse, indicating convergence to local minima?\n"
        "  - Should the agent avoid certain tile types or patterns?\n\n"
        "Output ONLY a valid JSON object with exactly these keys: "
        "rule, forbidden_action, trigger_feature.\n"
        "The rule should describe a STRATEGIC PATTERN to avoid, not just a single tile.\n"
        "For example: rule='Avoid moving toward cluster of revisited tiles',\n"
        "             trigger_feature='high_revisit_density'.\n"
        "forbidden_action must be one of: Turn Left, Turn Right, Move Forward."
    )
    
    # ── Try Ollama for multi-death analysis ──────────────────────
    try:
        import ollama
        print(f"  [Multi-Death LLM] Analyzing {death_count} deaths for strategic pattern…")
        resp = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp["message"]["content"].strip()
        
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = "\n".join(
                line for line in raw.splitlines()
                if not line.strip().startswith("```")
            ).strip()
        
        data = json.loads(raw)
        rule = data.get("rule", "").strip()
        action = data.get("forbidden_action", "").strip()
        trigger = data.get("trigger_feature", "").strip()
        
        if rule and action and trigger:
            print(f"  [Multi-Death Strategy] {rule}")
            return {
                "rule": rule,
                "forbidden_action": action,
                "trigger_feature": trigger,
            }
        
        print("  [Multi-Death LLM] Response incomplete — using fallback.")
    
    except Exception as exc:
        print(f"  [Multi-Death] Ollama unavailable ({exc.__class__.__name__}). Using fallback.")
    
    # ── Deterministic fallback heuristic ──────────────────────────
    # If failures cluster in same region, suggest avoiding that area
    if len(top_failures) <= 2:
        # Very few failure locations → likely hitting same obstacle repeatedly
        return {
            "rule": "Failure cluster detected — try alternative routes by preferring unvisited directions",
            "forbidden_action": "Move Forward",
            "trigger_feature": "failure_hotspot",
        }
    
    # If revisit penalty worsening, focus on breaking loops
    if penalty_trend == "worsening":
        return {
            "rule": "Looping behavior intensifying — avoid revisiting tiles at all costs",
            "forbidden_action": "Move Forward",
            "trigger_feature": "loop_escalation",
        }
    
    # Generic multi-death guidance
    return {
        "rule": "Multiple failures detected — prioritize reaching new, unexplored regions",
        "forbidden_action": "Move Forward",
        "trigger_feature": "repeated_failure",
    }

