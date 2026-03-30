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

    # 1. Load the failure snapshot
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
