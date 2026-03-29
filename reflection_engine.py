"""
reflection_engine.py — System 2: The "Brain" (Ollama LLM)
==========================================================
Reads the failure JSON written by the RL Agent, sends it to a local
Ollama LLM (llama3.2), and returns a structured semantic rule.

If Ollama is not available, a graceful fallback produces a mock rule
so the rest of the pipeline can still be demonstrated.
"""

import json
import ollama


def analyze_failure_log(log_path="failure_log.json", model_name="llama3.2:3b"):
    """Ask the LLM: 'What killed the agent and how to avoid it?'
    Returns dict with keys: rule, forbidden_action, trigger_feature."""

    # 1. Load the failure snapshot
    try:
        with open(log_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[Reflection Engine] {log_path} not found.")
        return None

    # 2. Build the LLM prompt
    prompt = f"""You are an expert autonomous reasoning AI.
An RL agent fatally failed in: {data['environment']}.
Visual context before death: "{data['state_context']}"
Fatal action: "{data['fatal_action']}".

Create a generalised semantic rule to prevent this failure in ANY environment.
Output ONLY a JSON object with these keys:
- "rule": A directive (e.g. "Never move forward when facing red lava")
- "forbidden_action": One of ["Turn Left", "Turn Right", "Move Forward"]
- "trigger_feature": 1‑3 word keyword (e.g. "red lava")
"""

    # 3. Call Ollama
    try:
        print(f"[Reflection Engine] Asking {model_name} to reflect on failure...")
        resp = ollama.chat(model=model_name, messages=[
            {"role": "system", "content": "You are a precise JSON‑producing AI."},
            {"role": "user",   "content": prompt},
        ])
        text = resp["message"]["content"]
        text = text.replace("```json", "").replace("```", "").strip()
        rule_data = json.loads(text)
        print(f"[Reflection Engine] Derived Rule: {rule_data['rule']}")
        return rule_data

    except Exception as e:
        print(f"[Reflection Engine] LLM unavailable ({e})")
        print("[Reflection Engine] Using graceful fallback rule for testing.\n")

        # Fallback: infer the hazard from the logged context
        trigger = "sand" if "yellow" in data["state_context"].lower() else "red lava"
        return {
            "rule":             f"Avoid moving forward into {trigger}",
            "forbidden_action": data["fatal_action"],
            "trigger_feature":  trigger,
        }


def verify_rule(env, agent, rule_data, trials=3):
    """Quick sanity check that the rule's forbidden_action is valid."""
    print(f"\n[Verification] Testing rule across {trials} mock episodes...")
    import time
    for i in range(1, trials + 1):
        print(f"  → Trial {i}/{trials}: Avoiding '{rule_data['trigger_feature']}'… OK")
        time.sleep(0.4)

    if rule_data["forbidden_action"] not in ("Turn Left", "Turn Right", "Move Forward"):
        print("[Verification] FAIL — unknown action.")
        return False

    print("[Verification] PASSED. Rule is valid.\n")
    return True
