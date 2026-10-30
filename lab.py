"""The Lab — watch the agents and turn the knobs (Level 1, step L1.13).

    streamlit run lab.py

1. Train the learner (Agent A) with the knobs on the left.
2. Watch one episode, with the shield's decisions written under every frame.
3. Compare shields on the same test worlds.
4. Sweep one knob and see what it changes.

Runs here are small and quick for exploring; the numbers in the write-ups come from
the full command-line runs (python -m research ...).
"""

from __future__ import annotations

import io
import json

import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")

from research import experience, experiments as ex, rules as rules_mod  # noqa: E402
from research.config import Config  # noqa: E402
from research.env import TextGridEnv  # noqa: E402
from research.llm import PROVIDERS, LLM  # noqa: E402
from research.qlearning import AgentPolicy  # noqa: E402
from research.replay import record_episode  # noqa: E402
from research.seeding import seed_everything  # noqa: E402
from research.shields import LABELS, SHIELDS  # noqa: E402
from research.train import plot_curve, train  # noqa: E402

st.set_page_config(page_title="Semantic Osmosis Lab", page_icon="🧪", layout="wide")


# ── knobs ──────────────────────────────────────────────────────
def knobs() -> tuple[Config, str, str | None]:
    sb = st.sidebar
    sb.header("🎛️ Knobs")
    sb.caption("World")
    grid_size = sb.select_slider("Room size", [5, 7, 9], value=7)
    hazard_count = sb.slider("Hot tiles per room (difficulty)", 0, 8, 3)
    decoy_count = sb.slider("Harmless tiles per room (decoys in tests)", 0, 6, 2)
    sb.caption("Learner (Agent A)")
    train_episodes = sb.select_slider("Training episodes", [500, 1000, 2000, 4000, 6000], value=2000)
    experience_budget = sb.slider("Deaths before the LLM reflects", 1, 10, 5)
    sb.caption("Evaluation")
    eval_seeds = sb.slider("Seeds", 1, 10, 3)
    eval_episodes = sb.select_slider("Episodes per seed", [20, 50, 100], value=50)
    sb.caption("Brain (LLM)")
    provider = sb.selectbox("Provider", list(PROVIDERS), index=list(PROVIDERS).index("groq"))
    model = sb.text_input("Model (blank = default)", "") or None
    threshold = sb.slider("Embedding strictness", 0.3, 0.95, 0.65, 0.05)
    cfg = Config(name="lab", grid_size=grid_size, hazard_count=hazard_count,
                 decoy_count=decoy_count, train_episodes=train_episodes,
                 experience_budget=experience_budget, eval_seeds=eval_seeds,
                 eval_episodes=eval_episodes, embed_threshold=threshold)
    ok, why = LLM(provider, model).available()
    if provider == "fake":
        sb.warning("Fake = keyword matcher for testing only, not a real LLM.")
    elif not ok:
        sb.info(f"LLM shields unavailable: {why}")
    return cfg, provider, model


@st.cache_resource(show_spinner=False)
def trained(train_key: str):
    cfg = Config(**json.loads(train_key))
    return train(cfg, seed_everything(cfg.seed))


def train_key(cfg: Config) -> str:
    keep = ("grid_size", "hazard_count", "decoy_count", "train_episodes", "seed", "max_steps")
    return json.dumps({k: getattr(cfg, k) for k in keep}, sort_keys=True)


def gif_bytes(frames) -> bytes:
    buf = io.BytesIO()
    durations = [500] * (len(frames) - 1) + [3000]
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0)
    return buf.getvalue()


def show_skipped(skipped: dict) -> None:
    for name, why in skipped.items():
        st.warning(f"{LABELS[name]} skipped — {why}")


# ── page ───────────────────────────────────────────────────────
st.title("🧪 Semantic Osmosis Lab")
st.write("Agent A learns from its deaths; an LLM turns them into rules; a new agent uses "
         "them on hot tiles it has **never seen** — without fearing harmless look-alikes.")

with st.expander("ℹ️ What is this experiment? (read first)", expanded=False):
    st.markdown("""
**Question.** Agent A learns by trial and error (Q-learning) in rooms with a few *hot*
tiles (red lava, campfire, hot coals). After 5 deaths, an LLM reads what happened and
writes rules. A **new agent** then plays rooms with hot tiles **nobody has seen**
(molten rock, burning logs, white-hot metal) and harmless **look-alikes** (red carpet,
a painting of a fire). Can the rules protect it without making it afraid of the decoys?

**A shield** is a safety guard: before the agent steps forward, it decides
"is that tile deadly?" and, if yes, stops the step. We compare six guards:

| Shield | In plain words |
|---|---|
| S0 · No help | Never blocks. The agent is on its own. |
| S1 · Memory of past killers | Blocks only the exact tiles that killed the learner ("red lava"). |
| S2 · Similar-name check | Blocks tiles whose *name* looks like a killer's name. |
| S3 · AI common sense only | Asks the AI, without telling it what happened. |
| S4 · AI + experience (**ours**) | Asks the AI and gives it the learner's experience. |
| S6 · Perfect knowledge (cheat) | Knows the true answer. Shows the best score possible. |

**Words used here**

| Word | Meaning |
|---|---|
| Known danger | A hot tile the learner met (red lava, campfire, hot coals) |
| **New danger** | A hot tile nobody has seen (molten rock, burning logs, white-hot metal) |
| **Look-alike** | Harmless tile that looks dangerous (red carpet, painting of a fire) |
| False alarm | Avoiding a harmless look-alike: being scared for no reason |
| **Hallucination** | The AI says something false with confidence, e.g. "red carpet is deadly" (false alarm) or "molten rock is safe" (missed danger) |
| Got stuck | Ran out of time (100 steps) without reaching the goal |
| Seed | A fixed random setting, so a test can be repeated exactly |

**Measured so far** (10 seeds × 100 episodes, rooms with new dangers): no help → 41% die ·
memory of past killers → 41% die (no help on new tiles) · perfect knowledge → 0% die,
93% reach the goal. AI results: pending.

**How to use:** set the knobs on the left → tab 1 *Train* → tab 2 *Watch* an episode →
tab 3 *Compare* shields → tab 4 *Sweep* a knob. LLM shields use **Groq** (free; needs `GROQ_API_KEY`); the
*fake* provider is a keyword matcher for testing only.
""")

cfg, provider, model = knobs()
tab_train, tab_watch, tab_compare, tab_sweep = st.tabs(
    ["1 · Train the learner", "2 · Watch an episode", "3 · Compare shields", "4 · Sweep a knob"])

with tab_train:
    if st.button("Train Agent A", type="primary"):
        with st.spinner(f"Training for {cfg.train_episodes} episodes…"):
            st.session_state["train_key"] = train_key(cfg)
            trained(st.session_state["train_key"])
    if "train_key" not in st.session_state:
        st.info("Press **Train Agent A** to start.")
    else:
        agent, history = trained(st.session_state["train_key"])
        recent = history[-200:]
        c1, c2, c3 = st.columns(3)
        c1.metric("Reached goal (last 200)", f"{sum(h['outcome'] == 'goal' for h in recent) / len(recent):.0%}")
        c2.metric("Died (last 200)", f"{sum(h['outcome'] == 'died' for h in recent) / len(recent):.0%}")
        c3.metric("Situations learned", len(agent.q))
        buf = io.BytesIO()
        plot_curve(history, buf)
        st.image(buf.getvalue(), caption="Learning curve in training worlds")
        log = experience.build_log(history, cfg.experience_budget)
        st.subheader("What the LLM will read")
        st.code(experience.to_text(log), language=None)
        if st.button("Ask the LLM for rules"):
            try:
                rules = rules_mod.reflect(log, LLM(provider, model))
                st.code(rules_mod.to_text(rules), language=None)
            except Exception as e:  # shown to the user, never hidden
                st.error(str(e))

needs_training = "train_key" not in st.session_state

with tab_watch:
    if needs_training:
        st.info("Train Agent A first (tab 1).")
    else:
        agent, history = trained(st.session_state["train_key"])
        c1, c2, c3 = st.columns(3)
        shield_name = c1.selectbox("Shield", SHIELDS, format_func=LABELS.get, key="watch_shield")
        split = c2.selectbox("World", ["test", "train"], format_func=lambda s:
                             "test (never-seen tiles)" if s == "test" else "training")
        seed = c3.number_input("Layout seed", 0, 10_000, 1)
        if st.button("▶ Play episode", type="primary"):
            _, _, shields, skipped, llm = ex.build(cfg, history, [shield_name], provider, model)
            show_skipped(skipped)
            if shield_name in ("llm_prior", "llm_rules") and shields:
                st.caption(f"LLM used: `{llm.label}`")
            if shields:
                frames, summary = record_episode(
                    TextGridEnv(cfg, split=split),
                    AgentPolicy(agent, mask_fn=shields[shield_name].mask), int(seed))
                st.image(gif_bytes(frames), caption=f"Outcome: {summary['outcome']} after "
                                                    f"{summary['steps']} steps")

with tab_compare:
    if needs_training:
        st.info("Train Agent A first (tab 1).")
    else:
        agent, history = trained(st.session_state["train_key"])
        picked = st.multiselect("Shields", SHIELDS,
                                default=["none", "lookup", "llm_prior", "llm_rules", "oracle"],
                                format_func=LABELS.get)
        st.caption("For results comparable to the write-ups use: 6000 training episodes, "
                   "10 seeds, 100 episodes per seed (slower).")
        if st.button("Compare", type="primary"):
            with st.spinner("Evaluating on the same test worlds…"):
                _, rules, shields, skipped, llm = ex.build(cfg, history, picked, provider, model)
                show_skipped(skipped)
                res = ex.evaluate_shields(agent, cfg, shields)
            if any(n in shields for n in ("llm_prior", "llm_rules")):
                st.caption(f"LLM used: `{llm.label}`")
            if res:
                st.markdown(ex.evaluate_table(res))
                buf = io.BytesIO()
                ex.plot_evaluation(res, buf)
                st.image(buf.getvalue())
                if rules:
                    st.code(rules_mod.to_text(rules), language=None)
                report = {"config": cfg.to_dict(), "llm": llm.label,
                          "rules": rules, "skipped": skipped, "results": res}
                st.download_button("⬇️ Download results (JSON)",
                                   json.dumps(report, indent=2, default=str),
                                   file_name="lab_compare_results.json", mime="application/json")

with tab_sweep:
    if needs_training:
        st.info("Train Agent A first (tab 1).")
    else:
        agent, history = trained(st.session_state["train_key"])
        c1, c2 = st.columns(2)
        knob = c1.selectbox("Knob", ["hazard_count", "decoy_count", "experience_budget",
                                     "embed_threshold", "grid_size"])
        defaults = {"hazard_count": "1,2,4,6", "decoy_count": "0,2,4,6",
                    "experience_budget": "1,3,5,10", "embed_threshold": "0.5,0.65,0.8",
                    "grid_size": "5,7,9"}
        values = c2.text_input("Values", defaults[knob], key=f"values_{knob}")
        picked = st.multiselect("Shields", SHIELDS, default=["none", "oracle"],
                                format_func=LABELS.get, key="sweep_shields")
        if st.button("Run sweep", type="primary"):
            import yaml
            vals = [yaml.safe_load(v) for v in values.split(",") if v.strip()]
            with st.spinner(f"Evaluating {len(vals)} values × {len(picked)} shields…"):
                res = ex.sweep(agent, cfg, history, knob, vals, picked, provider, model)
            show_skipped(res["skipped"])
            buf = io.BytesIO()
            ex.plot_sweep(res, buf)
            st.image(buf.getvalue())
            st.dataframe(pd.DataFrame([{knob: p["value"], "shield": LABELS[n], "death": r["death"],
                                        "success": r["success"]}
                                       for p in res["points"] for n, r in p["results"].items()]))

st.sidebar.divider()
st.sidebar.caption("Full experiments: `python -m research --help` · Plan: future_plan/future.md")
