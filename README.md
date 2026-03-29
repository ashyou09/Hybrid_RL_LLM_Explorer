---
title: Hybrid RL LLM Explorer
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: RL agent dies → LLM distills rule → Explorer navigates safely
---

# 🧠 Hybrid RL → LLM → Explorer
### Experiment 1: Semantic Safety Rule Transfer via Vector DB

A research demonstration of **cross-agent knowledge transfer** — an RL agent learns through failure, an LLM distills the lesson into a semantic rule, and a separate Explorer agent uses that rule to navigate safely through unseen environments.

---

## 📐 Architecture

```
┌──────────────┐     dies     ┌──────────────────┐    rule     ┌───────────────┐
│  DQN Agent   │ ──────────► │  LLM (llama3.2)  │ ─────────► │   ChromaDB    │
│  (explores)  │             │  (reflects)       │            │  (Vector DB)  │
└──────────────┘             └──────────────────┘            └───────┬───────┘
                                                                      │ query
                                                             ┌────────▼───────┐
                                                             │  Rule-Guided   │
                                                             │  Explorer      │
                                                             └────────────────┘
```

| Agent | Role |
|---|---|
| **DQN (rl_core.py)** | Explores blindly, dies on hazards, logs failure context |
| **LLM (reflection_engine.py)** | Reads failure log, generates a semantic safety rule |
| **ChromaDB (memory_hub.py)** | Stores rules as vectors; queried by cosine similarity |
| **Explorer (planner_agent.py)** | Navigates using only inherited rules — no map, no DFS |

---

## 🗺️ Experiment Phases

| Phase | Room | Hazard | Goal |
|---|---|---|---|
| 1 | Lava Room | Red Lava | Learn "avoid red lava" |
| 2 | Sand Room | Yellow Sand | Learn "avoid sand" |
| 3 | Final Exam | Both hazards | Navigate unseen maze using both rules |

---

## 🚀 Run Locally (Full Experience with Ollama)

```bash
# 1. Clone
git clone https://huggingface.co/spaces/YOUR_USERNAME/hybrid-rl-llm-explorer
cd hybrid-rl-llm-explorer

# 2. Install dependencies
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Install & run Ollama (for real LLM rules)
# Download from https://ollama.com
ollama pull llama3.2:3b

# 4. Run with side-by-side pygame display
python run_experiment.py

# OR run the Gradio web interface locally
python app.py
```

> **On HF Spaces**: Ollama is not available, so the fallback rule engine activates automatically. The full pipeline still demonstrates correctly.

---

## 📁 File Structure

```
├── app.py                 # Gradio HF Space entry point
├── run_experiment.py      # Main orchestrator (pygame side-by-side display)
├── display.py             # Unified pygame display (game + log panel)
├── environments.py        # Custom MiniGrid rooms (Lava, Sand, Combined)
├── rl_core.py             # DQN Agent + observation parser
├── reflection_engine.py   # LLM reflection (Ollama / fallback)
├── memory_hub.py          # ChromaDB vector store
├── planner_agent.py       # Rule-guided explorer agent
└── requirements.txt
```

---

## 🔬 Research Significance

This experiment demonstrates:
1. **Online failure learning** — knowledge extracted from real-time deaths, not offline datasets
2. **Semantic transfer** — rules encoded as text vectors, not hardcoded reward shaping
3. **Zero-shot generalisation** — Explorer navigates the *combined* room having never seen it
4. **Human-like navigation** — Explorer uses no pathfinding algorithm; turns randomly unless rules forbid it

---

## 📜 License

MIT — feel free to extend, fork, and build on this.
