# Setup Guide

Nothing heavy needs to run on your laptop. Pick one place to run:

| Where | Good for | Notes |
|---|---|---|
| **Claude cloud session** | Building and running experiments together | API keys go in the environment settings |
| **Google Colab** | Running experiments yourself | Free; set keys with Colab "Secrets" |
| **Hugging Face Space** | Showing the Lab panel to anyone | Already linked to this repo's `main` branch |
| Laptop | Optional | Only a browser is required |

## API keys

Keys are **never** written into code, committed, or pasted into chat. The code reads
them from environment variables.

| Variable | Provider | Needed from |
|---|---|---|
| `GROQ_API_KEY` | **Groq** (free; open-source Llama 3.3 70B) — **main**, used for the Level 1 runs | Level 1 |
| `OPENROUTER_API_KEY` | OpenRouter (free models) — optional | Level 4 comparison |
| `ANTHROPIC_API_KEY` | Claude Haiku 5.5 — optional; needs **API credits** (a Claude Pro subscription does not include them) | Level 4 comparison |

### Claude cloud session

Open the environment menu in the session title bar → **Edit** → add the key under
**Network secrets** (or **API credentials** / environment variables, depending on the
app version) with the variable name above. Start a new session to pick it up.

### Google Colab

Left sidebar → 🔑 **Secrets** → add `ANTHROPIC_API_KEY`, then in the notebook:

```python
import os
from google.colab import userdata
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
```

### Local (optional)

Create a `.env` file (already ignored by git) or export the variable in your shell.

## Installing

v0 demo:

```bash
pip install -r requirements.txt
```

Research code and the Lab (no PyTorch, no local LLM):

```bash
pip install -r requirements-research.txt
```

```bash
python -m research --help
```

```bash
streamlit run lab.py
```

Optional: `pip install anthropic` (Claude provider), `pip install sentence-transformers`
(embedding shield S2; downloads a model from huggingface.co).

### Network access

The LLM providers must be reachable from where you run:
`api.groq.com` (Groq), `openrouter.ai` (OpenRouter), `api.anthropic.com` (Claude),
`huggingface.co` (embedding model). On a laptop or Colab this is normally fine. In a
Claude cloud session, add the hosts under the environment's **Network access → Allowed
domains** (environment menu in the session title bar → **Edit**).

## Hugging Face Space deployment

Every push to `main` runs `.github/workflows/sync_hf.yml`, which deploys **the Lab**
(`lab.py`, `research/`, `configs/`, with `requirements-research.txt` installed as the
Space's `requirements.txt`) to `huggingface.co/spaces/ashyou09/hybrid-rl-llm-explorer`.
The Space is a Streamlit Space (`sdk: streamlit`, `app_file: lab.py` in the README header).

It needs one GitHub secret, `HF_TOKEN` (a Hugging Face token of type **Write**), under the
repo's **Settings → Secrets and variables → Actions**.

### LLM shields on the Space

Add the Groq key as a **Space secret**: Space **Settings → Variables and secrets →
New secret**, name `GROQ_API_KEY` (free key from console.groq.com). Without it, the Lab
still works and skips the LLM shields with a clear message. The Space can reach Groq
even though the Claude cloud session cannot.

To use Claude instead, add `ANTHROPIC_API_KEY` and pick Provider = claude. That needs
API credits in the Anthropic console (a Claude Pro subscription does not include them),
and a low monthly spend limit, because a public Space lets anyone use the key.

**Getting results from the Space:** in the Lab's *Compare shields* tab, set 6000 training
episodes, 10 seeds and 100 episodes per seed, run, and use **Download results (JSON)**.

Research results (images, GIFs) are not deployed, because the Space rejects binary files
that are not in Git LFS. The v0 Gradio demo (`app.py`) is no longer deployed; it runs locally.
