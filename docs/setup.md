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
| `ANTHROPIC_API_KEY` | **Claude Haiku 5.5** (cheapest Claude model) — **main**, used for the Level 1 runs | Level 1 |
| `GROQ_API_KEY` | Groq (free tier, open Llama models) — optional | Level 4 comparison |
| `OPENROUTER_API_KEY` | OpenRouter (free models) — optional | Level 4 comparison |

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

To let visitors run the LLM shields on the Space, add the Claude key as a **Space secret**:
Space **Settings → Variables and secrets → New secret**, name `ANTHROPIC_API_KEY`.
Without it, the Lab still works and skips the LLM shields with a clear message.

⚠️ A public Space lets anyone spend your key. Set a low monthly **spend limit** in the
Anthropic console first. With Claude Haiku 5.5 each judgement costs a tiny fraction of a cent.

Research results (images, GIFs) are not deployed, because the Space rejects binary files
that are not in Git LFS. The v0 Gradio demo (`app.py`) is no longer deployed; it runs locally.
