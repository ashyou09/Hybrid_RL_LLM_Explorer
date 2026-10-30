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
| `GROQ_API_KEY` | Groq (free tier, open Llama models) — main | Level 1, step L1.7 |
| `OPENROUTER_API_KEY` | OpenRouter (free models) — backup | optional |
| `ANTHROPIC_API_KEY` | Claude — reference model | Level 4 |

### Claude cloud session

Open the environment menu in the session title bar → **Edit** → add the key under
**Network secrets** (or **API credentials** / environment variables, depending on the
app version) with the variable name above. Start a new session to pick it up.

### Google Colab

Left sidebar → 🔑 **Secrets** → add `GROQ_API_KEY`, then in the notebook:

```python
import os
from google.colab import userdata
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
```

### Local (optional)

Create a `.env` file (already ignored by git) or export the variable in your shell.

## Installing

v0 demo:

```bash
pip install -r requirements.txt
```

Research code: a separate, lighter `requirements-research.txt` arrives with step L1.1.
The embedding comparison (PyTorch, CPU) will be an optional extra.
