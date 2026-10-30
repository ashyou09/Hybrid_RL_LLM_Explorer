"""One interface to several LLM providers, with a disk cache.

    llm = LLM("groq")                  # or "openrouter", "claude", "ollama", "fake"
    text = llm.complete(prompt, system=...)

Keys come from environment variables (or a local, git-ignored ``.env`` file):
GROQ_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY. Ollama needs no key.

Every answer is cached in ``results/llm_cache/<provider>__<model>.json`` keyed by a
hash of (system, prompt), so re-running an experiment makes zero API calls and the
cache file documents exactly what the model said. Calls use temperature 0 where the
provider allows it.

The "fake" provider is a keyword matcher used only to test the pipeline offline.
Its outputs are never reported as LLM results.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

CACHE_DIR = Path("results/llm_cache")

PROVIDERS = {
    # name: (default model, env var with the key)
    "groq": ("llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "openrouter": ("meta-llama/llama-3.3-70b-instruct:free", "OPENROUTER_API_KEY"),
    "claude": ("claude-haiku-5-5", "ANTHROPIC_API_KEY"),   # cheapest Claude model (needs API credits)
    "ollama": ("llama3.2:3b", None),
    "fake": ("keyword-v1", None),
}

OPENAI_COMPATIBLE_URLS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}
OLLAMA_URL = "http://localhost:11434/api/chat"

# Claude models that reject the `effort` setting (older generation).
NO_EFFORT_MODELS = ("claude-haiku-4-5", "claude-sonnet-4-5")


class LLMUnavailable(RuntimeError):
    """Raised when a provider cannot be used (missing key, package or server)."""


def load_dotenv(path: str | Path = ".env") -> None:
    """Read KEY=VALUE lines from a local .env file into os.environ (existing vars win)."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip():
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a model reply (tolerates code fences and preamble)."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object in reply: {text[:200]!r}")
    return json.loads(text[start:end + 1])


class LLM:
    def __init__(self, provider: str = "groq", model: str | None = None,
                 cache_dir: Path = CACHE_DIR, timeout: float = 120.0):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
        load_dotenv()
        self.provider = provider
        self.model = model or PROVIDERS[provider][0]
        self.timeout = timeout
        self.api_calls = 0       # real requests sent
        self.cache_hits = 0
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model)
        self.cache_path = Path(cache_dir) / f"{provider}__{safe}.json"
        self._cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def available(self) -> tuple[bool, str]:
        env = PROVIDERS[self.provider][1]
        if env and not os.environ.get(env):
            return False, f"{env} is not set (see docs/setup.md)"
        return True, ""

    # ── public API ─────────────────────────────────────────────
    def complete(self, prompt: str, system: str = "") -> str:
        key = hashlib.sha256(f"{system}\n\x00\n{prompt}".encode()).hexdigest()
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]["response"]
        ok, why = self.available()
        if not ok:
            raise LLMUnavailable(f"{self.label}: {why}")
        text = getattr(self, f"_call_{self.provider}")(prompt, system)
        self.api_calls += 1
        self._cache[key] = {"system": system, "prompt": prompt, "response": text}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=1, sort_keys=True))
        return text

    # ── providers ──────────────────────────────────────────────
    def _post(self, url: str, body: dict, headers: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise LLMUnavailable(f"{self.label}: HTTP {e.code} {e.read()[:300]!r}") from e
        except urllib.error.URLError as e:
            raise LLMUnavailable(f"{self.label}: cannot reach {url} ({e.reason})") from e

    def _messages(self, prompt: str, system: str) -> list[dict]:
        return ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]

    def _call_openai_compatible(self, prompt: str, system: str) -> str:
        env = PROVIDERS[self.provider][1]
        data = self._post(OPENAI_COMPATIBLE_URLS[self.provider],
                          {"model": self.model, "temperature": 0,
                           "messages": self._messages(prompt, system)},
                          {"Authorization": f"Bearer {os.environ[env]}"})
        return data["choices"][0]["message"]["content"]

    _call_groq = _call_openai_compatible
    _call_openrouter = _call_openai_compatible

    def _call_ollama(self, prompt: str, system: str) -> str:
        data = self._post(OLLAMA_URL, {"model": self.model, "stream": False,
                                       "options": {"temperature": 0},
                                       "messages": self._messages(prompt, system)}, {})
        return data["message"]["content"]

    def _call_claude(self, prompt: str, system: str) -> str:
        try:
            import anthropic
        except ImportError as e:
            raise LLMUnavailable("pip install anthropic to use the claude provider") from e
        client = anthropic.Anthropic()
        try:
            # short yes/no judgements: low effort where the model supports it
            extra = {} if self.model in NO_EFFORT_MODELS else {"output_config": {"effort": "low"}}
            response = client.messages.create(
                model=self.model, max_tokens=16000, **extra,
                **({"system": system} if system else {}),
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIStatusError as e:
            raise LLMUnavailable(f"{self.label}: HTTP {e.status_code} {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMUnavailable(f"{self.label}: connection error") from e
        if response.stop_reason == "refusal":
            raise LLMUnavailable(f"{self.label}: request refused")
        return "".join(b.text for b in response.content if b.type == "text")

    def _call_fake(self, prompt: str, system: str) -> str:
        return fake_reply(prompt)


# ── offline stand-in ───────────────────────────────────────────
_HOT_WORDS = ("lava", "fire", "flame", "burn", "hot", "molten", "glow", "coal", "heat")


def fake_reply(prompt: str) -> str:
    """Keyword stand-in that answers the reflection and judge prompts in the right
    format. It only checks the plumbing; it has no understanding."""
    if "TILE TO JUDGE:" in prompt:
        tile = prompt.split("TILE TO JUDGE:", 1)[1].splitlines()[0].lower()
        hot = any(w in tile for w in _HOT_WORDS)
        return json.dumps({"dangerous": hot, "rule": "R1" if hot else None,
                           "reason": "keyword match" if hot else "no danger keyword"})
    died = re.findall(r"\[(E\d+)\] DIED after stepping on", prompt)
    return json.dumps({"rules": [{
        "id": "R1", "if": "the tile is hot, burning or glowing",
        "then": "do not move forward onto it", "confidence": 0.9, "evidence": died,
    }]} if died else {"rules": []})
