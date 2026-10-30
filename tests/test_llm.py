import json
import os

import pytest

from research.llm import LLM, LLMUnavailable, extract_json, load_dotenv


def test_cache_makes_second_run_free(tmp_path):
    llm = LLM("fake", cache_dir=tmp_path)
    first = llm.complete("TILE TO JUDGE: molten rock")
    again = LLM("fake", cache_dir=tmp_path)            # fresh object, same cache file
    assert again.complete("TILE TO JUDGE: molten rock") == first
    assert (llm.api_calls, again.api_calls, again.cache_hits) == (1, 0, 1)
    saved = json.loads(llm.cache_path.read_text())
    assert next(iter(saved.values()))["prompt"] == "TILE TO JUDGE: molten rock"


def test_missing_key_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)                          # no .env here
    llm = LLM("groq", cache_dir=tmp_path)
    assert not llm.available()[0]
    with pytest.raises(LLMUnavailable, match="GROQ_API_KEY"):
        llm.complete("hello")


def test_dotenv_loads_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nFOO_TEST_KEY=from_file\nBAR_TEST_KEY=\nBAZ_TEST_KEY='x'\n")
    monkeypatch.setenv("BAZ_TEST_KEY", "from_env")
    monkeypatch.delenv("FOO_TEST_KEY", raising=False)
    load_dotenv(env)
    assert os.environ["FOO_TEST_KEY"] == "from_file"
    assert "BAR_TEST_KEY" not in os.environ
    assert os.environ["BAZ_TEST_KEY"] == "from_env"
    monkeypatch.delenv("FOO_TEST_KEY")


def test_extract_json_tolerates_fences_and_preamble():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_claude_request_shape(tmp_path, monkeypatch):
    """Haiku 4.5 gets no effort setting; newer models get effort=low."""
    anthropic = pytest.importorskip("anthropic")
    sent = {}

    class Block:
        type, text = "text", '{"rules": []}'

    class Resp:
        stop_reason, content = "end_turn", [Block()]

    class Messages:
        def create(self, **kw):
            sent.clear()
            sent.update(kw)
            return Resp()

    class Client:
        messages = Messages()

    monkeypatch.setattr(anthropic, "Anthropic", lambda: Client())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    LLM("claude", cache_dir=tmp_path).complete("hi", system="sys")
    assert sent["model"] == "claude-haiku-4-5" and "output_config" not in sent
    assert "temperature" not in sent and sent["system"] == "sys"
    LLM("claude", model="claude-haiku-5-5", cache_dir=tmp_path).complete("hi2")
    assert sent["output_config"] == {"effort": "low"}
