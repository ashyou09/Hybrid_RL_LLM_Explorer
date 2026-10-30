import pytest

from research import experiments as ex
from research.config import Config
from research.seeding import seed_everything
from research.train import train


@pytest.fixture(scope="module")
def trained():
    cfg = Config(train_episodes=300, eval_seeds=2, eval_episodes=10, experience_budget=3)
    agent, history = train(cfg, seed_everything(0))
    return cfg, agent, history


def test_build_skips_unavailable_shields_with_reason(trained, monkeypatch, tmp_path):
    cfg, _, history = trained
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    _, _, shields, skipped, _ = ex.build(cfg, history, ["none", "llm_prior"], "groq")
    assert "none" in shields and "GROQ_API_KEY" in skipped["llm_prior"]


def test_classify_oracle_is_perfect_and_lookup_misses_unseen(trained):
    cfg, _, history = trained
    _, _, shields, _, _ = ex.build(cfg, history, ["lookup", "oracle"], "fake")
    res = ex.classify(shields, cfg.deadly_categories)
    assert res["oracle"]["test_precision"] == 1.0 and res["oracle"]["test_recall"] == 1.0
    assert res["lookup"]["blocked_share"]["unseen hazards"] == 0.0
    assert "S6 oracle" in ex.classify_table(res)


def test_evaluate_oracle_never_dies(trained):
    cfg, agent, history = trained
    _, _, shields, _, _ = ex.build(cfg, history, ["none", "oracle"], "fake")
    res = ex.evaluate_shields(agent, cfg, shields)
    assert res["oracle"]["death_rate"]["mean"] == 0.0
    assert res["oracle"]["decoy_false_avoidance"] in (0.0, None)


def test_evaluation_is_paired_and_deterministic(trained):
    cfg, agent, history = trained
    runs = [ex.evaluate_shields(agent, cfg, ex.build(cfg, history, ["none"], "fake")[2])
            for _ in range(2)]
    assert runs[0]["none"]["success_rate"] == runs[1]["none"]["success_rate"]


def test_sweep_varies_one_knob(trained):
    cfg, agent, history = trained
    res = ex.sweep(agent, cfg, history, "hazard_count", [1, 3], ["none"], "fake")
    assert [p["value"] for p in res["points"]] == [1, 3]
