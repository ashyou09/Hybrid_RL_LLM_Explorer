import numpy as np

from research.config import Config
from research.env import FORWARD
from research.qlearning import QLearner, state_of
from research.seeding import seed_everything
from research.train import bootstrap_ci, evaluate, train

VIEW = {"front": "red lava", "left": "wall", "right": "empty floor", "goal_dir": "ahead"}


def test_state_has_no_coordinates_and_keeps_front_name():
    s = state_of(VIEW, FORWARD)
    assert s.startswith("red lava|wall|open|ahead|F")


def test_mask_is_never_violated():
    agent = QLearner(rng=np.random.default_rng(0))
    agent.q[state_of(VIEW)] = np.array([0.0, 0.0, 99.0])   # loves moving forward
    mask = [True, True, False]
    for eps in (0.0, 1.0):
        assert all(agent.act(state_of(VIEW), eps, mask) != FORWARD for _ in range(200))


def test_update_moves_towards_reward():
    agent = QLearner(alpha=0.5)
    agent.update("s", FORWARD, -10.0, "s2", done=True)
    assert agent.values("s")[FORWARD] == -5.0


def test_save_load_roundtrip(tmp_path):
    agent = QLearner()
    agent.update("s", 1, 2.0, "s2", done=True)
    agent.save(tmp_path / "q.json")
    loaded = QLearner.load(tmp_path / "q.json")
    assert np.allclose(loaded.values("s"), agent.values("s"))


def test_learns_to_reach_goal_without_hazards():
    cfg = Config(hazard_count=0, decoy_count=0, train_episodes=500,
                 eval_seeds=2, eval_episodes=50)
    agent, history = train(cfg, seed_everything(0))
    m = evaluate(agent, cfg, "train")
    assert m["success_rate"]["mean"] > 0.9
    assert len(history) == 500


def test_bootstrap_ci_brackets_mean():
    vals = np.array([0.5, 0.6, 0.7, 0.8])
    lo, hi = bootstrap_ci(vals)
    assert lo <= vals.mean() <= hi
