import pytest

from research.config import Config
from research.env import FORWARD
from research.llm import LLM
from research.rules import reflect, validate
from research.shields import make_shield

LOG = [
    {"id": "E1", "event": "died_on", "tile": "red lava", "description": "a pool of glowing red lava", "count": 2},
    {"id": "E2", "event": "died_on", "tile": "glowing hot coals", "description": "a bed of glowing hot coals", "count": 1},
    {"id": "E3", "event": "walked_safely_on", "tile": "grass", "description": "a patch of short green grass", "count": 9},
]
CFG = Config()


def view(front):
    return {"front": front, "left": "wall", "right": "empty floor", "goal_dir": "ahead"}


@pytest.fixture
def fake(tmp_path):
    return LLM("fake", cache_dir=tmp_path)


def test_reflection_produces_valid_rules_citing_evidence(fake):
    rules = reflect(LOG, fake)
    assert rules and rules[0]["id"] == "R1"
    assert set(rules[0]["evidence"]) <= {"E1", "E2", "E3"}


def test_validate_drops_bad_rules_and_unknown_evidence():
    rules = validate([{"if": "hot", "evidence": ["E1", "E99"], "confidence": 7},
                      {"then": "missing if"}, "not a dict"], LOG)
    assert len(rules) == 1
    assert rules[0]["evidence"] == ["E1"] and rules[0]["confidence"] == 1.0


def test_lookup_only_knows_exact_killers():
    s = make_shield("lookup", CFG, LOG)
    assert s.judge("red lava").dangerous
    assert not s.judge("molten rock").dangerous


def test_oracle_follows_hidden_categories():
    s = make_shield("oracle", CFG, LOG)
    assert s.judge("molten rock").dangerous and not s.judge("red carpet").dangerous


def test_mask_blocks_only_forward_and_counts(fake):
    s = make_shield("llm_rules", CFG, LOG, reflect(LOG, fake), fake)
    mask, note = s.mask(view("molten rock"))
    assert mask[FORWARD] is False and mask[0] and mask[1] and "BLOCKED" in note
    assert s.mask(view("wall")) == (None, "")
    assert s.seen["molten rock"] == 1 and s.blocked["molten rock"] == 1


def test_llm_judgements_are_memoised(fake):
    s = make_shield("llm_prior", CFG, LOG, llm=fake)
    s.judge("molten rock")
    s.judge("molten rock")
    assert fake.api_calls == 1


def test_none_never_blocks():
    s = make_shield("none", CFG, LOG)
    assert s.mask(view("red lava")) == (None, "")
