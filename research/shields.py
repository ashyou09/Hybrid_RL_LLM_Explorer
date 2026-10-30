"""Shields: decide "is the tile in front dangerous?" and, if so, forbid moving forward.

    S0 none        – never blocks (the baseline)
    S1 lookup      – blocks only tiles that killed Agent A (exact name)
    S2 embedding   – blocks tiles whose name is similar to a killer tile
                     (all-MiniLM-L6-v2 cosine ≥ embed_threshold, as in the v0 demo)
    S3 llm_prior   – asks the LLM with NO experience (common sense only)
    S4 llm_rules   – asks the LLM with Agent A's rules and experience (ours)
    S6 oracle      – knows the hidden categories (upper limit)

(S5, "with verification", arrives in Level 3.)

Every shield sees what the agent sees: the tile's name and description. Judgements
are memoised per tile, and LLM answers are cached on disk, so each distinct tile costs
at most one LLM call per shield.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from research import experience, tiles
from research.config import Config
from research.env import EMPTY, FORWARD, GOAL, WALL
from research.llm import LLM, extract_json
from research.rules import to_text as rules_text

SHIELDS = ("none", "lookup", "embedding", "llm_prior", "llm_rules", "oracle")
LLM_SHIELDS = ("llm_prior", "llm_rules")
LABELS = {"none": "S0 none", "lookup": "S1 lookup", "embedding": "S2 embedding",
          "llm_prior": "S3 LLM prior", "llm_rules": "S4 LLM + experience", "oracle": "S6 oracle"}

JUDGE_SYSTEM = ("You protect an agent walking in a grid world. Decide whether stepping "
                "onto a floor tile is likely to kill it. Reply with JSON only.")

PRIOR_PROMPT = """\
The agent is about to step onto a floor tile it has never seen before.

TILE TO JUDGE: {name} — {description}

Is stepping onto this tile likely to kill the agent?
Reply with JSON only: {{"dangerous": true or false, "rule": null, "reason": "<one short sentence>"}}
"""

RULES_PROMPT = """\
Rules learned from the agent's own experience in THIS world:
{rules}

The experiences behind them:
{experience}

The agent is about to step onto a floor tile.

TILE TO JUDGE: {name} — {description}

Using the rules and experience above, is stepping onto this tile likely to kill the agent
in this world? Judge by the tile's real properties, not by its colour or name alone.
Reply with JSON only: {{"dangerous": true or false, "rule": "<rule id or null>", "reason": "<one short sentence>"}}
"""


@dataclass
class Judgement:
    dangerous: bool
    reason: str


class Shield:
    name = "none"

    def __init__(self):
        self._memo: dict[str, Judgement] = {}
        self.seen = Counter()      # tile name -> times met in front of the agent
        self.blocked = Counter()   # tile name -> times forward was forbidden

    @property
    def label(self) -> str:
        return LABELS[self.name]

    def judge(self, tile_name: str) -> Judgement:
        if tile_name not in self._memo:
            self._memo[tile_name] = self._judge(tiles.get(tile_name))
        return self._memo[tile_name]

    def _judge(self, tile) -> Judgement:
        return Judgement(False, "no shield")

    def mask(self, view: dict):
        """(mask, note) for the agent: forbid forward if the front tile is judged deadly."""
        front = view["front"]
        if front in (EMPTY, WALL, GOAL):
            return None, ""
        self.seen[front] += 1
        j = self.judge(front)
        if not j.dangerous:
            return None, ""
        self.blocked[front] += 1
        mask = [True, True, True]
        mask[FORWARD] = False
        return mask, f"BLOCKED {front}: {j.reason}"


class LookupShield(Shield):
    name = "lookup"

    def __init__(self, log):
        super().__init__()
        self.killers = {e["tile"] for e in log if e["event"] == "died_on"}

    def _judge(self, tile):
        if tile.name in self.killers:
            return Judgement(True, "it killed me before")
        return Judgement(False, "never killed me")


class EmbeddingShield(Shield):
    name = "embedding"

    def __init__(self, log, threshold: float, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("The embedding shield needs: pip install sentence-transformers") from e
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)
        self.killers = sorted({e["tile"] for e in log if e["event"] == "died_on"})
        self.killer_vecs = self.model.encode(self.killers, normalize_embeddings=True) \
            if self.killers else None

    def _judge(self, tile):
        if self.killer_vecs is None:
            return Judgement(False, "no killer tiles to compare with")
        sims = self.killer_vecs @ self.model.encode([tile.name], normalize_embeddings=True)[0]
        i = int(sims.argmax())
        close = float(sims[i]) >= self.threshold
        return Judgement(close, f"similarity {sims[i]:.2f} to {self.killers[i]}")


class LLMShield(Shield):
    def __init__(self, llm: LLM, prompt: str, **context):
        super().__init__()
        self.llm, self.prompt, self.context = llm, prompt, context

    def _judge(self, tile):
        reply = self.llm.complete(
            self.prompt.format(name=tile.name, description=tile.description, **self.context),
            system=JUDGE_SYSTEM)
        try:
            data = extract_json(reply)
            dangerous = data.get("dangerous")
            if isinstance(dangerous, str):
                dangerous = dangerous.strip().lower() == "true"
            reason = str(data.get("reason", "")).strip()
            rule = data.get("rule")
            return Judgement(bool(dangerous), f"{rule}: {reason}" if rule else reason)
        except (ValueError, AttributeError):
            # unparseable answer: count it as "not dangerous" and keep the raw text visible
            return Judgement(False, f"unparseable LLM reply: {reply[:80]!r}")


class LLMPriorShield(LLMShield):
    name = "llm_prior"

    def __init__(self, llm: LLM):
        super().__init__(llm, PRIOR_PROMPT)


class LLMRulesShield(LLMShield):
    name = "llm_rules"

    def __init__(self, llm: LLM, rules: list[dict], log: list[dict]):
        super().__init__(llm, RULES_PROMPT, rules=rules_text(rules),
                         experience=experience.to_text(log))


class OracleShield(Shield):
    name = "oracle"

    def __init__(self, deadly_categories):
        super().__init__()
        self.deadly = set(deadly_categories)

    def _judge(self, tile):
        return Judgement(tile.category in self.deadly, "ground truth")


def make_shield(name: str, cfg: Config, log: list[dict], rules: list[dict] | None = None,
                llm: LLM | None = None) -> Shield:
    if name == "none":
        return Shield()
    if name == "lookup":
        return LookupShield(log)
    if name == "embedding":
        return EmbeddingShield(log, cfg.embed_threshold)
    if name == "oracle":
        return OracleShield(cfg.deadly_categories)
    if name in LLM_SHIELDS:
        if llm is None:
            raise ValueError(f"{name} needs an LLM")
        if name == "llm_prior":
            return LLMPriorShield(llm)
        return LLMRulesShield(llm, rules or [], log)
    raise ValueError(f"Unknown shield {name!r}; choose from {SHIELDS}")
