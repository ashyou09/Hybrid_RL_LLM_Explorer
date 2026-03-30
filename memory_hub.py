"""
memory_hub.py — System 3: Vector Database (ChromaDB + SentenceTransformers)
============================================================================
Stores LLM‑generated rules as high‑dimensional vectors.
Any agent can query the hub with a text description of what it sees;
if the cosine similarity exceeds the threshold, the danger rule is returned.
"""

import hashlib
import chromadb
import numpy as np
from chromadb.utils import embedding_functions


class MemoryHub:
    """Vector store that bridges RL knowledge → symbolic planners.

    Uses a local PersistentClient folder and wipes it at init so each run
    starts from a clean slate.
    """

    DB_PATH = ".chroma_db"

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        import shutil, os
        # Nuke any leftover data from previous runs
        shutil.rmtree(self.DB_PATH, ignore_errors=True)
        os.makedirs(self.DB_PATH, exist_ok=True)

        self._fallback_rules = []  # in-memory backup for HF/DB failures
        self.client = None
        self.collection = None

        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        try:
            self.client = chromadb.PersistentClient(path=self.DB_PATH)
            self.collection = self.client.create_collection(
                name="semantic_rules",
                embedding_function=self.embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"[Memory Hub] Initialized ChromaDB Vector Store ({self.DB_PATH}, clean slate).")
        except Exception as e:
            self.collection = None
            print(f"[Memory Hub] ChromaDB unavailable ({e}). Using in-memory fallback store.")

    def _embed_one(self, text):
        try:
            vec = self.embed_fn([text])[0]
            return np.asarray(vec, dtype=np.float32)
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return -1.0
        return float(np.dot(a, b) / denom)

    def _store_fallback(self, trigger, rule_data):
        emb = self._embed_one(trigger)
        self._fallback_rules.append({
            "trigger": trigger,
            "embedding": emb,
            "meta": {
                "rule": rule_data["rule"],
                "forbidden_action": rule_data["forbidden_action"],
            },
        })
        print(f"[Memory Hub] Stored rule for '{trigger}' (fallback store).")

    # ── store a new rule ──

    def store_verified_rule(self, rule_data):
        """Embed the trigger keyword so future queries on visual context match it."""
        trigger = rule_data["trigger_feature"]
        doc_id  = hashlib.md5(trigger.encode()).hexdigest()

        if self.collection is None:
            self._store_fallback(trigger, rule_data)
            return

        try:
            self.collection.add(
                documents=[trigger],
                metadatas=[{
                    "rule":             rule_data["rule"],
                    "forbidden_action": rule_data["forbidden_action"],
                }],
                ids=[doc_id],
            )
            print(f"[Memory Hub] Stored rule for '{trigger}'.")
        except Exception as e:
            # Runtime DB faults can happen on HF Spaces (sqlite/chroma internals).
            print(f"[Memory Hub] ChromaDB add failed ({e}). Switching to fallback store.")
            self.collection = None
            self._store_fallback(trigger, rule_data)

    # ── query: "is what I see dangerous?" ──

    def query_local_context(self, text, threshold=0.70, silent=False):
        """Return the matching rule metadata if `text` is semantically close
        to a stored danger trigger, else None.
        `threshold` is cosine similarity; distances below (1‑threshold) match."""
        if self.collection is not None:
            try:
                if self.collection.count() == 0:
                    return None

                results = self.collection.query(query_texts=[text], n_results=1)
                if not results["documents"][0]:
                    return None

                dist = results["distances"][0][0]       # cosine distance
                meta = results["metadatas"][0][0]

                if dist <= (1.0 - threshold):           # e.g. 0.30 for 0.70 threshold
                    if not silent:
                        print(f"[Memory Hub] Match ({dist:.2f} dist) → {meta['rule']}")
                    return meta
                return None
            except Exception as e:
                print(f"[Memory Hub] ChromaDB query failed ({e}). Switching to fallback store.")
                self.collection = None

        # Fallback path: in-memory nearest-neighbor using same embeddings
        if not self._fallback_rules:
            return None

        q_emb = self._embed_one(text)
        if q_emb is None:
            # Last-resort lexical fallback
            for item in self._fallback_rules:
                if text.lower() == item["trigger"].lower():
                    return item["meta"]
            return None

        best_sim = -1.0
        best_meta = None
        for item in self._fallback_rules:
            emb = item["embedding"]
            if emb is None:
                continue
            sim = self._cosine_similarity(q_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_meta = item["meta"]

        if best_meta is not None and best_sim >= threshold:
            if not silent:
                dist = 1.0 - best_sim
                print(f"[Memory Hub] Match ({dist:.2f} dist) → {best_meta['rule']} [fallback]")
            return best_meta
        return None
