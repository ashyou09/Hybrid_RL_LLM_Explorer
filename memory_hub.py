"""
memory_hub.py — System 3: Vector Database (ChromaDB + SentenceTransformers)
============================================================================
Stores LLM‑generated rules as high‑dimensional vectors.
Any agent can query the hub with a text description of what it sees;
if the cosine similarity exceeds the threshold, the danger rule is returned.
"""

import hashlib
import chromadb
from chromadb.utils import embedding_functions


class MemoryHub:
    """Persistent vector store that bridges RL knowledge → symbolic planners."""

    def __init__(self, db_path="./chroma_db", model_name="all-MiniLM-L6-v2"):
        # Delete any leftover data on disk first
        import shutil, os
        shutil.rmtree(db_path, ignore_errors=True)
        os.makedirs(db_path, exist_ok=True)

        # Fresh client — no stale in-memory collection state
        self.client = chromadb.PersistentClient(path=db_path)

        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        # get_or_create is safe even if a previous run left something behind
        self.collection = self.client.get_or_create_collection(
            name="semantic_rules",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        # Wipe any docs from a previous partial run that shared the same path
        existing = self.collection.count()
        if existing > 0:
            all_ids = self.collection.get()["ids"]
            self.collection.delete(ids=all_ids)

        print("[Memory Hub] Initialized ChromaDB Vector Store (clean slate).")

    # ── store a new rule ──

    def store_verified_rule(self, rule_data):
        """Embed the trigger keyword so future queries on visual context match it."""
        trigger = rule_data["trigger_feature"]
        doc_id  = hashlib.md5(trigger.encode()).hexdigest()

        self.collection.add(
            documents=[trigger],
            metadatas=[{
                "rule":             rule_data["rule"],
                "forbidden_action": rule_data["forbidden_action"],
            }],
            ids=[doc_id],
        )
        print(f"[Memory Hub] Stored rule for '{trigger}'.")

    # ── query: "is what I see dangerous?" ──

    def query_local_context(self, text, threshold=0.70, silent=False):
        """Return the matching rule metadata if `text` is semantically close
        to a stored danger trigger, else None.
        `threshold` is cosine similarity; distances below (1‑threshold) match."""
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
        except Exception:
            return None
