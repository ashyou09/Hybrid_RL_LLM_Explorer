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
        self.client = chromadb.PersistentClient(path=db_path)

        # Always start clean so previous runs don't leak rules
        try:
            self.client.delete_collection(name="semantic_rules")
        except Exception:
            pass

        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = self.client.create_collection(
            name="semantic_rules",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},   # cosine similarity index
        )
        print("[Memory Hub] Initialized ChromaDB Vector Store.")

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
