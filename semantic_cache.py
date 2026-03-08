"""
semantic_cache.py — In-memory semantic cache built from first principles.

Architecture:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  SemanticCache                                                       │
  │                                                                      │
  │  _buckets: Dict[int, List[CacheEntry]]                               │
  │     └── keyed by dominant_cluster of the query embedding            │
  │         This is a cluster-partitioned hash map.                      │
  │                                                                      │
  │  _global_entries: List[CacheEntry]                                   │
  │     └── fallback list for queries whose cluster is uncertain         │
  └─────────────────────────────────────────────────────────────────────┘

Design decisions:

1. DATA STRUCTURE — Cluster-partitioned list:
   - The naive cache is a flat list: lookup is O(N) — every new query must be
     compared against every cached query. This breaks at scale.
   - We partition cache entries by the *dominant cluster* of their query
     embedding. On lookup, we only search the matching cluster bucket (plus
     neighbouring buckets for boundary queries). This reduces average lookup
     from O(N) to O(N/k) — a k-fold speedup (k=15 in our case).
   - The cluster structure from Part 2 is doing *real work* here: it is not
     just an analysis artefact, it is the indexing structure for the cache.
   - No Redis, no SQLite, no external data structures. Just Python dicts and lists.

2. THE TUNABLE DECISION — SIMILARITY THRESHOLD (τ):
   - τ is the cosine similarity threshold above which we consider two queries
     "the same question phrased differently" and return the cached result.
   - This is the most consequential hyperparameter in the system.
   - τ=1.0: only exact (near-identical) queries hit the cache. High precision,
     near-zero recall. The cache is essentially useless.
   - τ=0.95: catches rephrased questions ("What is AI?" ≈ "What is artificial
     intelligence?"). Good balance for most applications.
   - τ=0.85: more aggressive. May return cached results for genuinely different
     queries that happen to share vocabulary. Risk of semantic confusion.
   - τ=0.75: too permissive for most use cases. "What are guns?" and
     "What is gun control policy?" might match at this level.
   - We expose τ as a constructor parameter with default=0.92.
   - The exploration: each threshold level reveals a different *behavioural
     regime*:
       τ > 0.95: cache behaves like an exact-match cache (safe, conservative)
       0.90–0.95: true semantic caching — the regime where it adds most value
       0.80–0.90: generalist caching — aggressive, useful for FAQ-style systems
       < 0.80: semantic noise — retrieves results for semantically distinct queries
   - We default to τ=0.92 as it sits in the "true semantic caching" regime.

3. THREAD SAFETY:
   - We use a threading.Lock to protect cache mutations. FastAPI runs handlers
     in a thread pool, so concurrent writes without locking would cause races.

4. RESULT COMPUTATION:
   - On a cache miss, we embed the query, find the top-5 most similar corpus
     documents from ChromaDB (filtered to the dominant cluster), and return
     a structured summary as the "result".
   - This simulates a retrieval-augmented generation (RAG) response.
"""

import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from embeddings import embed_query, query_similar
import json


# ── CacheEntry ─────────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    query: str
    query_embedding: np.ndarray   # shape (384,) — normalised
    result: Any                   # the cached result payload
    dominant_cluster: int         # which cluster bucket this lives in
    membership_vector: np.ndarray # full soft membership, shape (k,)
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0            # how many times this entry served a cache hit


# ── SemanticCache ──────────────────────────────────────────────────────────────

class SemanticCache:
    """
    Cluster-partitioned semantic cache.

    Parameters
    ----------
    similarity_threshold : float
        Cosine similarity τ above which a query is considered a cache hit.
        Default 0.92 (true semantic caching regime).
        See module docstring for full threshold analysis.
    n_clusters : int
        Number of fuzzy clusters (must match the clustering model).
    membership_matrix : np.ndarray or None
        (k, N) matrix from FCM. If provided, we use it to assign new query
        embeddings to clusters by projecting onto cluster centroids.
    cluster_centers_pca : np.ndarray or None
        (k, pca_dim) cluster centres in PCA space. Used to assign incoming
        query embeddings to clusters at query time.
    pca : fitted sklearn PCA or None
        If provided, used to project query embeddings before cluster assignment.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        n_clusters: int = 15,
        cluster_centers_pca: Optional[np.ndarray] = None,
        pca=None,
    ):
        self.similarity_threshold = similarity_threshold
        self.n_clusters = n_clusters
        self.cluster_centers_pca = cluster_centers_pca
        self.pca = pca

        # Core data structure: one bucket (list of CacheEntry) per cluster
        self._buckets: Dict[int, List[CacheEntry]] = {i: [] for i in range(n_clusters)}
        self._uncertain_bucket: List[CacheEntry] = []  # for queries with low max membership

        # Stats
        self._hit_count = 0
        self._miss_count = 0

        # Thread safety
        self._lock = threading.Lock()

    # ── Cluster assignment for a query ─────────────────────────────────────────

    def _assign_cluster(self, query_embedding: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Given a (384,) query embedding, return (dominant_cluster, membership_vector).

        We compute soft membership by measuring cosine distance from the query
        (projected into PCA space) to each cluster centroid, then inverting to
        get membership-like weights.

        This is not true FCM prediction (which requires all N points), but it
        is a principled approximation: new point membership ∝ 1/dist(q, centroid).
        """
        if self.cluster_centers_pca is None or self.pca is None:
            # Fallback: return cluster -1 (use uncertain bucket)
            return -1, np.ones(self.n_clusters) / self.n_clusters

        # Project to PCA space
        q_pca = self.pca.transform(query_embedding.reshape(1, -1))[0]  # (pca_dim,)

        # Euclidean distance to each centroid
        dists = np.linalg.norm(self.cluster_centers_pca - q_pca, axis=1)  # (k,)

        # Fuzzy C-Means membership formula: u_ij = 1 / sum_l (d_ij/d_lj)^(2/(m-1))
        # With m=2: u_ij = 1 / sum_l (d_ij/d_lj)^2
        # Avoid division by zero
        dists = np.clip(dists, 1e-10, None)
        m = 2.0
        exponent = 2.0 / (m - 1)  # = 2.0 for m=2

        inv_dists = 1.0 / (dists ** exponent)  # (k,)
        membership = inv_dists / inv_dists.sum()  # normalise to sum=1

        dominant = int(np.argmax(membership))
        return dominant, membership

    # ── Lookup ─────────────────────────────────────────────────────────────────

    def lookup(self, query: str, query_embedding: np.ndarray) -> Optional[dict]:
        """
        Check if a semantically equivalent query is cached.

        Returns a hit dict or None.

        Lookup strategy:
          1. Assign query to a cluster bucket.
          2. Search that bucket first (O(bucket_size)).
          3. If max_membership < 0.5 (boundary query), also search neighbouring
             buckets — this handles documents at cluster boundaries properly.
          4. Return the best match if similarity >= threshold.
        """
        dominant, membership = self._assign_cluster(query_embedding)

        with self._lock:
            best_score = -1.0
            best_entry = None

            # Determine which buckets to search
            buckets_to_search = []
            if dominant == -1:
                buckets_to_search.append(self._uncertain_bucket)
            else:
                buckets_to_search.append(self._buckets[dominant])
                # For boundary queries (max membership < 0.5), also check
                # secondary cluster bucket
                if membership.max() < 0.5:
                    secondary = int(np.argsort(membership)[-2])
                    buckets_to_search.append(self._buckets[secondary])
                # Always check uncertain bucket
                buckets_to_search.append(self._uncertain_bucket)

            for bucket in buckets_to_search:
                for entry in bucket:
                    # Cosine similarity of normalised vectors = dot product
                    score = float(np.dot(query_embedding, entry.query_embedding))
                    if score > best_score:
                        best_score = score
                        best_entry = entry

            if best_entry is not None and best_score >= self.similarity_threshold:
                self._hit_count += 1
                best_entry.hit_count += 1
                return {
                    "cache_hit": True,
                    "matched_query": best_entry.query,
                    "similarity_score": round(best_score, 4),
                    "result": best_entry.result,
                    "dominant_cluster": best_entry.dominant_cluster,
                }

            self._miss_count += 1
            return None

    # ── Store ──────────────────────────────────────────────────────────────────

    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: Any,
    ) -> CacheEntry:
        """
        Store a new cache entry in the appropriate cluster bucket.
        """
        dominant, membership = self._assign_cluster(query_embedding)

        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding,
            result=result,
            dominant_cluster=dominant,
            membership_vector=membership,
        )

        with self._lock:
            if dominant == -1:
                self._uncertain_bucket.append(entry)
            else:
                self._buckets[dominant].append(entry)

        return entry

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def total_entries(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._buckets.values()) + len(self._uncertain_bucket)

    @property
    def stats(self) -> dict:
        total = self._hit_count + self._miss_count
        return {
            "total_entries": self.total_entries,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(self._hit_count / total, 4) if total > 0 else 0.0,
            "similarity_threshold": self.similarity_threshold,
            "bucket_sizes": {
                str(k): len(v) for k, v in self._buckets.items() if len(v) > 0
            },
        }

    # ── Flush ──────────────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Clear all cache entries and reset stats."""
        with self._lock:
            for k in self._buckets:
                self._buckets[k] = []
            self._uncertain_bucket = []
            self._hit_count = 0
            self._miss_count = 0

    # ── Threshold analysis ─────────────────────────────────────────────────────

    def describe_threshold_behaviour(self) -> str:
        """
        Return a human-readable explanation of the current threshold regime.
        """
        τ = self.similarity_threshold
        if τ >= 0.98:
            regime = "near-exact match (cache barely activates)"
        elif τ >= 0.95:
            regime = "conservative semantic (same question, slightly different wording)"
        elif τ >= 0.90:
            regime = "true semantic caching (paraphrases, synonyms — recommended)"
        elif τ >= 0.80:
            regime = "aggressive semantic (topically similar questions may match)"
        else:
            regime = "permissive / noisy (semantically distinct queries may collide)"
        return f"τ={τ}: {regime}"


# ── Result computation (cache miss handler) ────────────────────────────────────

def compute_result(query: str, query_embedding: np.ndarray, dominant_cluster: int) -> dict:
    """
    On a cache miss: retrieve the top-5 most similar corpus documents
    and return a structured result payload.

    In a production RAG system this would call an LLM. Here we return
    the retrieved passages — the retrieval step is the expensive part.
    """
    results = query_similar(
        query_embedding=query_embedding,
        n_results=5,
        cluster_filter=dominant_cluster if dominant_cluster >= 0 else None,
    )

    passages = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        passages.append(
            {
                "text": doc[:500],
                "category": meta.get("category", "unknown"),
                "similarity": round(1 - dist, 4),  # ChromaDB returns distance, not similarity
            }
        )

    return {
        "query": query,
        "top_passages": passages,
        "retrieval_method": "cluster-filtered cosine similarity",
    }
