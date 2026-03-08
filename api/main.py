"""
api/main.py — FastAPI service exposing the semantic cache.

State management:
  - The SemanticCache is a module-level singleton initialised once at startup
    via FastAPI's lifespan context manager.
  - This means the cache persists across requests for the lifetime of the process.
  - The clustering model (PCA + centroids) is loaded from disk at startup.
  - The embedding model is lazy-loaded on first request.

Endpoints:
  POST /query           — semantic search with cache
  GET  /cache/stats     — cache telemetry
  DELETE /cache         — flush cache
  GET  /health          — health check
  GET  /threshold/{tau} — explore how threshold affects cache behaviour (bonus)
"""

import os
import sys
import json
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings import embed_query
from semantic_cache import SemanticCache, compute_result
from clustering import N_CLUSTERS, MODELS_DIR, get_dominant_cluster


# ── Globals (initialised at startup) ──────────────────────────────────────────

_cache: SemanticCache = None
_cluster_centers_pca: np.ndarray = None
_pca = None


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cache, _cluster_centers_pca, _pca

    print("Loading clustering artifacts...")
    try:
        _cluster_centers_pca = np.load(f"{MODELS_DIR}/cluster_centers.npy")
        _pca = joblib.load(f"{MODELS_DIR}/pca.joblib")
        print(f"Loaded cluster centers: {_cluster_centers_pca.shape}, PCA: {_pca.n_components_} components")
    except FileNotFoundError:
        print("WARNING: Clustering artifacts not found. Run scripts/build_index.py first.")
        print("Cache will operate without cluster partitioning (degraded performance).")
        _cluster_centers_pca = None
        _pca = None

    # Initialise cache with default threshold τ=0.92
    # This is the "true semantic caching" regime — see semantic_cache.py for analysis
    _cache = SemanticCache(
        similarity_threshold=float(os.environ.get("SIMILARITY_THRESHOLD", "0.92")),
        n_clusters=N_CLUSTERS,
        cluster_centers_pca=_cluster_centers_pca,
        pca=_pca,
    )
    print(f"Semantic cache ready. {_cache.describe_threshold_behaviour()}")

    yield  # Application runs here

    print("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Semantic Search & Cache API",
    description=(
        "Semantic search over the 20 Newsgroups corpus with a cluster-partitioned "
        "semantic cache. Built from first principles without Redis or caching libraries."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, example="What are the best programming languages?")


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: str | None = None
    similarity_score: float | None = None
    result: dict
    dominant_cluster: int


class CacheStats(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cache_entries": _cache.total_entries if _cache else 0,
        "threshold": _cache.similarity_threshold if _cache else None,
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Embed the query, check the semantic cache, and return results.

    Cache hit:  returns cached result with matched_query and similarity_score.
    Cache miss: computes result via ChromaDB retrieval, stores it, returns it.
    """
    if _cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialised")

    query = request.query.strip()

    # 1. Embed the query
    query_embedding = embed_query(query)

    # 2. Check cache
    hit = _cache.lookup(query, query_embedding)
    if hit:
        return QueryResponse(
            query=query,
            cache_hit=True,
            matched_query=hit["matched_query"],
            similarity_score=hit["similarity_score"],
            result=hit["result"],
            dominant_cluster=hit["dominant_cluster"],
        )

    # 3. Cache miss — compute result
    dominant_cluster, membership = _cache._assign_cluster(query_embedding)

    result = compute_result(query, query_embedding, dominant_cluster)

    # 4. Store in cache
    entry = _cache.store(query, query_embedding, result)

    return QueryResponse(
        query=query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result,
        dominant_cluster=dominant_cluster,
    )


@app.get("/cache/stats", response_model=CacheStats)
async def cache_stats():
    """Return current cache telemetry."""
    if _cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialised")
    s = _cache.stats
    return CacheStats(
        total_entries=s["total_entries"],
        hit_count=s["hit_count"],
        miss_count=s["miss_count"],
        hit_rate=s["hit_rate"],
    )


@app.delete("/cache")
async def flush_cache():
    """Flush all cache entries and reset hit/miss counters."""
    if _cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialised")
    _cache.flush()
    return {"status": "cache flushed", "entries_cleared": True}


@app.get("/cache/details")
async def cache_details():
    """Extended cache stats including bucket distribution and threshold regime."""
    if _cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialised")
    return _cache.stats


@app.get("/threshold/{tau}")
async def threshold_analysis(tau: float):
    """
    Explore the effect of a given similarity threshold τ on cache behaviour.
    Returns the behavioural regime description and a preview of what would
    happen to current cache entries under this threshold.

    This endpoint illustrates the key design trade-off: the tunable threshold
    is what separates exact-match caching from genuine semantic caching.
    """
    if not 0.0 <= tau <= 1.0:
        raise HTTPException(status_code=400, detail="τ must be in [0, 1]")

    # Describe the regime
    if tau >= 0.98:
        regime = "near-exact match (cache barely activates)"
        risk = "Very low false-positive rate but cache provides little value."
    elif tau >= 0.95:
        regime = "conservative semantic (same question, slightly different wording)"
        risk = "Low false-positive rate. Catches obvious rephrases."
    elif tau >= 0.90:
        regime = "true semantic caching (paraphrases, synonyms)"
        risk = "Balanced. This is the recommended production setting."
    elif tau >= 0.80:
        regime = "aggressive semantic (topically similar questions may match)"
        risk = "Higher false-positive risk. Useful for FAQ-style narrow-domain systems."
    else:
        regime = "permissive / noisy (semantically distinct queries may collide)"
        risk = "High false-positive rate. Semantically different queries may get same result."

    return {
        "tau": tau,
        "regime": regime,
        "risk_profile": risk,
        "current_cache_entries": _cache.total_entries,
        "note": (
            "The threshold is the single tunable decision that defines cache behaviour. "
            "Each regime reveals fundamentally different system properties — "
            "precision vs recall of semantic equivalence detection."
        ),
    }
