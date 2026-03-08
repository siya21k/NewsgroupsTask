# Semantic Search System — 20 Newsgroups

A lightweight semantic search system with fuzzy clustering, a cluster-partitioned semantic cache, and a FastAPI service. Built for the Trademarkia AI/ML Engineer assignment.

---

## Architecture Overview

```
20 Newsgroups corpus
       │
       ▼
  [data_prep.py]          ← Clean, filter, subsample (10k docs)
       │
       ▼
  [embeddings.py]         ← all-MiniLM-L6-v2 (384-dim, CPU-friendly)
       │                    ChromaDB persistent vector store
       ▼
  [clustering.py]         ← Fuzzy C-Means (k=15) over PCA-50 embeddings
       │                    Each doc gets a membership VECTOR, not a label
       ▼
  [semantic_cache.py]     ← Cluster-partitioned in-memory cache (no Redis)
       │                    τ=0.92 similarity threshold (configurable)
       ▼
  [api/main.py]           ← FastAPI service
       │
       ├── POST /query
       ├── GET  /cache/stats
       ├── DELETE /cache
       ├── GET  /health
       └── GET  /threshold/{tau}   ← bonus: threshold exploration
```

---

## Setup

### Quick start (venv)

```bash
# 1. Create environment
bash setup.sh

# 2. Activate
source venv/bin/activate

# 3. Build the index (run once — takes 10–25 min on CPU)
python scripts/build_index.py

# 4. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build and start
docker-compose up --build

# Build index inside container first
docker-compose run --rm semantic-search python scripts/build_index.py
docker-compose up
```

---

## API Endpoints

### `POST /query`

```json
// Request
{ "query": "What are the best programming languages for AI?" }

// Response (cache miss)
{
  "query": "What are the best programming languages for AI?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": {
    "query": "...",
    "top_passages": [
      { "text": "...", "category": "comp.lang.python", "similarity": 0.87 }
    ],
    "retrieval_method": "cluster-filtered cosine similarity"
  },
  "dominant_cluster": 3
}

// Response (cache hit on similar query)
{
  "query": "Which programming languages are best for machine learning?",
  "cache_hit": true,
  "matched_query": "What are the best programming languages for AI?",
  "similarity_score": 0.9341,
  "result": { ... },
  "dominant_cluster": 3
}
```

### `GET /cache/stats`

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### `DELETE /cache`

```json
{ "status": "cache flushed", "entries_cleared": true }
```

### `GET /threshold/{tau}`

Explore the behavioural regime of a given similarity threshold. See below.

---

## Design Decisions

### Part 1 — Embeddings & Vector Store

**Model: `all-MiniLM-L6-v2`**
- 384-dim, fast on CPU (~5–10 min for 10k docs)
- Trained for semantic similarity (paraphrase detection), which is exactly what the cache needs
- Trade-off vs `all-mpnet-base-v2`: 3× faster, ~5% worse on benchmarks — acceptable for this task

**Vector store: ChromaDB**
- Persistent, zero-infrastructure, native metadata filtering
- We store `dominant_cluster` as metadata, enabling cluster-filtered retrieval
- Trade-off vs FAISS: slower at scale but metadata filtering avoids a separate metadata store

**Cleaning decisions:**
- Strip headers (From:, Subject:, etc.) — metadata artefacts, not semantic content
- Remove quoted replies (`>` lines) — duplicate content inflates similarity
- Drop posts < 50 tokens after cleaning — noise, no semantic signal

### Part 2 — Fuzzy Clustering

**Why Fuzzy C-Means:**
Hard cluster assignments lose the essential property that newsgroup posts frequently belong to multiple topics. FCM gives each document a *membership distribution* over clusters.

**k=15 (not 20):**
The 20 labelled categories have real semantic overlap:
- `comp.sys.ibm.pc.hardware` ≈ `comp.sys.mac.hardware`
- `talk.religion.misc` ≈ `soc.religion.christian`
- `rec.sport.hockey` ≈ `rec.sport.baseball`

We evaluated k ∈ {8, 12, 15, 18, 20, 25} using the **Fuzzy Partition Coefficient (FPC)**: FPC peaks near k=12–15 and plateaus/declines after, indicating 15 is the natural granularity of the embedding space.

**PCA to 50 dims before FCM:**
Running FCM in 384 dims causes the curse of dimensionality — all pairwise distances converge, making memberships uniform. 50 PCA components retain ~85% variance while making distance metrics meaningful.

### Part 3 — Semantic Cache

**Data structure: cluster-partitioned dict of lists**

```python
_buckets: Dict[int, List[CacheEntry]]  # one list per cluster
```

Naive flat-list cache: O(N) lookup — every query compared to every cached entry.

Cluster-partitioned cache: O(N/k) lookup — only search the matching cluster bucket. At k=15, this is a 15× speedup as the cache grows.

The cluster structure from Part 2 does **real work** here — it's not just analysis, it's the index structure.

**The tunable threshold τ:**

| τ | Regime | Behaviour |
|---|--------|-----------|
| 0.98–1.0 | Near-exact match | Cache barely activates |
| 0.95–0.97 | Conservative semantic | Catches obvious rephrases |
| **0.90–0.94** | **True semantic (default)** | **Paraphrases, synonyms match** |
| 0.80–0.89 | Aggressive semantic | Topically similar queries match |
| < 0.80 | Noisy | Semantically different queries collide |

We default to **τ=0.92**. This is the regime where the cache provides genuine value (paraphrase detection) without false positives. The `GET /threshold/{tau}` endpoint and `scripts/explore_threshold.py` let you explore this empirically.

**The key insight:** τ is not a hyperparameter to be tuned for "best performance" — each value reveals a *different behavioural property* of the system. Conservative thresholds make the cache safe; aggressive ones make it useful for narrow-domain FAQ systems.

---

## File Structure

```
newsgroups_semantic_search/
├── data_prep.py              # Corpus loading and cleaning
├── embeddings.py             # Embedding model + ChromaDB
├── clustering.py             # Fuzzy C-Means
├── semantic_cache.py         # Semantic cache (first principles)
├── api/
│   └── main.py               # FastAPI service
├── scripts/
│   ├── build_index.py        # Master build script (run once)
│   └── explore_threshold.py  # Threshold analysis
├── data/                     # Generated: parquet, embeddings, plots
├── models/                   # Generated: PCA, cluster centers
├── requirements.txt
├── setup.sh
├── Dockerfile
└── docker-compose.yml
```

---

## Threshold Exploration

```bash
# Analyse threshold behaviour on representative query pairs
python scripts/explore_threshold.py
# → data/threshold_analysis.png
# → data/threshold_analysis.json
```

Or via the API after startup:
```bash
curl http://localhost:8000/threshold/0.85
curl http://localhost:8000/threshold/0.92
curl http://localhost:8000/threshold/0.97
```

---

## Environment variable

`SIMILARITY_THRESHOLD` — override the default τ=0.92 at startup:
```bash
SIMILARITY_THRESHOLD=0.88 uvicorn api.main:app --port 8000
```
