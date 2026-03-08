"""
embeddings.py — Embedding model and ChromaDB vector store setup.

Design decisions:

1. EMBEDDING MODEL — all-MiniLM-L6-v2:
   - 384-dimensional embeddings, fast on CPU (no GPU required for this task).
   - Trained specifically for semantic similarity tasks (as opposed to
     classification-tuned models), which is exactly what semantic search needs.
   - Better semantic quality than TF-IDF/BM25 for paraphrase detection, which
     is the core requirement of the semantic cache.
   - Trade-off acknowledged: larger models (e.g. all-mpnet-base-v2, 768-dim)
     produce better embeddings but are ~3× slower. For a 10k-doc corpus on CPU,
     MiniLM gives a practical embedding time of ~5–10 min vs ~20+ min.
   - We chose NOT to fine-tune on this corpus: the assignment scope is retrieval,
     and MiniLM's general-purpose semantic space already captures the topical
     distinctions in 20NG well.

2. VECTOR STORE — ChromaDB:
   - Persistent, local, zero-infrastructure. Fits the "lightweight" requirement.
   - Supports filtered retrieval by metadata (category, cluster) out of the box.
   - Better than FAISS for this use case because it natively stores metadata
     alongside vectors, letting us filter by cluster in Part 3 without a
     separate metadata store.
   - Trade-off: FAISS is faster for pure ANN lookup at large scale, but
     ChromaDB's overhead is negligible at 10k docs and we gain metadata filtering.

3. BATCH SIZE:
   - 128 docs per batch. Balances RAM usage vs overhead of repeated model
     forward passes. MiniLM peaks at ~800 MB at this batch size on CPU.
"""

import os
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
from typing import List, Optional


# ── Constants ──────────────────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "newsgroups_corpus"
BATCH_SIZE = 128


# ── Model singleton ────────────────────────────────────────────────────────────

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: List[str], show_progress: bool = True) -> np.ndarray:
    """
    Embed a list of strings and return a float32 ndarray of shape (N, 384).
    Uses batching to avoid OOM on large corpora.
    """
    model = get_model()
    all_embeddings = []
    iterator = range(0, len(texts), BATCH_SIZE)
    if show_progress:
        iterator = tqdm(iterator, desc="Embedding")

    for start in iterator:
        batch = texts[start : start + BATCH_SIZE]
        embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        all_embeddings.append(embs)

    return np.vstack(all_embeddings).astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (384,)."""
    model = get_model()
    return model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


# ── ChromaDB helpers ───────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_or_create_collection(client: Optional[chromadb.PersistentClient] = None):
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for normalised vectors
    )


def build_vector_store(df: pd.DataFrame, embeddings: np.ndarray) -> None:
    """
    Persist the corpus embeddings and metadata into ChromaDB.

    Metadata stored per document:
      - category     : newsgroup label string
      - category_id  : integer 0–19
      - token_count  : post length in tokens
      - dominant_cluster: set later after fuzzy clustering (default -1)
    """
    client = get_chroma_client()
    # Drop and recreate to allow idempotent rebuilds
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Inserting {len(df)} documents into ChromaDB...")
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Inserting"):
        batch_df = df.iloc[start : start + BATCH_SIZE]
        batch_embs = embeddings[start : start + BATCH_SIZE]

        collection.add(
            ids=[str(row.doc_id) for _, row in batch_df.iterrows()],
            embeddings=batch_embs.tolist(),
            documents=[row.clean_text[:2000] for _, row in batch_df.iterrows()],  # truncate for storage
            metadatas=[
                {
                    "category": row.category,
                    "category_id": int(row.category_id),
                    "token_count": int(row.token_count),
                    "dominant_cluster": -1,  # placeholder; filled after clustering
                }
                for _, row in batch_df.iterrows()
            ],
        )

    print(f"Vector store built. Collection size: {collection.count()}")


def update_cluster_metadata(doc_ids: List[str], cluster_labels: List[int]) -> None:
    """After fuzzy clustering, write the dominant cluster back to ChromaDB metadata."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    for start in tqdm(range(0, len(doc_ids), BATCH_SIZE), desc="Updating cluster metadata"):
        batch_ids = doc_ids[start : start + BATCH_SIZE]
        batch_clusters = cluster_labels[start : start + BATCH_SIZE]
        existing = collection.get(ids=batch_ids, include=["metadatas"])
        updated_metadatas = []
        for meta, cl in zip(existing["metadatas"], batch_clusters):
            meta["dominant_cluster"] = int(cl)
            updated_metadatas.append(meta)
        collection.update(ids=batch_ids, metadatas=updated_metadatas)


def query_similar(
    query_embedding: np.ndarray,
    n_results: int = 10,
    cluster_filter: Optional[int] = None,
):
    """
    Return the top-n most similar documents.
    Optionally filter to a specific dominant_cluster for cache-accelerated lookup.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    where = None
    if cluster_filter is not None:
        where = {"dominant_cluster": {"$eq": cluster_filter}}

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return results


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet("data/corpus.parquet")
    print("Embedding corpus...")
    embeddings = embed_texts(df["clean_text"].tolist())
    np.save("data/embeddings.npy", embeddings)
    print("Embeddings saved to data/embeddings.npy")
    build_vector_store(df, embeddings)
