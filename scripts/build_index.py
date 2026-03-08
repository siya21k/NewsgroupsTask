#!/usr/bin/env python3
"""
scripts/build_index.py

Master script that runs all preprocessing steps in order:
  1. Load and clean the 20 Newsgroups corpus
  2. Embed all documents
  3. Build ChromaDB vector store
  4. Run fuzzy C-Means clustering
  5. Update ChromaDB with cluster labels

Run once before starting the API server:
  python scripts/build_index.py

Takes ~10-25 minutes on CPU (dominated by embedding ~10k docs).
"""

import os
import sys
import numpy as np
import time

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


def step(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def main():
    t0 = time.time()

    # ── Step 1: Data preparation ────────────────────────────────────────────
    step("Step 1/5: Data preparation")
    if os.path.exists("data/corpus.parquet"):
        print("corpus.parquet found, skipping.")
        import pandas as pd
        df = pd.read_parquet("data/corpus.parquet")
        print(f"Loaded {len(df)} documents from cache.")
    else:
        from data_prep import load_and_prepare
        df = load_and_prepare(max_docs=10_000, min_tokens=50)
        df.to_parquet("data/corpus.parquet", index=False)
        print(f"Saved {len(df)} documents to data/corpus.parquet")

    # ── Step 2: Embeddings ──────────────────────────────────────────────────
    step("Step 2/5: Computing embeddings")
    if os.path.exists("data/embeddings.npy"):
        print("embeddings.npy found, skipping.")
        embeddings = np.load("data/embeddings.npy")
        print(f"Loaded embeddings: {embeddings.shape}")
    else:
        from embeddings import embed_texts
        embeddings = embed_texts(df["clean_text"].tolist())
        np.save("data/embeddings.npy", embeddings)
        print(f"Saved embeddings: {embeddings.shape} to data/embeddings.npy")

    # ── Step 3: Vector store ────────────────────────────────────────────────
    step("Step 3/5: Building ChromaDB vector store")
    from embeddings import build_vector_store, get_chroma_client, COLLECTION_NAME
    client = get_chroma_client()
    try:
        col = client.get_collection(COLLECTION_NAME)
        existing_count = col.count()
        if existing_count == len(df):
            print(f"ChromaDB already has {existing_count} docs, skipping rebuild.")
        else:
            print(f"ChromaDB has {existing_count} docs but corpus has {len(df)}, rebuilding.")
            build_vector_store(df, embeddings)
    except Exception:
        build_vector_store(df, embeddings)

    # ── Step 4: Fuzzy clustering ────────────────────────────────────────────
    step("Step 4/5: Fuzzy C-Means clustering")
    if (
        os.path.exists("models/membership_matrix.npy")
        and os.path.exists("models/cluster_centers.npy")
        and os.path.exists("models/pca.joblib")
        and os.path.exists("data/corpus_clustered.parquet")
    ):
        print("Clustering artifacts found, skipping.")
        from clustering import load_clustering_artifacts
        membership_matrix, centers, pca, df_clustered = load_clustering_artifacts()
    else:
        from clustering import (
            reduce_dimensions, evaluate_k, run_fuzzy_cmeans,
            describe_clusters, save_clustering_artifacts,
            plot_membership_heatmap, plot_k_selection, N_CLUSTERS
        )
        import json

        # Reduce dims
        reduced, pca = reduce_dimensions(embeddings)

        # Evaluate k (optional, can be commented out to save time)
        print("\nEvaluating candidate k values (subsample=2000)...")
        eval_df = evaluate_k(reduced, k_values=[8, 12, 15, 18, 20], sample_size=2000)
        print(eval_df.to_string(index=False))
        plot_k_selection(eval_df, "data/k_selection.png")

        # Run FCM with chosen k
        membership_matrix, centers, fpc = run_fuzzy_cmeans(reduced, k=N_CLUSTERS)

        # Analyse
        summary, df_clustered = describe_clusters(df, membership_matrix)

        print("\n=== Cluster Summary ===")
        for c, info in summary.items():
            print(f"\nCluster {c:2d} | size={info['size']:4d} | entropy={info['avg_entropy']:.3f}")
            for cat, pct in list(info["top_categories"].items())[:3]:
                print(f"           {cat}: {pct:.0%}")

        # Plots
        plot_membership_heatmap(membership_matrix, "data/membership_heatmap.png")

        # Serialisable summary
        for c in summary:
            summary[c]["top_categories"] = {k: float(v) for k, v in summary[c]["top_categories"].items()}
            summary[c].pop("sample_docs", None)
            summary[c].pop("boundary_docs", None)
        with open("data/cluster_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save
        save_clustering_artifacts(membership_matrix, centers, pca, df_clustered)

    # ── Step 5: Update ChromaDB with cluster labels ─────────────────────────
    step("Step 5/5: Writing cluster labels to ChromaDB")
    from clustering import get_dominant_cluster
    from embeddings import update_cluster_metadata

    dominant = get_dominant_cluster(membership_matrix)
    doc_ids = [str(i) for i in df_clustered["doc_id"].tolist()]
    update_cluster_metadata(doc_ids, dominant.tolist())

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  BUILD COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    print("\nStart the API with:")
    print("  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    main()
