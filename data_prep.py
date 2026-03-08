"""
data_prep.py — Corpus preparation for the 20 Newsgroups dataset.

Design decisions (justify in comments as required):

1. TEXT CLEANING:
   - We strip email headers (From:, Subject:, Lines:, etc.) because they are
     metadata artefacts, not semantic content. Keeping them would bias embeddings
     toward authorship/routing signals rather than topic signals.
   - We remove quoted reply blocks (lines starting with ">") because they
     duplicate content from other posts and would inflate similarity scores
     between semantically unrelated documents that happen to quote the same source.
   - We strip URLs and email addresses — these are low-entropy tokens that add
     noise without topical information.
   - We enforce a minimum token count of 50 after cleaning. Posts shorter than
     this are almost always administrative noise (e.g., "Thanks!", "+1") and
     provide no useful semantic signal.

2. DATASET SPLIT:
   - We use the full dataset (train + test combined) for building the corpus
     since we are doing unsupervised semantic analysis, not supervised classification.
     The train/test split was designed for classification benchmarking, not for us.

3. SUBSAMPLING:
   - We cap at 10,000 documents for practical embedding speed without GPU.
     We stratify the sample by category to preserve the topical distribution.
"""

import re
import string
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
from tqdm import tqdm


# Headers that are metadata noise, not semantic content
_HEADER_PATTERNS = [
    r"^From:.*$",
    r"^Subject:.*$",
    r"^Lines:.*$",
    r"^Organization:.*$",
    r"^X-.*:.*$",
    r"^Message-ID:.*$",
    r"^Date:.*$",
    r"^Newsgroups:.*$",
    r"^Path:.*$",
    r"^References:.*$",
    r"^NNTP-Posting-Host:.*$",
    r"^Reply-To:.*$",
    r"^Distribution:.*$",
    r"^Sender:.*$",
    r"^Followup-To:.*$",
    r"^Summary:.*$",
    r"^Keywords:.*$",
    r"^Expires:.*$",
]

_HEADER_RE = re.compile("|".join(_HEADER_PATTERNS), re.MULTILINE | re.IGNORECASE)
_QUOTED_RE = re.compile(r"^>.*$", re.MULTILINE)
_URL_RE = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    """
    Strip metadata and noise from a raw newsgroup post.
    Returns cleaned body text, or empty string if nothing survives.
    """
    text = _HEADER_RE.sub(" ", raw)
    text = _QUOTED_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    # Remove non-ASCII (mostly encoding artefacts in this dataset)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def token_count(text: str) -> int:
    """Approximate token count by whitespace splitting."""
    return len(text.split())


def load_and_prepare(
    max_docs: int = 10_000,
    min_tokens: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load 20 Newsgroups, clean, filter, and return a DataFrame.

    Columns:
      - doc_id       : integer index
      - raw_text     : original post
      - clean_text   : cleaned post body
      - category     : string label (e.g. "rec.sport.hockey")
      - category_id  : integer label (0–19)
      - token_count  : approximate word count after cleaning
    """
    print("Fetching 20 Newsgroups dataset (train + test)...")
    # remove_headers=() means we keep everything and do our own cleaning
    # This gives us more control than sklearn's built-in header stripping
    bunch = fetch_20newsgroups(
        subset="all",
        remove=(),  # intentional: we clean ourselves
        shuffle=True,
        random_state=random_state,
    )

    print(f"Loaded {len(bunch.data)} raw documents across {len(bunch.target_names)} categories.")

    records = []
    for idx, (raw, target) in enumerate(tqdm(zip(bunch.data, bunch.target), desc="Cleaning")):
        cleaned = clean_text(raw)
        tc = token_count(cleaned)
        if tc < min_tokens:
            continue  # discard near-empty posts
        records.append(
            {
                "doc_id": idx,
                "raw_text": raw,
                "clean_text": cleaned,
                "category": bunch.target_names[target],
                "category_id": int(target),
                "token_count": tc,
            }
        )

    df = pd.DataFrame(records)
    print(f"After cleaning: {len(df)} documents (dropped {len(bunch.data) - len(df)} short/empty)")

    # Stratified subsample to preserve category distribution
    if len(df) > max_docs:
        print(f"Subsampling to {max_docs} docs (stratified by category)...")
        df = (
            df.groupby("category_id", group_keys=False)
            .apply(
                lambda g: g.sample(
                    min(len(g), int(np.ceil(max_docs * len(g) / len(df)))),
                    random_state=random_state,
                )
            )
            .reset_index(drop=True)
        )
        df = df.sample(min(max_docs, len(df)), random_state=random_state).reset_index(drop=True)

    df["doc_id"] = df.index  # re-index cleanly
    print(f"Final corpus: {len(df)} documents")
    print(df["category"].value_counts().to_string())
    return df


if __name__ == "__main__":
    df = load_and_prepare()
    df.to_parquet("data/corpus.parquet", index=False)
    print("\nSaved to data/corpus.parquet")
