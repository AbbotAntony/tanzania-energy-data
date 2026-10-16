"""Sentence-embedding helpers (CPU only).

all-MiniLM-L6-v2 reads at most 256 word pieces (~200 words), but articles
average ~600 words. Each article is therefore split into ~150-word chunks;
the chunk embeddings are averaged and re-normalised, so the whole article
counts, not just its opening.
"""

from __future__ import annotations

import re

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"   # pinned Hub commit
CHUNK_WORDS = 150


def chunk_words(text: str, size: int = CHUNK_WORDS) -> list[str]:
    """Split on sentence ends into chunks of about ``size`` words (never empty)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur, n = [], [], 0
    for s in sentences:
        w = len(s.split())
        if cur and n + w > size:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [""]


def mean_pool(chunk_emb: np.ndarray, owner: np.ndarray, n_docs: int) -> np.ndarray:
    """Average chunk rows per document (``owner`` = doc index of each chunk), unit length."""
    out = np.zeros((n_docs, chunk_emb.shape[1]))
    np.add.at(out, owner, chunk_emb)
    out /= np.bincount(owner, minlength=n_docs)[:, None]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def embed_documents(texts: list[str], model_name: str = MODEL_NAME, batch_size: int = 64) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    model = SentenceTransformer(model_name, revision=MODEL_REVISION, device="cpu")
    chunks, owner = [], []
    for i, t in enumerate(texts):
        cs = chunk_words(t)
        chunks.extend(cs)
        owner.extend([i] * len(cs))
    emb = model.encode(chunks, batch_size=batch_size, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    return mean_pool(emb, np.array(owner), len(texts))
