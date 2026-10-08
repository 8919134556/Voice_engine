"""
cosine.py — compare two embeddings by the angle between them.

    cosine_similarity = (a · b) / (|a| * |b|)

     1.0  -> pointing in exactly the same direction ("very similar")
     0.0  -> at 90 degrees (unrelated)
    -1.0  -> pointing in opposite directions

Only the direction matters, not the length. For L2-normalized vectors
(length 1) it is simply the dot product.
"""

from itertools import combinations

import numpy as np


def cosine_similarity(embedding_a, embedding_b):
    a = np.asarray(embedding_a, dtype=np.float64).ravel()
    b = np.asarray(embedding_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Embeddings must have the same shape: {a.shape} vs {b.shape}")
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def pairwise_similarities(embeddings):
    """
    embeddings: dict {name: vector}
    returns: list of (name_a, name_b, similarity) for every pair, e.g.
             audio_001 vs audio_002, audio_001 vs audio_003, audio_002 vs audio_003
    """
    return [
        (name_a, name_b, cosine_similarity(embeddings[name_a], embeddings[name_b]))
        for name_a, name_b in combinations(sorted(embeddings), 2)
    ]
