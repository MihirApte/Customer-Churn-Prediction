"""Semantic search over the playbook using sentence embeddings and cosine similarity."""

from typing import Callable, NamedTuple

import numpy as np

from ..config import EMBEDDING_MODEL_NAME

# An embedder maps a list of texts to a 2-D array, one row (vector) per text.
Embedder = Callable[[list], np.ndarray]


class Hit(NamedTuple):
    entry: object             # PlaybookEntry
    score: float              # cosine similarity, higher = closer in meaning
    matched_on: tuple = ()    # which driver(s) / the combined query found this entry


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. Loaded on first use so that importing
    this module (and running tests with a fake embedder) stays fast."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def __call__(self, texts: list) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so a dot product equals cosine similarity."""
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class Retriever:
    """Embeds every playbook entry once, then finds the entries closest to a query.

    With a few dozen entries an exact NumPy dot product is instant. If the
    playbook grew to thousands of entries, only this class would change (for
    example to FAISS or a vector database); callers would not.
    """

    def __init__(self, entries: list, embed: Embedder):
        if not entries:
            raise ValueError("Retriever needs at least one playbook entry")
        self.entries = list(entries)
        self._embed = embed
        self._matrix = _normalise(embed([e.retrieval_text for e in self.entries]))

    def retrieve(self, query: str, k: int = 3, allowed=None) -> list:
        """The k entries most similar in meaning to `query`, best first.

        `allowed` is an optional function entry -> bool. Entries it rejects are
        never returned, so metadata can rule out entries that are close in wording
        but wrong for the customer. Returns fewer than k (possibly none) if fewer
        entries are allowed.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        q = _normalise(self._embed([query]))[0]
        scores = self._matrix @ q
        order = np.argsort(scores)[::-1]
        if allowed is not None:
            order = [i for i in order if allowed(self.entries[i])]
        return [Hit(self.entries[i], float(scores[i])) for i in list(order)[:k]]
