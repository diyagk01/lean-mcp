from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    model_name: str = "unknown"

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


class HashEmbeddingProvider(EmbeddingProvider):
    """
    Zero-dependency fallback embedding provider.
    Produces deterministic vectors from token hashes.
    """

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension
        self.model_name = f"hash-{dimension}"

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]


class FastEmbedProvider(EmbeddingProvider):
    """
    Local ONNX embeddings via fastembed.
    Falls back to HashEmbeddingProvider if fastembed is not installed.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        fallback_dimension: int = 256,
    ) -> None:
        self.model_name = model_name
        self._fallback = HashEmbeddingProvider(dimension=fallback_dimension)
        self._model = None
        try:
            from fastembed import TextEmbedding  # type: ignore

            self._model = TextEmbedding(model_name=model_name)
        except Exception:
            self._model = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            return self._fallback.embed_texts(texts)
        return [list(v) for v in self._model.embed(texts)]
