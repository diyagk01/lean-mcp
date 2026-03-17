from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional, Protocol

from .util import tokenize


class EmbeddingProvider(Protocol):
    dim: int

    def embed_text(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class HashingEmbeddingProvider:
    """
    Deterministic local embedding (no network calls).

    Uses signed feature hashing over tokens into a fixed-size dense vector,
    followed by L2 normalization. Good enough for routing tests and small catalogs.
    """

    dim: int = 384

    def embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = tokenize(text)
        if not toks:
            return vec

        for t in toks:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = -1.0 if (h[4] & 1) else 1.0
            # weight by a tiny log factor to reduce repetition dominance
            vec[idx] += sign * (1.0 + math.log1p(len(t)))

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


@dataclass
class SentenceTransformersEmbeddingProvider:
    """
    Higher-accuracy semantic embeddings via sentence-transformers (optional dependency).

    You must install:
      pip install -e ".[embeddings]"

    You must choose a model name (no defaults are hardcoded here beyond a required parameter).
    """

    model_name: str
    device: Optional[str] = None
    normalize: bool = True
    _model: object | None = None

    @property
    def dim(self) -> int:
        # Not known until model is loaded; kept for interface compatibility.
        # Most of the SDK doesn't rely on dim being present at init time.
        return 0

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. "
                'Install with: pip install -e ".[embeddings]"'
            ) from e

        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        model = self._get_model()
        vec = model.encode([text], normalize_embeddings=self.normalize)[0]
        # vec may be numpy; ensure plain Python floats
        return [float(x) for x in vec]

