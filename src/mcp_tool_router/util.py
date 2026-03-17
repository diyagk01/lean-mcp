from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable


_token_re = re.compile(r"[A-Za-z0-9_]+")


def stable_tool_id(server: str, name: str, version: str | None) -> str:
    v = version or "0"
    return f"{server}:{name}:{v}"


def stable_resource_id(server: str, uri: str, version: str | None) -> str:
    v = version or "0"
    return f"{server}:{uri}:{v}"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(payload: Any) -> str:
    return sha256_text(canonical_json(payload))


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _token_re.finditer(text)]


def l2_norm(vec: Iterable[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector length mismatch")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def rough_token_estimate(text: str) -> int:
    # Simple heuristic: ~4 chars/token in English; clamp at least 1 if non-empty.
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)

