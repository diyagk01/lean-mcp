from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypedDict


Json = dict[str, Any]


@dataclass(frozen=True)
class Tool:
    server: str
    name: str
    description: str
    input_schema: Json
    version: Optional[str] = None
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class Resource:
    server: str
    uri: str
    name: str
    description: str = ""
    mime_type: Optional[str] = None
    version: Optional[str] = None
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCandidate:
    tool_id: str
    score: float
    server: str
    name: str
    version: Optional[str]


@dataclass(frozen=True)
class RoutingDecision:
    user_query: str
    best_tool: Optional[ToolCandidate]
    candidates: list[ToolCandidate]
    confidence: float
    margin: float
    abstained: bool


class ContextPayload(TypedDict):
    tools: list[Json]
    system_context: str
    token_estimate: int


ResourceContentFetcher = Callable[[str], str]


class RoutingTrace(TypedDict, total=False):
    """
    Lightweight, serializable explanation of a routing decision.

    This is meant for logging / observability:
      - which thresholds were in effect
      - why the router abstained or selected
      - what the top candidates looked like
    """

    user_query: str
    server_filter: Optional[str]
    top_k: int
    min_confidence: float
    min_margin: float
    reason: str  # e.g. "selected", "abstained_low_confidence", "abstained_low_margin"
    confidence: float
    margin: float
    best_tool: Optional[Json]
    candidates: list[Json]
    original_candidates: list[Json]
    rerank: Json
