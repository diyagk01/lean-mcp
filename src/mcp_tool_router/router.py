from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .embeddings import EmbeddingProvider
from .rerank import RerankWeights, rerank_candidates
from .store import SQLiteStore
from .types import RoutingDecision, ToolCandidate, RoutingTrace


@dataclass(frozen=True)
class RouterConfig:
    top_k: int = 8
    min_confidence: float = 0.22
    min_margin: float = 0.04
    tool_top_n: int = 1
    resource_top_k: int = 4
    resource_chunk_size: int = 1200
    rerank_enabled: bool = True
    rerank_weights: RerankWeights = RerankWeights()


@dataclass
class ToolRouter:
    store: SQLiteStore
    embedder: EmbeddingProvider
    config: RouterConfig = RouterConfig()

    def route(
        self,
        user_query: str,
        *,
        top_k: Optional[int] = None,
        server_filter: Optional[str] = None,
        min_confidence: Optional[float] = None,
        min_margin: Optional[float] = None,
    ) -> RoutingDecision:
        decision, _trace = self.route_with_trace(
            user_query,
            top_k=top_k,
            server_filter=server_filter,
            min_confidence=min_confidence,
            min_margin=min_margin,
        )
        return decision

    def route_with_trace(
        self,
        user_query: str,
        *,
        top_k: Optional[int] = None,
        server_filter: Optional[str] = None,
        min_confidence: Optional[float] = None,
        min_margin: Optional[float] = None,
    ) -> tuple[RoutingDecision, RoutingTrace]:
        self.store.init()
        q_emb = self.embedder.embed_text(user_query)
        k = self.config.top_k if top_k is None else int(top_k)
        # When reranking is enabled, pull a larger pool than top_k so we can
        # recover near-duplicate tools that may sit just outside top_k.
        pool_k = k
        if self.config.rerank_enabled:
            pool_k = max(k, min(50, k * 3))
        base_candidates = self.store.query_tools(q_emb, top_k=pool_k, server_filter=server_filter)

        candidates = base_candidates
        rerank_trace = None
        if self.config.rerank_enabled and len(base_candidates) >= 2:
            tool_ids = [c.tool_id for c in base_candidates]
            payloads = self.store.get_tool_payloads_json(tool_ids)
            base_dicts = [
                {
                    "tool_id": c.tool_id,
                    "server": c.server,
                    "name": c.name,
                    "version": c.version,
                    "score": float(c.score),
                }
                for c in base_candidates
            ]
            reranked, rerank_trace = rerank_candidates(
                user_query=user_query,
                base_candidates=base_dicts,
                tool_payloads=payloads,
                weights=self.config.rerank_weights,
            )
            # Convert reranked dicts back into ToolCandidate with final_score as score.
            candidates = [
                ToolCandidate(
                    tool_id=str(r["tool_id"]),
                    score=float(r["final_score"]),
                    server=str(r["server"]),
                    name=str(r["name"]),
                    version=None if r.get("version") is None else str(r.get("version")),
                )
                for r in reranked
            ]
            # Keep only the requested top_k for downstream selection/output.
            candidates = candidates[:k]

        best: ToolCandidate | None = None
        confidence = 0.0
        margin = 0.0

        if candidates:
            best = candidates[0]
            confidence = float(best.score)
            if len(candidates) >= 2:
                margin = float(candidates[0].score - candidates[1].score)

        c_thr = self.config.min_confidence if min_confidence is None else float(min_confidence)
        m_thr = self.config.min_margin if min_margin is None else float(min_margin)

        abstained = True
        if best is not None and confidence >= c_thr and margin >= m_thr:
            abstained = False
        else:
            best = None

        decision = RoutingDecision(
            user_query=user_query,
            best_tool=best,
            candidates=candidates,
            confidence=confidence,
            margin=margin,
            abstained=abstained,
        )
        trace: RoutingTrace = {
            "user_query": user_query,
            "server_filter": server_filter,
            "top_k": k,
            "min_confidence": c_thr,
            "min_margin": m_thr,
            "confidence": confidence,
            "margin": margin,
            "reason": "selected" if not abstained else ("abstained_low_confidence" if confidence < c_thr else "abstained_low_margin" if margin < m_thr else "abstained_other"),
            "original_candidates": [
                {
                    "tool_id": c.tool_id,
                    "server": c.server,
                    "name": c.name,
                    "version": c.version,
                    "score": float(c.score),
                }
                for c in base_candidates
            ],
            "candidates": [
                {
                    "tool_id": c.tool_id,
                    "server": c.server,
                    "name": c.name,
                    "version": c.version,
                    "score": float(c.score),
                }
                for c in candidates
            ],
            "rerank": rerank_trace or {},
        }
        if best is not None:
            trace["best_tool"] = {
                "tool_id": best.tool_id,
                "server": best.server,
                "name": best.name,
                "version": best.version,
                "score": float(best.score),
            }
        else:
            trace["best_tool"] = None

        return decision, trace

