from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .context import ContextBuilder
from .embeddings import EmbeddingProvider, HashingEmbeddingProvider
from .router import RouterConfig, ToolRouter
from .store import SQLiteStore
from .sync import DescriptorSync
from .types import ContextPayload, ResourceContentFetcher, RoutingDecision, RoutingTrace


@dataclass
class Client:
    db_path: str = "tool_router.sqlite3"
    embedder: Optional[EmbeddingProvider] = None
    config: RouterConfig = RouterConfig()

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = HashingEmbeddingProvider()
        self.store = SQLiteStore(self.db_path)
        self.router = ToolRouter(self.store, self.embedder, self.config)
        self.context_builder = ContextBuilder(self.store, self.embedder)
        self.syncer = DescriptorSync(
            store=self.store,
            embedder=self.embedder,
            resource_chunk_size=self.config.resource_chunk_size,
        )

    def sync_descriptors(
        self,
        *,
        root: str,
        delete_stale: bool = False,
        fetch_resource_content: Optional[ResourceContentFetcher] = None,
    ) -> dict[str, int]:
        return self.syncer.sync_from_root(
            root,
            delete_stale=delete_stale,
            fetch_resource_content=fetch_resource_content,
        )

    def route(
        self,
        user_query: str,
        *,
        top_k: Optional[int] = None,
        server_filter: Optional[str] = None,
        min_confidence: Optional[float] = None,
        min_margin: Optional[float] = None,
    ) -> RoutingDecision:
        return self.router.route(
            user_query,
            top_k=top_k,
            server_filter=server_filter,
            min_confidence=min_confidence,
            min_margin=min_margin,
        )

    def route_with_trace(
        self,
        user_query: str,
        *,
        top_k: Optional[int] = None,
        server_filter: Optional[str] = None,
        min_confidence: Optional[float] = None,
        min_margin: Optional[float] = None,
    ) -> tuple[RoutingDecision, RoutingTrace]:
        """
        Same as route(), but also returns a lightweight explanation object.

        Useful for logging / debugging:
          - which thresholds were used
          - why we abstained or selected
          - the top candidates and their scores
        """
        eff_top_k = self.config.top_k if top_k is None else int(top_k)
        eff_min_conf = (
            self.config.min_confidence if min_confidence is None else float(min_confidence)
        )
        eff_min_margin = self.config.min_margin if min_margin is None else float(min_margin)

        decision, trace = self.router.route_with_trace(
            user_query,
            top_k=eff_top_k,
            server_filter=server_filter,
            min_confidence=eff_min_conf,
            min_margin=eff_min_margin,
        )
        return decision, trace

    def build_llm_context(
        self,
        *,
        user_query: str,
        decision: RoutingDecision,
        tool_top_n: Optional[int] = None,
        resource_top_k: Optional[int] = None,
        resource_server_filter: Optional[str] = None,
    ) -> ContextPayload:
        return self.context_builder.build(
            user_query=user_query,
            decision=decision,
            tool_top_n=self.config.tool_top_n if tool_top_n is None else int(tool_top_n),
            resource_top_k=self.config.resource_top_k if resource_top_k is None else int(resource_top_k),
            resource_server_filter=resource_server_filter,
        )

