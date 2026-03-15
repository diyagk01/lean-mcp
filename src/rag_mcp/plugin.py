from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .client import RagMcpClient
from .sync import DescriptorSyncer, ResourceContentFetcher, ResourceSyncReport, SyncReport
from .types import AssembledContext, McpTool, RetrievedTool


@dataclass(frozen=True)
class RoutingDecision:
    best_tool: Optional[McpTool]
    candidates: List[RetrievedTool]
    confidence: float
    margin: float


class RagMcpPlugin:
    """
    Installable server-side plugin interface:
    - register_tools()
    - route_query()
    - assemble_for_llm()
    """

    def __init__(
        self,
        *,
        server_name: str,
        top_k: int = 5,
        confidence_threshold: float = 0.15,
        margin_threshold: float = 0.05,
        db_path: str = ":memory:",
    ) -> None:
        self.server_name = server_name
        self.client = RagMcpClient(top_k=top_k, db_path=db_path, inject_top_n=1)
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

    def register_tools(self, tools: List[McpTool]) -> None:
        scoped = [
            McpTool(
                name=t.name,
                description=t.description,
                input_schema=t.input_schema,
                server=t.server or self.server_name,
                metadata=t.metadata,
                version=t.version,
            )
            for t in tools
        ]
        self.client.add_tools(scoped)

    def sync_tools_from_descriptor_root(
        self,
        descriptor_root: str,
        *,
        servers: Optional[List[str]] = None,
        delete_stale: bool = False,
    ) -> SyncReport:
        syncer = DescriptorSyncer(client=self.client)
        return syncer.sync_tools_from_directory(
            descriptor_root,
            servers=servers,
            delete_stale=delete_stale,
        )

    def sync_resources_from_descriptor_root(
        self,
        descriptor_root: str,
        *,
        servers: Optional[List[str]] = None,
        delete_stale: bool = False,
        include_raw_descriptor: bool = False,
        chunk_size: int = 800,
        fetch_content: bool = False,
        resource_fetcher: Optional[ResourceContentFetcher] = None,
        max_fetched_content_chars: int = 100_000,
    ) -> ResourceSyncReport:
        syncer = DescriptorSyncer(client=self.client)
        return syncer.sync_resources_from_directory(
            descriptor_root,
            servers=servers,
            delete_stale=delete_stale,
            include_raw_descriptor=include_raw_descriptor,
            chunk_size=chunk_size,
            fetch_content=fetch_content,
            resource_fetcher=resource_fetcher,
            max_fetched_content_chars=max_fetched_content_chars,
        )

    def retrieve_resource_context(
        self,
        query: str,
        *,
        top_k: int = 4,
        server: Optional[str] = None,
    ) -> List[str]:
        return self.client.resource_retriever.retrieve(query, top_k=top_k, server=server)

    def route_query(self, query: str, top_k: Optional[int] = None) -> RoutingDecision:
        candidates = self.client.retriever.retrieve(query=query, top_k=top_k)
        if not candidates:
            return RoutingDecision(
                best_tool=None,
                candidates=[],
                confidence=0.0,
                margin=0.0,
            )
        top1 = candidates[0]
        top2 = candidates[1] if len(candidates) > 1 else None
        margin = top1.score - top2.score if top2 else top1.score
        confidence = top1.score
        # If there is only one candidate, select it directly. Threshold-based
        # abstention is mainly useful for ambiguous multi-candidate retrieval.
        if top2 is None:
            return RoutingDecision(
                best_tool=top1.tool,
                candidates=candidates,
                confidence=confidence,
                margin=margin,
            )
        if confidence < self.confidence_threshold and margin < self.margin_threshold:
            return RoutingDecision(
                best_tool=None,
                candidates=candidates,
                confidence=confidence,
                margin=margin,
            )
        return RoutingDecision(
            best_tool=top1.tool,
            candidates=candidates,
            confidence=confidence,
            margin=margin,
        )

    def assemble_for_llm(
        self,
        query: str,
        *,
        routing: Optional[RoutingDecision] = None,
        resource_chunks: Optional[List[str]] = None,
    ) -> AssembledContext:
        if routing is None:
            routing = self.route_query(query)
        retrieved = routing.candidates
        if routing.best_tool is not None:
            retrieved = [r for r in retrieved if r.tool.name == routing.best_tool.name]
        return self.client.assembler.assemble(
            query=query,
            retrieved_tools=retrieved,
            resource_chunks=resource_chunks,
        )
