from __future__ import annotations

from typing import List, Optional

from .assembler import ContextAssembler
from .embedding import EmbeddingProvider, FastEmbedProvider
from .resource import ResourceRetriever
from .retriever import ToolRetriever
from .types import AssembledContext, McpTool
from .validator import ToolValidator
from .vector_store import SQLiteVectorStore


class RagMcpClient:
    def __init__(
        self,
        *,
        top_k: int = 5,
        inject_top_n: int = 1,
        validate: bool = False,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[SQLiteVectorStore] = None,
        db_path: str = ":memory:",
    ) -> None:
        emb = embedding_provider or FastEmbedProvider()
        store = vector_store or SQLiteVectorStore(db_path=db_path)
        self.retriever = ToolRetriever(emb, store, top_k=top_k)
        self.resource_retriever = ResourceRetriever(emb, store)
        self.assembler = ContextAssembler(inject_top_n=inject_top_n)
        self.validator = ToolValidator()
        self.validate = validate

    def add_tools(self, tools: List[McpTool]) -> None:
        self.retriever.add_tools(tools)

    def build_context(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        resource_chunks: Optional[List[str]] = None,
    ) -> AssembledContext:
        retrieved = self.retriever.retrieve(query=query, top_k=top_k)
        if self.validate:
            retrieved = self.validator.filter(retrieved, query)
        return self.assembler.assemble(
            query=query,
            retrieved_tools=retrieved,
            resource_chunks=resource_chunks,
        )
