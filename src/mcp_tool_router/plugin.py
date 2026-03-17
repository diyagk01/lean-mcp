from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .client import Client
from .types import ContextPayload, RoutingDecision


@dataclass
class Plugin:
    """
    Optional wrapper for integration points.

    This SDK intentionally does not call LLMs or execute tools; this wrapper just
    exposes a single "route + build context" surface for downstream code.
    """

    client: Client

    def route_and_build(
        self,
        user_query: str,
        *,
        top_k: Optional[int] = None,
        tool_top_n: Optional[int] = None,
        resource_top_k: Optional[int] = None,
        resource_server_filter: Optional[str] = None,
    ) -> tuple[RoutingDecision, ContextPayload]:
        decision = self.client.route(user_query, top_k=top_k)
        payload = self.client.build_llm_context(
            user_query=user_query,
            decision=decision,
            tool_top_n=tool_top_n,
            resource_top_k=resource_top_k,
            resource_server_filter=resource_server_filter,
        )
        return decision, payload

