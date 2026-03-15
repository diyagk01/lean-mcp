from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    server: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: Optional[str] = None


@dataclass(frozen=True)
class McpResource:
    id: str
    content: str
    uri: Optional[str] = None
    server: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedTool:
    tool: McpTool
    score: float


@dataclass(frozen=True)
class AssembledContext:
    tools: List[Dict[str, Any]]
    system_context: str
    retrieved_tools: List[RetrievedTool]
    token_estimate: int
