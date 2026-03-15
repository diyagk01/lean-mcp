from __future__ import annotations

from typing import Callable, List, Optional

from .types import RetrievedTool


class ToolValidator:
    """
    Lightweight optional validation.
    You can inject a custom callback for stronger checks.
    """

    def __init__(
        self,
        checker: Optional[Callable[[str, RetrievedTool], bool]] = None,
        max_candidates: int = 5,
    ) -> None:
        self.checker = checker
        self.max_candidates = max_candidates

    def filter(self, retrieved_tools: List[RetrievedTool], query: str) -> List[RetrievedTool]:
        limited = retrieved_tools[: self.max_candidates]
        if self.checker is None:
            return limited
        return [tool for tool in limited if self.checker(query, tool)]
