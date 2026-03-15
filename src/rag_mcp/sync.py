from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from .client import RagMcpClient
from .types import McpResource, McpTool

ResourceContentFetcher = Callable[[str, str], Optional[str]]


@dataclass(frozen=True)
class SyncReport:
    discovered_servers: int
    discovered_tools: int
    indexed_tools: int
    added: int
    updated: int
    skipped: int
    removed: int
    errors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResourceSyncReport:
    discovered_servers: int
    discovered_resources: int
    indexed_resources: int
    fetched_payloads: int
    added: int
    updated: int
    skipped: int
    removed: int
    errors: List[str] = field(default_factory=list)


class DescriptorSyncer:
    """
    Sync MCP tool descriptor JSON files into the rag_mcp vector index.

    Expected layout:
      <descriptor_root>/<server_name>/tools/<tool_name>.json
    """

    def __init__(self, client: RagMcpClient, namespace: str = "tools") -> None:
        self.client = client
        self.namespace = namespace

    def sync_tools_from_directory(
        self,
        descriptor_root: str,
        *,
        servers: Optional[List[str]] = None,
        delete_stale: bool = False,
    ) -> SyncReport:
        root = Path(descriptor_root)
        allowlist: Optional[Set[str]] = set(servers) if servers else None
        discovered: List[McpTool] = []
        errors: List[str] = []
        discovered_servers: Set[str] = set()

        if not root.exists():
            return SyncReport(
                discovered_servers=0,
                discovered_tools=0,
                indexed_tools=0,
                added=0,
                updated=0,
                skipped=0,
                removed=0,
                errors=[f"descriptor root does not exist: {descriptor_root}"],
            )

        for server_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            server_name = server_dir.name
            if allowlist is not None and server_name not in allowlist:
                continue
            tools_dir = server_dir / "tools"
            if not tools_dir.is_dir():
                continue
            discovered_servers.add(server_name)
            for tool_file in sorted(tools_dir.glob("*.json")):
                try:
                    discovered.append(_tool_from_descriptor(server_name, tool_file))
                except Exception as exc:  # pragma: no cover
                    errors.append(f"failed to parse {tool_file}: {exc}")

        if not discovered:
            return SyncReport(
                discovered_servers=len(discovered_servers),
                discovered_tools=0,
                indexed_tools=0,
                added=0,
                updated=0,
                skipped=0,
                removed=0,
                errors=errors,
            )

        retriever = self.client.retriever
        texts = [self._tool_text(t) for t in discovered]
        vectors = retriever.embedding_provider.embed_texts(texts)

        added = 0
        updated = 0
        skipped = 0
        indexed_ids: Set[str] = set()

        for tool, vector, text in zip(discovered, vectors, texts):
            tool_id = self._tool_id(tool)
            indexed_ids.add(tool_id)
            tool_hash = self._tool_hash(tool)
            existing = retriever.vector_store.get_payload(self.namespace, tool_id)
            if existing and existing.get("tool_hash") == tool_hash:
                skipped += 1
                continue

            payload: Dict[str, object] = {
                "tool": _serialize_tool(tool),
                "tool_text": text,
                "tool_hash": tool_hash,
                "embedding_model": retriever.embedding_provider.model_name,
                "source": "descriptor_sync",
            }
            retriever.vector_store.upsert(self.namespace, tool_id, vector, payload)
            if existing is None:
                added += 1
            else:
                updated += 1

        removed = 0
        if delete_stale:
            for item_id, payload in retriever.vector_store.list_items(self.namespace):
                raw_tool = payload.get("tool")
                if not isinstance(raw_tool, dict):
                    continue
                server_name = raw_tool.get("server")
                if allowlist is not None and server_name not in allowlist:
                    continue
                if allowlist is None and server_name not in discovered_servers:
                    continue
                if item_id in indexed_ids:
                    continue
                retriever.vector_store.delete(self.namespace, item_id)
                removed += 1

        return SyncReport(
            discovered_servers=len(discovered_servers),
            discovered_tools=len(discovered),
            indexed_tools=added + updated + skipped,
            added=added,
            updated=updated,
            skipped=skipped,
            removed=removed,
            errors=errors,
        )

    def _tool_id(self, tool: McpTool) -> str:
        version = tool.version or "0"
        server = tool.server or "default"
        return f"{server}:{tool.name}:{version}"

    def _tool_hash(self, tool: McpTool) -> str:
        canonical = {
            "name": tool.name,
            "description": tool.description,
            "server": tool.server,
            "version": tool.version,
            "metadata": tool.metadata,
            "input_schema": tool.input_schema,
        }
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _tool_text(self, tool: McpTool) -> str:
        tags = tool.metadata.get("tags", [])
        tags_text = ",".join(tags) if isinstance(tags, list) else ""
        return f"{tool.server or 'default'} | {tool.name} | {tool.description} | tags:{tags_text}"

    def sync_resources_from_directory(
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
        root = Path(descriptor_root)
        allowlist: Optional[Set[str]] = set(servers) if servers else None
        discovered: List[McpResource] = []
        errors: List[str] = []
        discovered_servers: Set[str] = set()

        if not root.exists():
            return ResourceSyncReport(
                discovered_servers=0,
                discovered_resources=0,
                indexed_resources=0,
                fetched_payloads=0,
                added=0,
                updated=0,
                skipped=0,
                removed=0,
                errors=[f"descriptor root does not exist: {descriptor_root}"],
            )

        for server_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            server_name = server_dir.name
            if allowlist is not None and server_name not in allowlist:
                continue
            resources_dir = server_dir / "resources"
            if not resources_dir.is_dir():
                continue
            discovered_servers.add(server_name)
            for resource_file in sorted(resources_dir.glob("*.json")):
                try:
                    discovered.append(
                        _resource_from_descriptor(
                            server_name,
                            resource_file,
                            include_raw_descriptor=include_raw_descriptor,
                            fetch_content=fetch_content,
                            resource_fetcher=resource_fetcher,
                            max_fetched_content_chars=max_fetched_content_chars,
                        )
                    )
                except Exception as exc:  # pragma: no cover
                    errors.append(f"failed to parse {resource_file}: {exc}")

        if not discovered:
            return ResourceSyncReport(
                discovered_servers=len(discovered_servers),
                discovered_resources=0,
                indexed_resources=0,
                fetched_payloads=0,
                added=0,
                updated=0,
                skipped=0,
                removed=0,
                errors=errors,
            )

        retriever = self.client.resource_retriever

        chunk_ids: List[str] = []
        chunk_texts: List[str] = []
        chunk_payloads: List[Dict[str, object]] = []
        for resource in discovered:
            resource_hash = _resource_hash(resource)
            parts = _chunk_text(resource.content, chunk_size=chunk_size)
            for idx, part in enumerate(parts):
                chunk_id = f"{resource.id}:{idx}"
                chunk_ids.append(chunk_id)
                chunk_texts.append(part)
                chunk_payloads.append(
                    {
                        "resource_id": resource.id,
                        "server": resource.server,
                        "uri": resource.uri,
                        "chunk_index": idx,
                        "text": part,
                        "resource_hash": resource_hash,
                        "embedding_model": retriever.embedding_provider.model_name,
                        "source": "descriptor_sync_resource",
                        "fetched_payload": bool(resource.metadata.get("fetched_payload")),
                    }
                )

        vectors = retriever.embedding_provider.embed_texts(chunk_texts) if chunk_texts else []
        added = 0
        updated = 0
        skipped = 0
        fetched_payloads = 0
        indexed_ids: Set[str] = set()
        indexed_resource_ids: Set[str] = set()

        for chunk_id, vector, payload in zip(chunk_ids, vectors, chunk_payloads):
            indexed_ids.add(chunk_id)
            resource_id = payload["resource_id"]
            if isinstance(resource_id, str):
                indexed_resource_ids.add(resource_id)
            if payload.get("fetched_payload"):
                fetched_payloads += 1
            existing = retriever.vector_store.get_payload(retriever.namespace, chunk_id)
            if (
                existing
                and existing.get("resource_hash") == payload.get("resource_hash")
                and existing.get("text") == payload.get("text")
            ):
                skipped += 1
                continue

            retriever.vector_store.upsert(retriever.namespace, chunk_id, vector, payload)
            if existing is None:
                added += 1
            else:
                updated += 1

        removed = 0
        if delete_stale:
            for item_id, payload in retriever.vector_store.list_items(retriever.namespace):
                server_name = payload.get("server")
                if allowlist is not None and server_name not in allowlist:
                    continue
                if allowlist is None and server_name not in discovered_servers:
                    continue
                resource_id = payload.get("resource_id")
                if not isinstance(resource_id, str):
                    continue
                if resource_id in indexed_resource_ids and item_id in indexed_ids:
                    continue
                retriever.vector_store.delete(retriever.namespace, item_id)
                removed += 1

        return ResourceSyncReport(
            discovered_servers=len(discovered_servers),
            discovered_resources=len(discovered),
            indexed_resources=added + updated + skipped,
            fetched_payloads=fetched_payloads,
            added=added,
            updated=updated,
            skipped=skipped,
            removed=removed,
            errors=errors,
        )


def _tool_from_descriptor(server_name: str, tool_file: Path) -> McpTool:
    raw = json.loads(tool_file.read_text(encoding="utf-8"))
    name = str(raw.get("name", tool_file.stem))
    description = str(raw.get("description", "")).strip() or f"{name} tool"
    input_schema = raw.get("arguments") or raw.get("input_schema") or {
        "type": "object",
        "properties": {},
    }
    metadata: Dict[str, object] = {"descriptor_file": str(tool_file)}
    if "outputSchema" in raw:
        metadata["output_schema"] = raw["outputSchema"]
    return McpTool(
        name=name,
        description=description,
        input_schema=dict(input_schema),  # type: ignore[arg-type]
        server=server_name,
        metadata=metadata,
        version=raw.get("version"),  # type: ignore[arg-type]
    )


def _serialize_tool(tool: McpTool) -> Dict[str, object]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "server": tool.server,
        "metadata": tool.metadata,
        "version": tool.version,
    }


def _resource_from_descriptor(
    server_name: str,
    resource_file: Path,
    *,
    include_raw_descriptor: bool = False,
    fetch_content: bool = False,
    resource_fetcher: Optional[ResourceContentFetcher] = None,
    max_fetched_content_chars: int = 100_000,
) -> McpResource:
    raw = json.loads(resource_file.read_text(encoding="utf-8"))
    name = str(raw.get("name", resource_file.stem))
    uri = raw.get("uri")
    description = str(raw.get("description", "")).strip()
    mime_type = str(raw.get("mimeType", "")).strip()
    fetched_content: Optional[str] = None
    fetched_payload = False
    if fetch_content and isinstance(uri, str) and resource_fetcher is not None:
        maybe_content = resource_fetcher(uri, server_name)
        if maybe_content:
            fetched_payload = True
            fetched_content = maybe_content[:max_fetched_content_chars]
    content = _build_resource_content(
        name=name,
        server_name=server_name,
        uri=uri if isinstance(uri, str) else None,
        description=description,
        mime_type=mime_type,
        raw_descriptor=raw,
        include_raw_descriptor=include_raw_descriptor,
        fetched_content=fetched_content,
    )

    return McpResource(
        id=f"{server_name}:{name}",
        content=content,
        uri=uri if isinstance(uri, str) else None,
        server=server_name,
        metadata={
            "descriptor_file": str(resource_file),
            "mimeType": mime_type,
            "fetched_payload": fetched_payload,
        },
    )


def _resource_hash(resource: McpResource) -> str:
    canonical = {
        "id": resource.id,
        "content": resource.content,
        "uri": resource.uri,
        "server": resource.server,
        "metadata": resource.metadata,
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()


def _chunk_text(text: str, chunk_size: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > chunk_size and cur:
            chunks.append(" ".join(cur))
            cur = []
            cur_len = 0
        cur.append(w)
        cur_len += len(w) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _build_resource_content(
    *,
    name: str,
    server_name: str,
    uri: Optional[str],
    description: str,
    mime_type: str,
    raw_descriptor: Dict[str, object],
    include_raw_descriptor: bool,
    fetched_content: Optional[str],
) -> str:
    content_lines = [
        f"resource_name: {name}",
        f"server: {server_name}",
    ]
    if uri:
        content_lines.append(f"uri: {uri}")
    if description:
        content_lines.append(f"description: {description}")
    if mime_type:
        content_lines.append(f"mime_type: {mime_type}")
    if include_raw_descriptor:
        content_lines.append("raw_descriptor_json:")
        content_lines.append(json.dumps(raw_descriptor, sort_keys=True))
    if fetched_content:
        content_lines.append("resource_payload:")
        content_lines.append(fetched_content)
    return "\n".join(content_lines)
