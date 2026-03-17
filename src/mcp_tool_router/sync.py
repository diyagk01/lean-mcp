from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .embeddings import EmbeddingProvider
from .store import SQLiteStore
from .types import Resource, ResourceContentFetcher, Tool
from .util import (
    canonical_json,
    content_hash,
    sha256_text,
    stable_resource_id,
    stable_tool_id,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_name_tokens(name: str) -> list[str]:
    # Simple tokenization for snake_case and kebab-case, plus collapsing whitespace.
    out: list[str] = []
    for part in name.replace("-", "_").replace(" ", "_").split("_"):
        part = part.strip()
        if part:
            out.append(part.lower())
    return out


def _infer_operation(tokens: list[str]) -> str:
    # Prefer leading verb, but allow common verb anywhere.
    verbs = [
        "read",
        "get",
        "fetch",
        "list",
        "search",
        "find",
        "query",
        "create",
        "add",
        "new",
        "update",
        "edit",
        "patch",
        "set",
        "write",
        "put",
        "delete",
        "remove",
        "drop",
        "move",
        "rename",
        "copy",
    ]
    if tokens:
        if tokens[0] in verbs:
            return tokens[0]
    for t in tokens:
        if t in verbs:
            return t
    return "unknown"


def _infer_side_effects(operation: str) -> str:
    # Heuristic; used only to add disambiguating text (not enforced).
    mutating = {"create", "add", "new", "update", "edit", "patch", "set", "write", "put", "delete", "remove", "drop", "move", "rename", "copy"}
    if operation in mutating:
        return "yes"
    if operation == "unknown":
        return "unknown"
    return "no"


def _infer_object_tokens(tokens: list[str]) -> list[str]:
    stop = {
        "read",
        "get",
        "fetch",
        "list",
        "search",
        "find",
        "query",
        "create",
        "add",
        "new",
        "update",
        "edit",
        "patch",
        "set",
        "write",
        "put",
        "delete",
        "remove",
        "drop",
        "move",
        "rename",
        "copy",
        "tool",
        "mcp",
    }
    return [t for t in tokens if t not in stop]


def _operation_synonyms(operation: str, objects: list[str]) -> list[str]:
    """
    Adds common user phrasing to help near-duplicate tool disambiguation.
    Heuristic only (no per-tool hardcoding).
    """
    base: dict[str, list[str]] = {
        "read": ["read", "open", "view", "show", "display", "contents", "lines"],
        "get": ["get", "fetch", "retrieve", "show"],
        "fetch": ["fetch", "retrieve", "download"],
        "list": ["list", "ls", "browse", "show files", "show folders", "directory contents"],
        "search": ["search", "find", "locate", "grep", "match", "pattern", "glob"],
        "find": ["find", "locate", "search"],
        "query": ["query", "search", "filter"],
        "create": ["create", "make", "new", "add"],
        "add": ["add", "create", "insert"],
        "update": ["update", "edit", "modify", "change"],
        "edit": ["edit", "modify", "change", "replace", "patch"],
        "patch": ["patch", "edit", "modify"],
        "write": ["write", "save", "update", "edit"],
        "delete": ["delete", "remove", "rm"],
        "remove": ["remove", "delete"],
        "move": ["move", "rename"],
        "rename": ["rename", "move"],
        "copy": ["copy", "duplicate"],
    }
    syns = list(base.get(operation, []))

    obj_set = set(objects)
    if "file" in obj_set:
        if operation in {"read", "get", "fetch"}:
            syns += ["cat", "head", "tail"]
        if operation in {"edit", "update", "write", "patch"}:
            syns += ["apply patch", "replace text"]
    if "directory" in obj_set or "dir" in obj_set or "folder" in obj_set:
        syns += ["folder", "directory"]
    return list(dict.fromkeys(syns))


def _capability_summary(operation: str, objects: list[str]) -> tuple[str, str]:
    """
    Returns (can_text, cannot_text) strings to inject strong disambiguators.
    Note: "cannot" is best-effort; embeddings may not respect negation perfectly,
    but the extra tokens still often help separation for near-duplicates.
    """
    obj = " ".join(objects) if objects else "resource"
    if _infer_side_effects(operation) == "no":
        can = f"can: {operation} {obj}; read-only"
        cannot = "cannot: modify; cannot: write; cannot: delete; cannot: create"
        return can, cannot
    if _infer_side_effects(operation) == "yes":
        can = f"can: {operation} {obj}; mutating"
        cannot = "cannot: read-only"
        return can, cannot
    return f"can: {operation} {obj}", "cannot: unknown"


def _schema_path_join(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _collect_schema_anchors(schema: Any, *, prefix: str = "", depth: int = 0, max_depth: int = 3) -> list[str]:
    """
    Recursively collect schema anchors:
    - property paths (including nested)
    - required paths (best-effort)
    - enum values (bounded)
    - short property descriptions
    """
    if depth > max_depth:
        return []
    if not isinstance(schema, dict):
        return []

    lines: list[str] = []

    props = schema.get("properties")
    required = schema.get("required")
    if isinstance(props, dict):
        for k, v in props.items():
            if not isinstance(k, str):
                continue
            path = _schema_path_join(prefix, k)
            entry = [path]
            if isinstance(v, dict):
                t = v.get("type")
                if isinstance(t, str):
                    entry.append(f"type={t}")
                enum = v.get("enum")
                if isinstance(enum, list) and enum:
                    entry.append("enum=" + ",".join(str(x) for x in enum[:8]))
                desc = v.get("description")
                if isinstance(desc, str) and desc.strip():
                    entry.append(f"desc={desc.strip()[:120]}")
            lines.append(" - " + " | ".join(entry))

            # Recurse into nested schemas.
            if isinstance(v, dict):
                lines.extend(
                    _collect_schema_anchors(
                        v, prefix=path, depth=depth + 1, max_depth=max_depth
                    )
                )

    if isinstance(required, list) and required:
        reqs = [str(x) for x in required[:50]]
        if prefix:
            lines.append("required@" + prefix + ": " + ", ".join(reqs))
        else:
            lines.append("required: " + ", ".join(reqs))

    # Handle array items
    items = schema.get("items")
    if isinstance(items, dict):
        lines.extend(
            _collect_schema_anchors(
                items, prefix=_schema_path_join(prefix, "items"), depth=depth + 1, max_depth=max_depth
            )
        )

    # Lightly surface anyOf/oneOf branches (bounded)
    for key in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(key)
        if isinstance(branches, list):
            for i, br in enumerate(branches[:5]):
                if isinstance(br, dict):
                    lines.extend(
                        _collect_schema_anchors(
                            br,
                            prefix=_schema_path_join(prefix, f"{key}[{i}]"),
                            depth=depth + 1,
                            max_depth=max_depth,
                        )
                    )

    return lines


def _tool_text_repr(t: Tool) -> str:
    name_tokens = _split_name_tokens(t.name)
    op = _infer_operation(name_tokens)
    obj_tokens = _infer_object_tokens(name_tokens)
    syns = _operation_synonyms(op, obj_tokens)
    can_text, cannot_text = _capability_summary(op, obj_tokens)

    parts = [
        f"server: {t.server}",
        f"name: {t.name}",
        f"operation: {op}",
        f"side_effects: {_infer_side_effects(op)}",
        f"objects: {', '.join(obj_tokens) if obj_tokens else ''}",
        f"synonyms: {', '.join(syns)}" if syns else "synonyms:",
        can_text,
        cannot_text,
        f"description: {t.description}",
    ]
    schema = t.input_schema or {}
    schema_summary = summarize_json_schema(schema)
    if schema_summary:
        parts.append("input_schema:")
        parts.append(schema_summary)
    if t.metadata:
        parts.append(f"metadata: {canonical_json(t.metadata)}")
    return "\n".join(parts)


def _resource_text_repr(r: Resource, content: str) -> str:
    parts = [
        f"server: {r.server}",
        f"uri: {r.uri}",
        f"name: {r.name}",
    ]
    if r.description:
        parts.append(f"description: {r.description}")
    if r.metadata:
        parts.append(f"metadata: {canonical_json(r.metadata)}")
    if content:
        parts.append("content:")
        parts.append(content)
    return "\n".join(parts)


def chunk_text(text: str, *, chunk_size: int) -> list[str]:
    if chunk_size <= 0:
        return [text]
    out: list[str] = []
    i = 0
    while i < len(text):
        out.append(text[i : i + chunk_size])
        i += chunk_size
    return out


def summarize_json_schema(schema: dict) -> str:
    """
    Extracts high-signal lexical anchors from a JSON schema:
    property names, required fields, and enum values.

    This improves retrieval accuracy (especially for lexical embeddings like hashing)
    without changing the public API.
    """
    if not isinstance(schema, dict) or not schema:
        return ""

    lines: list[str] = []
    lines.extend(_collect_schema_anchors(schema))

    # Common schema keywords that help retrieval even without properties.
    if "title" in schema and isinstance(schema["title"], str):
        lines.append(f"title: {schema['title']}")
    if "description" in schema and isinstance(schema["description"], str):
        lines.append(f"schema_description: {schema['description']}")

    return "\n".join(lines).strip()


@dataclass
class DescriptorSync:
    store: SQLiteStore
    embedder: EmbeddingProvider
    resource_chunk_size: int = 1200
    max_resource_content_chars: int = 50_000

    def sync_from_root(
        self,
        root: str | os.PathLike[str],
        *,
        delete_stale: bool = False,
        fetch_resource_content: Optional[ResourceContentFetcher] = None,
    ) -> dict[str, int]:
        """
        Syncs tools/resources from:
          Tools: <root>/<server>/tools/*.json
          Resources: <root>/<server>/resources/*.json
        """
        root_path = Path(root)
        self.store.init()

        seen_tool_ids: set[str] = set()
        seen_resource_ids: set[str] = set()

        tools_upserted = 0
        resources_upserted = 0
        chunks_upserted = 0

        for server_dir in sorted([p for p in root_path.iterdir() if p.is_dir()]):
            server = server_dir.name

            tool_dir = server_dir / "tools"
            if tool_dir.exists():
                for p in sorted(tool_dir.glob("*.json")):
                    raw = _read_json(p)
                    # Descriptor compatibility:
                    # - input_schema / schema / arguments are common variants
                    schema = raw.get("input_schema", raw.get("schema", raw.get("arguments", {})))
                    tool = Tool(
                        server=server,
                        name=str(raw["name"]),
                        description=str(raw.get("description", "")),
                        input_schema=dict(schema),
                        version=raw.get("version"),
                        metadata=dict(raw.get("metadata", {})),
                    )
                    tool_id = stable_tool_id(tool.server, tool.name, tool.version)
                    seen_tool_ids.add(tool_id)

                    payload = {
                        "server": tool.server,
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                        "version": tool.version,
                        "metadata": tool.metadata,
                    }
                    chash = content_hash(payload)
                    prev = self.store.get_tool_hash(tool_id)
                    if prev == chash:
                        continue

                    text_repr = _tool_text_repr(tool)
                    emb = self.embedder.embed_text(text_repr)
                    self.store.upsert_tool(
                        tool_id=tool_id,
                        server=tool.server,
                        name=tool.name,
                        version=tool.version,
                        content_hash=chash,
                        text_repr=text_repr,
                        payload_json=canonical_json(payload),
                        embedding=emb,
                    )
                    tools_upserted += 1

            res_dir = server_dir / "resources"
            if res_dir.exists():
                for p in sorted(res_dir.glob("*.json")):
                    raw = _read_json(p)
                    res = Resource(
                        server=server,
                        uri=str(raw["uri"]),
                        name=str(raw.get("name", raw["uri"])),
                        description=str(raw.get("description", "")),
                        mime_type=raw.get("mime_type", raw.get("mimeType")),
                        version=raw.get("version"),
                        metadata=dict(raw.get("metadata", {})),
                    )
                    resource_id = stable_resource_id(res.server, res.uri, res.version)
                    seen_resource_ids.add(resource_id)

                    descriptor_payload = {
                        "server": res.server,
                        "uri": res.uri,
                        "name": res.name,
                        "description": res.description,
                        "mime_type": res.mime_type,
                        "version": res.version,
                        "metadata": res.metadata,
                    }
                    rhash = content_hash(descriptor_payload)

                    prev = self.store.get_resource_hash(resource_id)
                    if prev != rhash:
                        self.store.upsert_resource(
                            resource_id=resource_id,
                            server=res.server,
                            uri=res.uri,
                            name=res.name,
                            version=res.version,
                            content_hash=rhash,
                            descriptor_json=canonical_json(descriptor_payload),
                        )
                        resources_upserted += 1

                    content = ""
                    if fetch_resource_content is not None:
                        try:
                            content = fetch_resource_content(res.uri) or ""
                        except Exception:
                            content = ""
                    else:
                        # optional inline content field in descriptor
                        content = str(raw.get("content", "")) if raw.get("content") else ""

                    if content:
                        content = content[: self.max_resource_content_chars]

                    # Only (re)chunk when the actual resource content changed.
                    if content:
                        content_hash_full = sha256_text(content)
                        prev_content_hash = self.store.get_resource_content_hash(resource_id)
                        if prev_content_hash != content_hash_full:
                            # update the resource row with content hash (without changing descriptor hash)
                            self.store.upsert_resource(
                                resource_id=resource_id,
                                server=res.server,
                                uri=res.uri,
                                name=res.name,
                                version=res.version,
                                content_hash=rhash,
                                descriptor_json=canonical_json(descriptor_payload),
                                resource_content_hash=content_hash_full,
                            )

                            self.store.delete_chunks_for_resource(resource_id)
                            chunks = chunk_text(content, chunk_size=self.resource_chunk_size)
                            for idx, chunk in enumerate(chunks):
                                chunk_payload_hash = sha256_text(
                                    f"{resource_id}:{idx}:{sha256_text(chunk)}"
                                )
                                chunk_id = f"chunk:{chunk_payload_hash}"
                                text_repr = _resource_text_repr(res, chunk)
                                emb = self.embedder.embed_text(text_repr)
                                self.store.upsert_resource_chunk(
                                    chunk_id=chunk_id,
                                    resource_id=resource_id,
                                    server=res.server,
                                    uri=res.uri,
                                    chunk_index=idx,
                                    content_hash=sha256_text(chunk),
                                    text=chunk,
                                    embedding=emb,
                                )
                                chunks_upserted += 1

        deleted_tools = 0
        deleted_resources = 0
        if delete_stale:
            deleted_tools = self.store.delete_tools_not_in(seen_tool_ids)
            deleted_resources = self.store.delete_resources_not_in(seen_resource_ids)

        return {
            "tools_upserted": tools_upserted,
            "resources_upserted": resources_upserted,
            "resource_chunks_upserted": chunks_upserted,
            "tools_deleted": deleted_tools,
            "resources_deleted": deleted_resources,
        }

