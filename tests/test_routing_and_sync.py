from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_tool_router import Client


def _write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def test_routing_selects_best_tool(tmp_path: Path) -> None:
    root = tmp_path / "desc"

    _write_json(
        root / "github" / "tools" / "search_issues.json",
        {
            "name": "search_issues",
            "description": "Search GitHub issues by query and filters.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "version": "1",
        },
    )
    _write_json(
        root / "slack" / "tools" / "post_message.json",
        {
            "name": "post_message",
            "description": "Post a message to a Slack channel.",
            "input_schema": {
                "type": "object",
                "properties": {"channel": {"type": "string"}, "text": {"type": "string"}},
                "required": ["channel", "text"],
            },
            "version": "1",
        },
    )

    client = Client(db_path=str(tmp_path / "db.sqlite3"))
    stats = client.sync_descriptors(root=str(root), delete_stale=True)
    assert stats["tools_upserted"] == 2

    decision = client.route("Find issues about auth token refresh failing", top_k=5)
    assert decision.abstained is False
    assert decision.best_tool is not None
    assert decision.best_tool.server == "github"
    assert decision.best_tool.name == "search_issues"
    assert decision.confidence >= 0.0
    assert decision.margin >= 0.0


def test_sync_idempotent_and_delete_stale(tmp_path: Path) -> None:
    root = tmp_path / "desc"
    tool_a = root / "srv" / "tools" / "a.json"
    tool_b = root / "srv" / "tools" / "b.json"

    _write_json(
        tool_a,
        {
            "name": "a",
            "description": "Tool A does alpha.",
            "input_schema": {"type": "object", "properties": {}},
            "version": "1",
        },
    )
    _write_json(
        tool_b,
        {
            "name": "b",
            "description": "Tool B does beta.",
            "input_schema": {"type": "object", "properties": {}},
            "version": "1",
        },
    )

    client = Client(db_path=str(tmp_path / "db.sqlite3"))
    stats1 = client.sync_descriptors(root=str(root), delete_stale=True)
    assert stats1["tools_upserted"] == 2
    assert stats1["tools_deleted"] == 0

    # Re-sync without changes should be idempotent (0 upserts)
    stats2 = client.sync_descriptors(root=str(root), delete_stale=True)
    assert stats2["tools_upserted"] == 0

    # Remove tool_b and delete stale entries
    tool_b.unlink()
    stats3 = client.sync_descriptors(root=str(root), delete_stale=True)
    assert stats3["tools_deleted"] == 1


def test_resource_retrieval_included_in_context(tmp_path: Path) -> None:
    root = tmp_path / "desc"

    _write_json(
        root / "docs" / "tools" / "lookup.json",
        {
            "name": "lookup",
            "description": "Lookup a documentation page by title.",
            "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}},
            "version": "1",
        },
    )

    _write_json(
        root / "docs" / "resources" / "auth_spec.json",
        {
            "uri": "docs://auth_spec",
            "name": "Auth Spec",
            "description": "Authentication specification and token refresh rules.",
            "content": "Token refresh: clients should refresh 2 minutes before expiry. "
            "If refresh fails with 401, re-authenticate and retry once.",
            "version": "1",
        },
    )

    client = Client(db_path=str(tmp_path / "db.sqlite3"))
    stats = client.sync_descriptors(root=str(root), delete_stale=True)
    assert stats["resources_upserted"] == 1
    assert stats["resource_chunks_upserted"] >= 1

    decision = client.route("How should token refresh work when 401 happens?")
    payload = client.build_llm_context(
        user_query="How should token refresh work when 401 happens?",
        decision=decision,
        resource_top_k=2,
    )
    assert "Relevant resource context:" in payload["system_context"]
    assert "refresh" in payload["system_context"].lower()


def test_abstain_when_thresholds_high(tmp_path: Path) -> None:
    root = tmp_path / "desc"

    _write_json(
        root / "srv" / "tools" / "do_thing.json",
        {
            "name": "do_thing",
            "description": "Does a thing.",
            "arguments": {"type": "object", "properties": {"x": {"type": "string"}}},
            "version": "1",
        },
    )

    client = Client(db_path=str(tmp_path / "db.sqlite3"))
    client.sync_descriptors(root=str(root), delete_stale=True)

    decision = client.route(
        "completely unrelated query about astrophysics",
        min_confidence=0.999,
        min_margin=0.999,
    )
    assert decision.abstained is True
    assert decision.best_tool is None


def test_resource_chunk_sync_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "desc"

    _write_json(
        root / "docs" / "resources" / "r.json",
        {
            "uri": "docs://r",
            "name": "R",
            "description": "Test resource",
            "mimeType": "text/plain",
            "content": "hello world " * 200,
            "version": "1",
        },
    )

    client = Client(db_path=str(tmp_path / "db.sqlite3"))
    first = client.sync_descriptors(root=str(root), delete_stale=True)
    assert first["resource_chunks_upserted"] >= 1

    second = client.sync_descriptors(root=str(root), delete_stale=True)
    # Content unchanged -> should not rewrite chunks
    assert second["resource_chunks_upserted"] == 0


def test_schema_terms_improve_routing(tmp_path: Path) -> None:
    root = tmp_path / "desc"

    # Same-ish names/descriptions; only schema differs.
    _write_json(
        root / "srv" / "tools" / "alpha.json",
        {
            "name": "alpha",
            "description": "Perform an action for a destination.",
            "arguments": {
                "type": "object",
                "properties": {"channel": {"type": "string"}},
                "required": ["channel"],
            },
            "version": "1",
        },
    )
    _write_json(
        root / "srv" / "tools" / "beta.json",
        {
            "name": "beta",
            "description": "Perform an action for a destination.",
            "arguments": {
                "type": "object",
                "properties": {"repo": {"type": "string"}},
                "required": ["repo"],
            },
            "version": "1",
        },
    )

    client = Client(db_path=str(tmp_path / "db.sqlite3"))
    client.sync_descriptors(root=str(root), delete_stale=True)

    decision = client.route(
        "Send this update to the channel general",
        top_k=5,
        min_confidence=0.0,
        min_margin=0.0,
    )
    assert decision.best_tool is not None
    assert decision.best_tool.name == "alpha"

