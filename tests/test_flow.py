import json
from pathlib import Path

from rag_mcp import McpTool, RagMcpPlugin


def test_plugin_route_and_assemble():
    plugin = RagMcpPlugin(server_name="demo", top_k=3)
    plugin.register_tools(
        [
            McpTool(
                name="web_search",
                description="Search the web for up-to-date information.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
            McpTool(
                name="send_email",
                description="Send an email message to a recipient.",
                input_schema={
                    "type": "object",
                    "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
                    "required": ["to", "body"],
                },
            ),
        ]
    )

    decision = plugin.route_query("search latest ai funding news")
    assert decision.best_tool is not None
    assert decision.best_tool.name == "web_search"

    context = plugin.assemble_for_llm(
        "search latest ai funding news",
        routing=decision,
    )
    assert len(context.tools) == 1
    assert context.tools[0]["name"] == "web_search"


def test_descriptor_sync_and_stale_deletion(tmp_path: Path):
    descriptor_root = tmp_path / "mcps"
    tools_dir = descriptor_root / "serpapi" / "tools"
    tools_dir.mkdir(parents=True)

    search_tool = {
        "name": "search",
        "description": "Search the web and weather data using SerpApi.",
        "arguments": {
            "type": "object",
            "properties": {
                "params": {"type": "object", "additionalProperties": True},
            },
        },
    }
    news_tool = {
        "name": "news",
        "description": "Search latest news headlines.",
        "arguments": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    (tools_dir / "search.json").write_text(json.dumps(search_tool), encoding="utf-8")
    (tools_dir / "news.json").write_text(json.dumps(news_tool), encoding="utf-8")

    db_path = tmp_path / "rag_mcp.db"
    plugin = RagMcpPlugin(server_name="demo", top_k=3, db_path=str(db_path))

    first = plugin.sync_tools_from_descriptor_root(str(descriptor_root))
    assert first.discovered_servers == 1
    assert first.discovered_tools == 2
    assert first.added == 2
    assert first.updated == 0
    assert first.skipped == 0
    assert first.removed == 0
    assert first.errors == []

    decision = plugin.route_query("weather in London")
    assert decision.best_tool is not None
    assert decision.best_tool.name == "search"

    # Re-sync unchanged descriptors should be idempotent.
    second = plugin.sync_tools_from_descriptor_root(str(descriptor_root))
    assert second.added == 0
    assert second.updated == 0
    assert second.skipped == 2

    # Remove one descriptor, then delete stale index entries.
    (tools_dir / "news.json").unlink()
    third = plugin.sync_tools_from_descriptor_root(
        str(descriptor_root),
        delete_stale=True,
    )
    assert third.discovered_tools == 1
    assert third.removed == 1


def test_descriptor_sync_indexes_tools_from_all_servers_by_default(tmp_path: Path):
    descriptor_root = tmp_path / "mcps"
    serpapi_tools = descriptor_root / "serpapi" / "tools"
    github_tools = descriptor_root / "github" / "tools"
    serpapi_tools.mkdir(parents=True)
    github_tools.mkdir(parents=True)

    (serpapi_tools / "search.json").write_text(
        json.dumps(
            {
                "name": "search",
                "description": "Search the public web.",
                "arguments": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ),
        encoding="utf-8",
    )
    (github_tools / "create-issue.json").write_text(
        json.dumps(
            {
                "name": "create_issue",
                "description": "Create a GitHub issue in a repository.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string"},
                        "title": {"type": "string"},
                    },
                    "required": ["repo", "title"],
                },
            }
        ),
        encoding="utf-8",
    )

    plugin = RagMcpPlugin(server_name="demo", top_k=4, db_path=str(tmp_path / "rag_mcp.db"))
    report = plugin.sync_tools_from_descriptor_root(str(descriptor_root))

    assert report.discovered_servers == 2
    assert report.discovered_tools == 2
    assert report.added == 2

    web_decision = plugin.route_query("search latest AI breakthroughs")
    assert web_decision.best_tool is not None
    assert web_decision.best_tool.server == "serpapi"
    assert web_decision.best_tool.name == "search"

    issue_decision = plugin.route_query("open an issue in octocat hello world")
    assert issue_decision.best_tool is not None
    assert issue_decision.best_tool.server == "github"
    assert issue_decision.best_tool.name == "create_issue"


def test_single_candidate_routing_selects_tool(tmp_path: Path):
    descriptor_root = tmp_path / "mcps"
    tools_dir = descriptor_root / "serpapi" / "tools"
    tools_dir.mkdir(parents=True)

    only_tool = {
        "name": "search",
        "description": "Universal search for weather and web.",
        "arguments": {
            "type": "object",
            "properties": {
                "params": {"type": "object", "additionalProperties": True},
            },
        },
    }
    (tools_dir / "search.json").write_text(json.dumps(only_tool), encoding="utf-8")

    plugin = RagMcpPlugin(
        server_name="demo",
        top_k=3,
        db_path=str(tmp_path / "rag_mcp.db"),
        confidence_threshold=0.99,
        margin_threshold=0.99,
    )
    plugin.sync_tools_from_descriptor_root(str(descriptor_root))

    decision = plugin.route_query("weather in London")
    assert decision.best_tool is not None
    assert decision.best_tool.name == "search"


def test_descriptor_resource_sync_and_retrieve(tmp_path: Path):
    descriptor_root = tmp_path / "mcps"
    resources_dir = descriptor_root / "serpapi" / "resources"
    resources_dir.mkdir(parents=True)

    google_resource = {
        "uri": "serpapi://engines/google",
        "name": "serpapi-engine-google",
        "description": "SerpApi engine specification for google.",
        "mimeType": "application/json",
    }
    bing_resource = {
        "uri": "serpapi://engines/bing",
        "name": "serpapi-engine-bing",
        "description": "SerpApi engine specification for bing.",
        "mimeType": "application/json",
    }
    (resources_dir / "google.json").write_text(
        json.dumps(google_resource), encoding="utf-8"
    )
    (resources_dir / "bing.json").write_text(json.dumps(bing_resource), encoding="utf-8")

    plugin = RagMcpPlugin(server_name="demo", db_path=str(tmp_path / "rag_mcp.db"))
    first = plugin.sync_resources_from_descriptor_root(str(descriptor_root))
    assert first.discovered_servers == 1
    assert first.discovered_resources == 2
    assert first.added == 2
    assert first.updated == 0
    assert first.skipped == 0
    assert first.errors == []

    chunks = plugin.retrieve_resource_context("google engine", top_k=2, server="serpapi")
    assert chunks
    assert any("serpapi-engine-google" in c for c in chunks)

    second = plugin.sync_resources_from_descriptor_root(str(descriptor_root))
    assert second.added == 0
    assert second.updated == 0
    assert second.skipped == 2

    (resources_dir / "bing.json").unlink()
    third = plugin.sync_resources_from_descriptor_root(
        str(descriptor_root),
        delete_stale=True,
    )
    assert third.discovered_resources == 1
    assert third.removed == 1


def test_descriptor_resource_sync_with_payload_fetch(tmp_path: Path):
    descriptor_root = tmp_path / "mcps"
    resources_dir = descriptor_root / "serpapi" / "resources"
    resources_dir.mkdir(parents=True)

    resource_descriptor = {
        "uri": "serpapi://engines/google",
        "name": "serpapi-engine-google",
        "description": "SerpApi engine specification for google.",
        "mimeType": "application/json",
    }
    (resources_dir / "google.json").write_text(
        json.dumps(resource_descriptor), encoding="utf-8"
    )

    def fetcher(uri: str, server: str):
        assert uri == "serpapi://engines/google"
        assert server == "serpapi"
        return json.dumps(
            {
                "engine": "google",
                "required_params": ["q"],
                "marker": "PAYLOAD_FETCHED_OK",
            }
        )

    plugin = RagMcpPlugin(server_name="demo", db_path=str(tmp_path / "rag_mcp.db"))
    report = plugin.sync_resources_from_descriptor_root(
        str(descriptor_root),
        fetch_content=True,
        resource_fetcher=fetcher,
    )
    assert report.discovered_resources == 1
    assert report.fetched_payloads == 1

    chunks = plugin.retrieve_resource_context("PAYLOAD_FETCHED_OK", top_k=3, server="serpapi")
    assert chunks
    assert any("PAYLOAD_FETCHED_OK" in c for c in chunks)
