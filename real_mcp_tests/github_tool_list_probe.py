from __future__ import annotations

import asyncio
import os

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main() -> int:
    print("REAL GITHUB MCP PROBE")
    print("Attempting to start @modelcontextprotocol/server-github via stdio and list tools.")
    print("")

    # Most GitHub servers require a token. We do not invent one.
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("No GITHUB_TOKEN or GH_TOKEN found in environment.")
        print("This server likely requires GitHub auth to initialize.")
        print("REALNESS CHECK: REAL server not started (missing auth).")
        return 2

    params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={**os.environ, "GITHUB_TOKEN": token},
    )

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print(f"Discovered {len(tools.tools)} tools.")
                print("Tool names:", ", ".join(t.name for t in tools.tools))
                print("")
                print("REALNESS CHECK: REAL (tools fetched at runtime).")
                return 0
    except Exception as e:
        print("Failed to start or query GitHub server.")
        print(f"Error: {e!r}")
        print("")
        print("REALNESS CHECK: PARTIALLY REAL (attempted real server, but could not complete tool discovery).")
        return 3


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

