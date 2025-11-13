"""jira_client.py

Lightweight wrapper around the FastMCP / Streamable HTTP client to call
Jira-related tools exposed by an MCP server.

This module provides JiraMCPClient which is an async context manager that
connects to an MCP Streamable HTTP endpoint and exposes simple methods such
as `search_issues` to call the server's tool endpoints.

Usage:
    async with JiraMCPClient() as client:
        issues = await client.search_issues('assignee = "Kiet Ho" AND project = TEST')
        print(issues)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import asyncio
import os

from pydantic_ai.mcp import MCPServerStreamableHTTP


class JiraMCPClient:
    """Async client for calling Jira tools on a Streamable HTTP MCP server.

    This wraps `pydantic_ai.mcp.MCPServerStreamableHTTP` which itself uses the
    MCP `streamablehttp_client` transport under the hood.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        timeout: float = 5.0,
        sse_read_timeout: float = 300.0,
        tool_prefix: Optional[str] = None,
    ) -> None:
        self.url = url or os.getenv("MCP_SERVER_URL", "http://localhost:9000/mcp")
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        self.tool_prefix = tool_prefix
        self._server: Optional[MCPServerStreamableHTTP] = None

    async def __aenter__(self) -> "JiraMCPClient":
        self._server = MCPServerStreamableHTTP(
            url=self.url,
            timeout=self.timeout,
            sse_read_timeout=self.sse_read_timeout,
            tool_prefix=self.tool_prefix,
        )
        await self._server.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if self._server:
            await self._server.__aexit__(exc_type, exc, tb)

    async def search_issues(
        self,
        jql: str,
        fields: str = "summary,status,priority",
        max_results: int = 200,
    ) -> List[Dict[str, Any]]:
        """Search Jira issues by JQL via the MCP `jira_search` tool.

        Returns the parsed list of issues (list of dicts) on success. If the
        server returns a different tool name, update the called tool accordingly.
        """
        if not self._server:
            raise RuntimeError("JiraMCPClient must be used as an async context manager")

        # Call the tool. Many MCP servers expose a `jira_search` tool that accepts
        # parameters similar to the langchain MCP adapter. Adjust the tool name
        # if your MCP server uses a different tool id (e.g. `jira_search_issues`).
        params = {"jql": jql, "fields": fields, "limit": max_results}

        # Use the MCPServer helper to call the tool (this wraps JSON-RPC + streaming)
        result = await self._server.call_tool("jira_search", params)

        # The MCPServer.call_tool maps tool result parts to python types. Often the
        # first returned item is the desired text or structure. Normalize to a list
        # of issue dicts where possible.
        if result is None:
            return []

        # If the result is a single JSON string, try to parse it
        import json

        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except Exception:
                # Not JSON — return the raw string in a simple wrapper
                return [{"text": result}]

            # Parsed JSON may contain an `issues` key or be a list itself
            if isinstance(parsed, dict) and "issues" in parsed:
                return parsed["issues"]
            if isinstance(parsed, list):
                return parsed
            # Otherwise return the parsed object as a single item
            return [parsed]

        # If it's already a list/dict, normalize
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            # common shape: {"issues": [...]}
            if "issues" in result and isinstance(result["issues"], list):
                return result["issues"]
            return [result]

        # Fallback: return textual representation
        return [{"value": str(result)}]


async def _cli_list():
    jql = 'assignee = "Sang Phạm"'
    async with JiraMCPClient() as client:
        issues = await client.search_issues(jql)
        print(f"Found {len(issues)} issues assigned to Sang Phạm")
        for issue in issues:
            key = issue.get("key") or issue.get("id")
            summary = issue.get("summary") or (issue.get("fields") or {}).get("summary")
            status = (issue.get("status") or {}).get("name") if isinstance(issue.get("status"), dict) else (issue.get("fields") or {}).get("status", {}).get("name")
            priority = (issue.get("priority") or {}).get("name") if isinstance(issue.get("priority"), dict) else (issue.get("fields") or {}).get("priority", {}).get("name")
            print(f"- {key}: {summary} [status: {status}] [priority: {priority}]")


if __name__ == "__main__":
    asyncio.run(_cli_list())
