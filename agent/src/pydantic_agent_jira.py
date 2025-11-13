"""pydantic_agent_jira.py

Create a Pydantic/FASTMCP-backed agent that connects to an MCP "atlassian"
server over the Streamable HTTP transport.

This module provides `PydanticAgentJira`, an async context-managed wrapper
that builds a `pydantic_ai.Agent` with a `MCPServerStreamableHTTP` client
pointing at the local MCP server (defaults to http://localhost:9000/mcp).

The agent registers a couple of small tools that proxy to the MCP tools
(`jira_search`, `jira_create_issue`) so the agent can call the Jira backend
through the MCP server.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional


from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStreamableHTTP
# `pydantic_ai.ag_ui` is an optional UI dependency (extra). Import it only
# for type checking; at runtime provide a lightweight fallback so this module
# works even if the `ag-ui` extras are not installed in the environment.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.ag_ui import StateDeps  # type: ignore
else:
    # Minimal runtime stand-in that supports subscription syntax: StateDeps[T]
    class _StateDepsProxy:
        def __class_getitem__(cls, item):
            return item

    StateDeps = _StateDepsProxy
from tools import ProverbsState

from dotenv import load_dotenv
# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)


class PydanticAgentJira:
    """Encapsulates an Agent connected to an MCP (Streamable HTTP) Jira server.

    Usage:
        async with PydanticAgentJira() as pa:
            issues = await pa.search_issues('assignee = "Kiet Ho" AND project = TEST')
            print(issues)
    """

    def __init__(
        self,
        mcp_url: Optional[str] = None,
        timeout: float = 5.0,
        sse_read_timeout: float = 300.0,
        tool_prefix: Optional[str] = None,
        model: str = "openai:gpt-4o-mini",
    ) -> None:
        self.mcp_url = mcp_url or os.getenv("MCP_SERVER_URL", "http://localhost:9000/mcp")
        # Ensure path ends with /mcp if user passed base url without path
        if self.mcp_url.endswith("/mcp") is False and self.mcp_url.endswith("/mcp/") is False:
            # allow both http://host:9000 and http://host:9000/mcp
            if self.mcp_url.endswith("/"):
                self.mcp_url = self.mcp_url + "mcp"
            else:
                self.mcp_url = self.mcp_url + "/mcp"

        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        self.tool_prefix = tool_prefix
        self.model = model

        # Create the MCP client wrapper (pydantic-ai helper class)
        self.mcp_server = MCPServerStreamableHTTP(
            url=self.mcp_url,
            timeout=self.timeout,
            sse_read_timeout=self.sse_read_timeout,
            tool_prefix=self.tool_prefix,
        )

        # Attempt to create an Agent only if an LLM provider is configured.
        # Creating an OpenAI-backed Agent will raise if OPENAI_API_KEY isn't
        # set in the environment. Many use-cases for this helper only need the
        # MCP client to call `jira_*` tools directly, so we make agent
        # construction optional and non-fatal.
        self.agent = None
        try:
            if os.getenv("OPENAI_API_KEY"):
                self.agent = Agent(
                    name="pydantic_agent_jira",
                    model=self.model,
                    system_prompt=(
                        "You are a Jira assistant connected to an MCP Jira backend. "
                        "Use registered tools to search and manage Jira issues."
                    ),
                    retries=1,
                    mcp_servers=[self.mcp_server],
                )
            else:
                # No OpenAI API key found — skip Agent creation and rely on
                # the MCP client wrapper for direct tool calls.
                print(
                    "[pydantic_agent_jira] OPENAI_API_KEY not set — skipping LLM Agent creation."
                )
        except Exception as err:  # pragma: no cover - defensive fallback
            # If agent construction fails (e.g. missing API key or provider
            # initialization error), continue with only the MCP client so the
            # module can still call jira_* tools.
            print(f"[pydantic_agent_jira] Agent creation failed: {err}; continuing without LLM agent.")

        # Register agent tools that proxy to MCP jira_* endpoints (if agent exists)
        if self.agent is not None:
            self._register_agent_tools()

    def _register_agent_tools(self) -> None:
        """Register agent tools for Jira operations via MCP."""
        
        @self.agent.tool
        async def search_issues(
            ctx: RunContext[StateDeps[ProverbsState]],
            jql: str,
            fields: str = "summary,status,priority",
            limit: int = 50,
        ) -> Any:
            """Search Jira issues using JQL query via the MCP jira_search tool.
            
            Args:
                jql: Jira Query Language query string (e.g., 'assignee = "Kiet Ho"')
                fields: Comma-separated list of fields to return
                limit: Maximum number of issues to return
                
            Returns:
                List of matching issues with the requested fields.
            """
            result = await self.mcp_server.call_tool(
                "jira_search",
                {"jql": jql, "fields": fields, "limit": limit},
            )
            return result

        @self.agent.tool
        async def create_issue(
            ctx: RunContext[StateDeps[ProverbsState]],
            project_key: str,
            summary: str,
            description: str,
            issue_type: str = "Task",
            priority: str = "Medium",
            assignee: Optional[str] = None,
        ) -> Any:
            """Create a new Jira issue via the MCP jira_create_issue tool.
            
            Args:
                project_key: Jira project key (e.g., 'TEST', 'PROJ')
                summary: Brief issue title/summary
                description: Detailed description of the issue
                issue_type: Type of issue (Task, Bug, Story, Epic, Subtask, etc.)
                priority: Priority level (Low, Medium, High, Highest, etc.)
                assignee: Optional username/email to assign the issue to
                
            Returns:
                Created issue details including the new issue key.
            """
            params = {
                "project_key": project_key,
                "summary": summary,
                "description": description,
                "issue_type": issue_type,
                "priority": priority,
            }
            if assignee:
                params["assignee"] = assignee
                
            result = await self.mcp_server.call_tool("jira_create_issue", params)
            return result

        @self.agent.tool
        async def update_issue(
            ctx: RunContext[StateDeps[ProverbsState]],
            issue_key: str,
            status: Optional[str] = None,
            assignee: Optional[str] = None,
            priority: Optional[str] = None,
            comment: Optional[str] = None,
            labels: Optional[List[str]] = None,
        ) -> Any:
            """Update an existing Jira issue via the MCP jira_update_issue tool.
            
            Args:
                issue_key: Issue key (e.g., 'TEST-123')
                status: New status to transition to (optional)
                assignee: New assignee username/email (optional)
                priority: New priority level (optional)
                comment: Comment text to add to the issue (optional)
                labels: List of labels to add/update (optional)
                
            Returns:
                Confirmation of the update operation.
            """
            params = {"issue_key": issue_key}
            if status:
                params["status"] = status
            if assignee:
                params["assignee"] = assignee
            if priority:
                params["priority"] = priority
            if comment:
                params["comment"] = comment
            if labels:
                params["labels"] = labels
                
            result = await self.mcp_server.call_tool("jira_update_issue", params)
            return result

    async def __aenter__(self) -> "PydanticAgentJira":
        # Safely enter the MCP client and (optionally) the LLM Agent.
        # Different objects may implement sync context managers, async
        # context managers, or none at all — handle all cases gracefully.

        # Enter MCP server context (if it supports context management)
        mcp_enter = getattr(self.mcp_server, "__aenter__", None)
        if mcp_enter is not None:
            if asyncio.iscoroutinefunction(mcp_enter):
                await mcp_enter()
            else:
                # sync context manager
                self.mcp_server.__enter__()

        # Enter Agent context only if an Agent was created and it supports
        # a context manager. The Agent implementation may be sync-only.
        if self.agent is not None:
            agent_aenter = getattr(self.agent, "__aenter__", None)
            if agent_aenter is not None:
                if asyncio.iscoroutinefunction(agent_aenter):
                    await agent_aenter()
                else:
                    self.agent.__enter__()

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        # Safely exit the Agent (if present) and the MCP client. Mirror the
        # entering logic: prefer async exit if available, otherwise call the
        # sync exit methods.

        if self.agent is not None:
            agent_aexit = getattr(self.agent, "__aexit__", None)
            if agent_aexit is not None:
                if asyncio.iscoroutinefunction(agent_aexit):
                    await agent_aexit(exc_type, exc, tb)
                else:
                    # sync context manager exit: call __exit__
                    self.agent.__exit__(exc_type, exc, tb)

        mcp_aexit = getattr(self.mcp_server, "__aexit__", None)
        if mcp_aexit is not None:
            if asyncio.iscoroutinefunction(mcp_aexit):
                await mcp_aexit(exc_type, exc, tb)
            else:
                self.mcp_server.__exit__(exc_type, exc, tb)

    # Convenience high-level methods that call the underlying tools directly
    async def search_issues(self, jql: str, fields: str = "summary,status,priority", limit: int = 200):
        return await self.mcp_server.call_tool("jira_search", {"jql": jql, "fields": fields, "limit": limit})

    async def create_issue(self, project_key: str, summary: str, description: str, **kwargs):
        args = {"project_key": project_key, "summary": summary, "description": description, **kwargs}
        return await self.mcp_server.call_tool("jira_create_issue", args)


async def _demo():
    jql = 'assignee = "Sang Phạm"'
    async with PydanticAgentJira() as pa:
        issues = await pa.search_issues(jql, limit=200)
        print("Demo: issues for Sang Phạm ->")
        print(issues)


if __name__ == "__main__":
    asyncio.run(_demo())
