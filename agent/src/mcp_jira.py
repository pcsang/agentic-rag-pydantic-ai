from pathlib import Path
from dotenv import load_dotenv
import os
import asyncio
import json
from typing import Any, Optional
from fastmcp.client import Client
from pydantic_ai import RunContext
from pydantic_ai.ag_ui import StateDeps
import tools

# load environment variables from agent/.env explicitly
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# MCP configuration - prefer FASTMCP_URL or MCP_SERVER_URL, fall back to localhost:9000
mcp_url = os.getenv("FASTMCP_URL") or os.getenv("MCP_SERVER_URL") or "http://localhost:9000/mcp"


async def call_mcp_jira_tool(tool_name: str, params: dict) -> str:
    """Helper to call a Jira tool via the local MCP server.

    This is a direct copy of the original helper but moved into a dedicated module so
    that MCP-related logic is kept separate from agent wiring.
    """
    try:
        async with Client(mcp_url) as client:
            result = await client.call_tool(tool_name, params)
            # Extract text from CallToolResult
            if hasattr(result, 'content') and result.content:
                text_parts = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        text_parts.append(item.text)
                    else:
                        text_parts.append(str(item))
                return "\n".join(text_parts)
            return str(result)
    except Exception as e:
        return f"Error calling {tool_name}: {str(e)}"


########## Plain async Jira functions (not decorated here) ##########
# These functions mirror the prior `@agent.tool` implementations but are
# defined without any agent decoration. The agent wiring module will
# register them as tools on the Agent instance.

async def jira_search(ctx: RunContext[StateDeps[tools.ProverbsState]], jql: str, limit: int = 10) -> str:
    return await call_mcp_jira_tool("jira_search", {"jql": jql, "limit": min(limit, 50)})


async def jira_get_issue(ctx: RunContext[StateDeps[tools.ProverbsState]], issue_key: str, fields: str = "summary,status,assignee,description,priority") -> str:
    return await call_mcp_jira_tool("jira_get_issue", {"issue_key": issue_key, "fields": fields})


async def jira_create_issue(ctx: RunContext[StateDeps[tools.ProverbsState]], project_key: str, summary: str, issue_type: str, description: Optional[str] = None, assignee: Optional[str] = None) -> str:
    params = {"project_key": project_key, "summary": summary, "issue_type": issue_type}
    if description:
        params["description"] = description
    if assignee:
        params["assignee"] = assignee
    return await call_mcp_jira_tool("jira_create_issue", params)


async def jira_add_comment(ctx: RunContext[StateDeps[tools.ProverbsState]], issue_key: str, comment: str) -> str:
    return await call_mcp_jira_tool("jira_add_comment", {"issue_key": issue_key, "comment": comment})


async def jira_search_fields(ctx: RunContext[StateDeps[tools.ProverbsState]], keyword: str = "", limit: int = 10) -> str:
    return await call_mcp_jira_tool("jira_search_fields", {"keyword": keyword, "limit": limit})


async def jira_get_project_issues(ctx: RunContext[StateDeps[tools.ProverbsState]], project_key: str, limit: int = 10) -> str:
    return await call_mcp_jira_tool("jira_get_project_issues", {"project_key": project_key, "limit": min(limit, 50)})


async def jira_get_all_projects(ctx: RunContext[StateDeps[tools.ProverbsState]]) -> str:
    return await call_mcp_jira_tool("jira_get_all_projects", {})


async def jira_update_issue(ctx: RunContext[StateDeps[tools.ProverbsState]], issue_key: str, summary: Optional[str] = None, description: Optional[str] = None, assignee: Optional[str] = None, status: Optional[str] = None, priority: Optional[str] = None) -> str:
    params = {"issue_key": issue_key}
    if summary:
        params["summary"] = summary
    if description:
        params["description"] = description
    if assignee:
        params["assignee"] = assignee
    if status:
        params["status"] = status
    if priority:
        params["priority"] = priority
    return await call_mcp_jira_tool("jira_update_issue", params)


async def jira_transition_issue(ctx: RunContext[StateDeps[tools.ProverbsState]], issue_key: str, transition_name: str) -> str:
    return await call_mcp_jira_tool("jira_transition_issue", {"issue_key": issue_key, "transition_name": transition_name})


async def jira_get_transitions(ctx: RunContext[StateDeps[tools.ProverbsState]], issue_key: str) -> str:
    return await call_mcp_jira_tool("jira_get_transitions", {"issue_key": issue_key})


async def jira_delete_issue(ctx: RunContext[StateDeps[tools.ProverbsState]], issue_key: str) -> str:
    return await call_mcp_jira_tool("jira_delete_issue", {"issue_key": issue_key})


async def jira_get_user_profile(ctx: RunContext[StateDeps[tools.ProverbsState]], username: str) -> str:
    return await call_mcp_jira_tool("jira_get_user_profile", {"username": username})


async def jira_create_issue_link(ctx: RunContext[StateDeps[tools.ProverbsState]], issue_key1: str, issue_key2: str, link_type: str) -> str:
    params = {"issue_key1": issue_key1, "issue_key2": issue_key2, "link_type": link_type}
    return await call_mcp_jira_tool("jira_create_issue_link", params)


async def jira_remove_issue_link(ctx: RunContext[StateDeps[tools.ProverbsState]], link_id: str) -> str:
    return await call_mcp_jira_tool("jira_remove_issue_link", {"link_id": link_id})


async def jira_get_link_types(ctx: RunContext[StateDeps[tools.ProverbsState]]) -> str:
    return await call_mcp_jira_tool("jira_get_link_types", {})


async def jira_add_worklog(ctx: RunContext[StateDeps[tools.ProverbsState]], issue_key: str, time_spent: str, comment: Optional[str] = None) -> str:
    params = {"issue_key": issue_key, "time_spent": time_spent}
    if comment:
        params["comment"] = comment
    return await call_mcp_jira_tool("jira_add_worklog", params)


async def jira_get_worklog(ctx: RunContext[StateDeps[tools.ProverbsState]], issue_key: str) -> str:
    return await call_mcp_jira_tool("jira_get_worklog", {"issue_key": issue_key})


async def jira_get_agile_boards(ctx: RunContext[StateDeps[tools.ProverbsState]]) -> str:
    return await call_mcp_jira_tool("jira_get_agile_boards", {})


async def jira_get_board_issues(ctx: RunContext[StateDeps[tools.ProverbsState]], board_id: str, limit: int = 20) -> str:
    return await call_mcp_jira_tool("jira_get_board_issues", {"board_id": board_id, "limit": limit})


async def jira_get_sprints_from_board(ctx: RunContext[StateDeps[tools.ProverbsState]], board_id: str) -> str:
    return await call_mcp_jira_tool("jira_get_sprints_from_board", {"board_id": board_id})


async def jira_get_sprint_issues(ctx: RunContext[StateDeps[tools.ProverbsState]], sprint_id: str, limit: int = 20) -> str:
    return await call_mcp_jira_tool("jira_get_sprint_issues", {"sprint_id": sprint_id, "limit": limit})


async def jira_create_version(ctx: RunContext[StateDeps[tools.ProverbsState]], project_key: str, version_name: str, description: Optional[str] = None) -> str:
    params = {"project_key": project_key, "version_name": version_name}
    if description:
        params["description"] = description
    return await call_mcp_jira_tool("jira_create_version", params)


async def jira_get_project_versions(ctx: RunContext[StateDeps[tools.ProverbsState]], project_key: str) -> str:
    return await call_mcp_jira_tool("jira_get_project_versions", {"project_key": project_key})


__all__ = [
    # helper
    "call_mcp_jira_tool",
    # jira functions
    "jira_search",
    "jira_get_issue",
    "jira_create_issue",
    "jira_add_comment",
    "jira_search_fields",
    "jira_get_project_issues",
    "jira_get_all_projects",
    "jira_update_issue",
    "jira_transition_issue",
    "jira_get_transitions",
    "jira_delete_issue",
    "jira_get_user_profile",
    "jira_create_issue_link",
    "jira_remove_issue_link",
    "jira_get_link_types",
    "jira_add_worklog",
    "jira_get_worklog",
    "jira_get_agile_boards",
    "jira_get_board_issues",
    "jira_get_sprints_from_board",
    "jira_get_sprint_issues",
    "jira_create_version",
    "jira_get_project_versions",
]
