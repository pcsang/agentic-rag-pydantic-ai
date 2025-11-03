from textwrap import dedent
from pydantic import BaseModel, Field
from pydantic_ai import Agent, WebSearchTool, RunContext
from pydantic_ai.ag_ui import StateDeps
from ag_ui.core import EventType, StateSnapshotEvent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

# load environment variables from agent/.env explicitly
from pathlib import Path
from dotenv import load_dotenv
import os

# Import for async Jira tool support
import asyncio
import json
from typing import Optional
from fastmcp.client import Client

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# MCP configuration - prefer FASTMCP_URL or MCP_SERVER_URL, fall back to localhost:9000
mcp_url = os.getenv("FASTMCP_URL") or os.getenv("MCP_SERVER_URL") or "http://localhost:9000/mcp"

# =====
# State
# =====
class ProverbsState(BaseModel):
  """List of the proverbs being written."""
  proverbs: list[str] = Field(
    default_factory=list,
    description='The list of already written proverbs',
  )

# =====
# Agent
# =====
agent = Agent(
  model = OpenAIResponsesModel('gpt-4.1-mini'),
  builtin_tools=[WebSearchTool()],
  model_settings=OpenAIResponsesModelSettings(openai_include_web_search_sources=True),
  deps_type=StateDeps[ProverbsState],
  system_prompt=dedent("""
    You are a helpful assistant that helps manage and discuss proverbs.
    
    The user has a list of proverbs that you can help them manage.
    You have tools available to add, set, or retrieve proverbs from the list.
    
    When discussing proverbs, ALWAYS use the get_proverbs tool to see the current list before
    mentioning, updating, or discussing proverbs with the user.
    When you update the proverbs list (for example via `add_proverbs` or `set_proverbs`),
    you MUST call the `emit_state` tool after the update so the frontend can render the
    updated state in the chat.
    
    You also have access to web search capabilities to find current information about proverbs
    and their origins when needed.
  """).strip()
)


def run_web_search(query: str, force_use_web_search: bool = True):
  """Run the agent on `query` and print the textual output plus any web-search link references.

  If `force_use_web_search` is True the prompt is prefixed with an explicit instruction to use the
  web search tool and to list source links — that encourages the model to invoke the builtin tool.

  The function prints debug information (including the raw `result.response`) to help diagnose
  why sources may not be present for your provider.
  """
  prompt = query
  if force_use_web_search:
    prompt = "Please use the built-in web search tool and cite source URLs. " + query

  result = agent.run_sync(prompt)
  print("\n=== Agent Output ===")
  print(result.output)

  # Print raw response for debugging (shows builtin_tool_calls structure when present)
  print("\n=== Raw result.response (debug) ===")
  try:
    # Some ModelResponse objects are large; repr once
    print(repr(result.response))
  except Exception:
    try:
      print(result.response)
    except Exception:
      pass

  # Try to surface web-search sources from builtin tool return parts
  bt_calls = getattr(result.response, 'builtin_tool_calls', None)
  if not bt_calls:
    print("\n(no builtin tool calls returned) — the model did not invoke any built-in tools for this prompt.")
    print("Try making the request more explicitly: 'Use the web search tool and list sources for:' + your query")
    return result

  print("\n=== Builtin tool calls / sources ===")
  for call_part, return_part in bt_calls:
    print("\n-- Tool call:", getattr(call_part, 'tool_name', call_part))
    # Try to access return_part content safely
    try:
      rp = getattr(return_part, 'content', return_part)
    except Exception:
      rp = getattr(return_part, '__dict__', return_part)

    # Attempt to find urls in common locations
    urls = []
    if isinstance(rp, dict):
      for k in ('search_results', 'results', 'content', 'links', 'sources', 'source_urls'):
        if k in rp and isinstance(rp[k], list):
          for item in rp[k]:
            if isinstance(item, dict) and 'url' in item:
              urls.append(item['url'])
            elif isinstance(item, str) and item.startswith('http'):
              urls.append(item)

    # Fallback: search the string representation for http(s) links
    if not urls:
      import re
      text = str(rp)
      urls = re.findall(r'https?://\S+', text)

    if urls:
      print(f"Found {len(urls)} source link(s):")
      for u in urls:
        print(f" - {u}")
    else:
      print("No explicit URLs found in this builtin tool return part.\nReturn part content:\n", rp)

  return result

# =====
# Tools
# =====
@agent.tool
def get_proverbs(ctx: RunContext[StateDeps[ProverbsState]]) -> list[str]:
  """Get the current list of proverbs."""
  print(f"📖 Getting proverbs: {ctx.deps.state.proverbs}")
  return ctx.deps.state.proverbs

@agent.tool
async def add_proverbs(ctx: RunContext[StateDeps[ProverbsState]], proverbs: list[str]) -> StateSnapshotEvent:
  ctx.deps.state.proverbs.extend(proverbs)
  return StateSnapshotEvent(
    type=EventType.STATE_SNAPSHOT,
    snapshot=ctx.deps.state,
  )

@agent.tool
async def set_proverbs(ctx: RunContext[StateDeps[ProverbsState]], proverbs: list[str]) -> StateSnapshotEvent:
  ctx.deps.state.proverbs = proverbs
  return StateSnapshotEvent(
    type=EventType.STATE_SNAPSHOT,
    snapshot=ctx.deps.state,
  )


@agent.tool
def get_weather(_: RunContext[StateDeps[ProverbsState]], location: str) -> str:
  """Get the weather for a given location. Ensure location is fully spelled out."""
  return f"The weather in {location} is sunny."


@agent.tool
def emit_state(ctx: RunContext[StateDeps[ProverbsState]]) -> StateSnapshotEvent:
  """Emit the current agent state as a StateSnapshotEvent so frontends can render it.

  This tool is intended to be called by the agent (or in tests) after state mutations
  to ensure the UI receives a visible state snapshot in the chat.
  """
  return StateSnapshotEvent(
    type=EventType.STATE_SNAPSHOT,
    snapshot=ctx.deps.state,
  )


# =====
# MCP Jira Tools
# =====
async def call_mcp_jira_tool(tool_name: str, params: dict) -> str:
  """Helper to call a Jira tool via the local MCP server."""
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


@agent.tool
async def jira_search(ctx: RunContext[StateDeps[ProverbsState]], jql: str, limit: int = 10) -> str:
  """Search Jira issues using JQL (Jira Query Language).
  
  Args:
    jql: JQL query string (e.g., 'project = PROJ AND status = "In Progress"')
    limit: Maximum number of results (default 10, max 50)
  
  Returns:
    JSON string with search results
  """
  return await call_mcp_jira_tool("jira_search", {"jql": jql, "limit": min(limit, 50)})


@agent.tool
async def jira_get_issue(ctx: RunContext[StateDeps[ProverbsState]], issue_key: str, fields: str = "summary,status,assignee,description,priority") -> str:
  """Get details of a specific Jira issue.
  
  Args:
    issue_key: Jira issue key (e.g., 'PROJ-123')
    fields: Comma-separated fields to return
  
  Returns:
    JSON string with issue details
  """
  return await call_mcp_jira_tool("jira_get_issue", {"issue_key": issue_key, "fields": fields})


@agent.tool
async def jira_create_issue(
  ctx: RunContext[StateDeps[ProverbsState]],
  project_key: str,
  summary: str,
  issue_type: str,
  description: Optional[str] = None,
  assignee: Optional[str] = None,
) -> str:
  """Create a new Jira issue.
  
  Args:
    project_key: Jira project key (e.g., 'PROJ')
    summary: Issue summary/title
    issue_type: Type of issue (e.g., 'Task', 'Bug', 'Story')
    description: Optional issue description
    assignee: Optional assignee email or username
  
  Returns:
    JSON string with created issue details
  """
  params = {
    "project_key": project_key,
    "summary": summary,
    "issue_type": issue_type,
  }
  if description:
    params["description"] = description
  if assignee:
    params["assignee"] = assignee
  
  return await call_mcp_jira_tool("jira_create_issue", params)


@agent.tool
async def jira_add_comment(ctx: RunContext[StateDeps[ProverbsState]], issue_key: str, comment: str) -> str:
  """Add a comment to a Jira issue.
  
  Args:
    issue_key: Jira issue key (e.g., 'PROJ-123')
    comment: Comment text
  
  Returns:
    JSON string with comment details
  """
  return await call_mcp_jira_tool("jira_add_comment", {"issue_key": issue_key, "comment": comment})


@agent.tool
async def jira_search_fields(ctx: RunContext[StateDeps[ProverbsState]], keyword: str = "", limit: int = 10) -> str:
  """Search and list Jira fields by keyword.
  
  Args:
    keyword: Keyword to search for (empty to list all)
    limit: Maximum number of results (default 10)
  
  Returns:
    JSON string with field definitions
  """
  return await call_mcp_jira_tool("jira_search_fields", {"keyword": keyword, "limit": limit})


@agent.tool
async def jira_get_project_issues(ctx: RunContext[StateDeps[ProverbsState]], project_key: str, limit: int = 10) -> str:
  """Get all issues in a Jira project.
  
  Args:
    project_key: Jira project key (e.g., 'PROJ')
    limit: Maximum number of results (default 10, max 50)
  
  Returns:
    JSON string with project issues
  """
  return await call_mcp_jira_tool("jira_get_project_issues", {"project_key": project_key, "limit": min(limit, 50)})


# =====
# Additional Jira Tools (for advanced workflows)
# =====

@agent.tool
async def jira_get_all_projects(ctx: RunContext[StateDeps[ProverbsState]]) -> str:
  """Get all available Jira projects in the instance.
  
  Returns:
    JSON string with list of all projects
  """
  return await call_mcp_jira_tool("jira_get_all_projects", {})


@agent.tool
async def jira_update_issue(
  ctx: RunContext[StateDeps[ProverbsState]],
  issue_key: str,
  summary: Optional[str] = None,
  description: Optional[str] = None,
  assignee: Optional[str] = None,
  status: Optional[str] = None,
  priority: Optional[str] = None,
) -> str:
  """Update a Jira issue with new values.
  
  Args:
    issue_key: Jira issue key (e.g., 'PROJ-123')
    summary: Optional new summary/title
    description: Optional new description
    assignee: Optional new assignee
    status: Optional new status (e.g., 'In Progress', 'Done')
    priority: Optional new priority (e.g., 'High', 'Medium', 'Low')
  
  Returns:
    JSON string with update confirmation
  """
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


@agent.tool
async def jira_transition_issue(ctx: RunContext[StateDeps[ProverbsState]], issue_key: str, transition_name: str) -> str:
  """Move a Jira issue to a new status/state.
  
  Args:
    issue_key: Jira issue key (e.g., 'PROJ-123')
    transition_name: Name of the transition (e.g., 'Start Progress', 'Done')
  
  Returns:
    JSON string with transition confirmation
  """
  return await call_mcp_jira_tool("jira_transition_issue", {"issue_key": issue_key, "transition_name": transition_name})


@agent.tool
async def jira_get_transitions(ctx: RunContext[StateDeps[ProverbsState]], issue_key: str) -> str:
  """Get available transitions (status changes) for an issue.
  
  Args:
    issue_key: Jira issue key (e.g., 'PROJ-123')
  
  Returns:
    JSON string with available transitions
  """
  return await call_mcp_jira_tool("jira_get_transitions", {"issue_key": issue_key})


@agent.tool
async def jira_delete_issue(ctx: RunContext[StateDeps[ProverbsState]], issue_key: str) -> str:
  """Delete a Jira issue (requires permission).
  
  Args:
    issue_key: Jira issue key (e.g., 'PROJ-123')
  
  Returns:
    JSON string with deletion confirmation
  """
  return await call_mcp_jira_tool("jira_delete_issue", {"issue_key": issue_key})


@agent.tool
async def jira_get_user_profile(ctx: RunContext[StateDeps[ProverbsState]], username: str) -> str:
  """Get profile information for a Jira user.
  
  Args:
    username: Jira username or email
  
  Returns:
    JSON string with user profile details
  """
  return await call_mcp_jira_tool("jira_get_user_profile", {"username": username})


@agent.tool
async def jira_create_issue_link(
  ctx: RunContext[StateDeps[ProverbsState]],
  issue_key1: str,
  issue_key2: str,
  link_type: str,
) -> str:
  """Link two Jira issues together.
  
  Args:
    issue_key1: First issue key (e.g., 'PROJ-123')
    issue_key2: Second issue key (e.g., 'PROJ-456')
    link_type: Type of link (e.g., 'relates to', 'blocks', 'duplicates')
  
  Returns:
    JSON string with link confirmation
  """
  return await call_mcp_jira_tool("jira_create_issue_link", {
    "issue_key1": issue_key1,
    "issue_key2": issue_key2,
    "link_type": link_type,
  })


@agent.tool
async def jira_remove_issue_link(ctx: RunContext[StateDeps[ProverbsState]], link_id: str) -> str:
  """Remove a link between two Jira issues.
  
  Args:
    link_id: ID of the issue link to remove
  
  Returns:
    JSON string with removal confirmation
  """
  return await call_mcp_jira_tool("jira_remove_issue_link", {"link_id": link_id})


@agent.tool
async def jira_get_link_types(ctx: RunContext[StateDeps[ProverbsState]]) -> str:
  """Get all available link types in Jira.
  
  Returns:
    JSON string with link type definitions
  """
  return await call_mcp_jira_tool("jira_get_link_types", {})


@agent.tool
async def jira_add_worklog(
  ctx: RunContext[StateDeps[ProverbsState]],
  issue_key: str,
  time_spent: str,
  comment: Optional[str] = None,
) -> str:
  """Add work log entry (time tracking) to an issue.
  
  Args:
    issue_key: Jira issue key (e.g., 'PROJ-123')
    time_spent: Time spent string (e.g., '2h', '30m', '1d 2h')
    comment: Optional comment about the work
  
  Returns:
    JSON string with worklog confirmation
  """
  params = {"issue_key": issue_key, "time_spent": time_spent}
  if comment:
    params["comment"] = comment
  
  return await call_mcp_jira_tool("jira_add_worklog", params)


@agent.tool
async def jira_get_worklog(ctx: RunContext[StateDeps[ProverbsState]], issue_key: str) -> str:
  """Get work log entries for a Jira issue.
  
  Args:
    issue_key: Jira issue key (e.g., 'PROJ-123')
  
  Returns:
    JSON string with worklog entries
  """
  return await call_mcp_jira_tool("jira_get_worklog", {"issue_key": issue_key})


@agent.tool
async def jira_get_agile_boards(ctx: RunContext[StateDeps[ProverbsState]]) -> str:
  """Get all Agile/Scrum boards in the Jira instance.
  
  Returns:
    JSON string with list of boards
  """
  return await call_mcp_jira_tool("jira_get_agile_boards", {})


@agent.tool
async def jira_get_board_issues(ctx: RunContext[StateDeps[ProverbsState]], board_id: str, limit: int = 20) -> str:
  """Get issues on an Agile board.
  
  Args:
    board_id: Agile board ID (e.g., '1')
    limit: Maximum number of results (default 20)
  
  Returns:
    JSON string with board issues
  """
  return await call_mcp_jira_tool("jira_get_board_issues", {"board_id": board_id, "limit": limit})


@agent.tool
async def jira_get_sprints_from_board(ctx: RunContext[StateDeps[ProverbsState]], board_id: str) -> str:
  """Get sprints associated with an Agile board.
  
  Args:
    board_id: Agile board ID (e.g., '1')
  
  Returns:
    JSON string with sprint list
  """
  return await call_mcp_jira_tool("jira_get_sprints_from_board", {"board_id": board_id})


@agent.tool
async def jira_get_sprint_issues(ctx: RunContext[StateDeps[ProverbsState]], sprint_id: str, limit: int = 20) -> str:
  """Get issues in a Jira sprint.
  
  Args:
    sprint_id: Sprint ID (e.g., '1')
    limit: Maximum number of results (default 20)
  
  Returns:
    JSON string with sprint issues
  """
  return await call_mcp_jira_tool("jira_get_sprint_issues", {"sprint_id": sprint_id, "limit": limit})


@agent.tool
async def jira_create_version(
  ctx: RunContext[StateDeps[ProverbsState]],
  project_key: str,
  version_name: str,
  description: Optional[str] = None,
) -> str:
  """Create a new version/release in a Jira project.
  
  Args:
    project_key: Jira project key (e.g., 'PROJ')
    version_name: Name of the version (e.g., 'v1.2.3')
    description: Optional version description
  
  Returns:
    JSON string with created version details
  """
  params = {"project_key": project_key, "version_name": version_name}
  if description:
    params["description"] = description
  
  return await call_mcp_jira_tool("jira_create_version", params)


@agent.tool
async def jira_get_project_versions(ctx: RunContext[StateDeps[ProverbsState]], project_key: str) -> str:
  """Get all versions/releases for a project.
  
  Args:
    project_key: Jira project key (e.g., 'PROJ')
  
  Returns:
    JSON string with project versions
  """
  return await call_mcp_jira_tool("jira_get_project_versions", {"project_key": project_key})
