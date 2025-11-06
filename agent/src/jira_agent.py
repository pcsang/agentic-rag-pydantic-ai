"""
Jira Agent for Issue Management
"""

from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps
from tools import ProverbsState
import httpx
import os

print("🎫 JIRA AGENT INITIALIZING")

MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:9000/mcp")

async def call_mcp(tool_name: str, params: dict) -> str:
    """Call MCP tool"""
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": "1", 
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": params}
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(MCP_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if "result" in result and "content" in result["result"]:
                return "\n".join(str(c.get("text", str(c))) for c in result["result"]["content"])
            return str(result.get("result", "No result"))
        return f"HTTP {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

jira_agent = Agent(
    name="jira_agent",
    model="openai:gpt-4o-mini", 
    system_prompt="You help with Jira. Use search_issues for queries.",
    retries=1
)

print("✅ Jira Agent created")

@jira_agent.tool
async def search_issues(
    ctx: RunContext[StateDeps[ProverbsState]],
    jql: str
) -> str:
    """Search Jira issues"""
    print(f"\n🔍 JIRA SEARCH: {jql}")
    result = await call_mcp("jira_search", {"jql_query": jql, "max_results": 10})
    return f"## Jira Search Results\n\n{result}\n\n**Query:** {jql}"

__all__ = ["jira_agent", "ProverbsState"]
