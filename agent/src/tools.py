from pydantic import BaseModel, Field
from typing import List, Optional
from pydantic_ai import RunContext
from pydantic_ai.ag_ui import StateDeps
from ag_ui.core import EventType, StateSnapshotEvent
import httpx
import os
import json


# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class ProverbsState(BaseModel):
    """List of the proverbs being written."""
    proverbs: List[str] = Field(
        default_factory=list,
        description='The list of already written proverbs',
    )


class RAGState(BaseModel):
    """State for RAG Agent"""
    query_count: int = Field(default=0, description="Number of queries processed")
    last_query: str = Field(default="", description="Last query processed")
    last_sources: List[str] = Field(default_factory=list, description="Sources from last query")
    last_strategy: str = Field(default="", description="Last strategy used")


# ============================================================================
# RAG HELPER FUNCTIONS - ASSESSMENT & GRADING
# ============================================================================

def assess_local_info(query: str, local_response: str, local_sources: List[str]) -> dict:
    """
    Assess the quality and sufficiency of local RAG information.
    
    Args:
        query: The original query
        local_response: Response from local RAG
        local_sources: Sources returned by local RAG
    
    Returns:
        Assessment dict with quality, confidence, and reasoning
    """
    print(f"🔍 Assessing local information for: {query[:50]}...")
    
    # Check if we have meaningful response and sources
    has_meaningful_response = bool(
        local_response and 
        len(local_response.strip()) > 20 and
        "no relevant" not in local_response.lower() and
        "not enough information" not in local_response.lower() and
        "unable to find" not in local_response.lower()
    )
    
    has_sources = bool(local_sources and len(local_sources) > 0)
    response_length = len(local_response.strip()) if local_response else 0
    
    # Case 1: No meaningful info at all
    if not has_meaningful_response and not has_sources:
        return {
            "quality": "none",
            "confidence": 0.0,
            "reasoning": "No meaningful response or sources from local RAG"
        }
    
    # Case 2: Check for partial/insufficient indicators
    insufficient_indicators = [
        "insufficient", "unclear", "more details", "please provide",
        "not specific", "limited information", "partial", "may be incomplete"
    ]
    
    is_partial = any(
        indicator in local_response.lower() 
        for indicator in insufficient_indicators
    )
    
    # Case 3: Assess based on response length and indicators
    if is_partial or response_length < 100:
        return {
            "quality": "partial",
            "confidence": 0.4,
            "reasoning": f"Local RAG has some info but seems incomplete (length: {response_length}, sources: {len(local_sources)})"
        }
    
    # Case 4: Sufficient information
    return {
        "quality": "sufficient",
        "confidence": 0.8,
        "reasoning": f"Local RAG provides comprehensive information (length: {response_length}, sources: {len(local_sources)})"
    }


def grade_response_quality(query: str, context: str) -> dict:
    """
    Grade the quality of response based on context using heuristics.
    Returns grading scores for relevancy, faithfulness, and whether web search is needed.
    
    Args:
        query: The user's query
        context: The provided context/response
    
    Returns:
        Grading dict with scores and flags
    """
    print(f"📊 Grading response quality...")
    
    # Calculate basic metrics
    query_words = set(query.lower().split())
    context_words = set(context.lower().split())
    
    # Calculate relevancy based on word overlap
    common_words = query_words & context_words
    relevancy = min(1.0, len(common_words) / max(len(query_words), 1) * 1.5)
    
    # Check for quality indicators
    quality_positive = [
        "based on", "according to", "documented", "showed",
        "evidence", "research", "study", "found", "demonstrated"
    ]
    quality_negative = [
        "unclear", "unknown", "unsure", "cannot say", "not found",
        "insufficient", "limited", "may not", "uncertain"
    ]
    
    positive_count = sum(1 for p in quality_positive if p in context.lower())
    negative_count = sum(1 for n in quality_negative if n in context.lower())
    
    # Faithfulness score based on quality indicators
    faithfulness = min(1.0, (positive_count * 0.1) - (negative_count * 0.1) + 0.5)
    faithfulness = max(0.0, min(1.0, faithfulness))
    
    # Context quality based on length and structure
    context_length = len(context.strip())
    has_structure = "\n" in context and "." in context
    
    context_quality = min(1.0, (context_length / 500) * 0.8)
    if has_structure:
        context_quality += 0.2
    context_quality = min(1.0, context_quality)
    
    # Decide if web search is needed
    needs_web = (
        relevancy < 0.5 or
        context_quality < 0.4 or
        negative_count > positive_count or
        context_length < 50
    )
    
    return {
        "relevancy": relevancy,
        "faithfulness": faithfulness,
        "context_quality": context_quality,
        "needs_web": needs_web,
        "reasoning": f"Relevancy: {relevancy:.2f}, Context quality: {context_quality:.2f}"
    }




def get_proverbs(ctx: RunContext[StateDeps["ProverbsState"]]) -> list[str]:
    """Get the current list of proverbs."""
    try:
        print(f"📖 Getting proverbs: {ctx.deps.state.proverbs}")
        return ctx.deps.state.proverbs
    except Exception:
        return []


async def add_proverbs(ctx: RunContext[StateDeps["ProverbsState"]], proverbs: list[str]) -> StateSnapshotEvent:
    ctx.deps.state.proverbs.extend(proverbs)
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


async def set_proverbs(ctx: RunContext[StateDeps["ProverbsState"]], proverbs: list[str]) -> StateSnapshotEvent:
    ctx.deps.state.proverbs = proverbs
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


def get_weather(ctx: RunContext[StateDeps["ProverbsState"]], location: str) -> str:
    """Get the weather for a given location. Ensure location is fully spelled out."""
    return f"The weather in {location} is sunny."


def emit_state(ctx: RunContext[StateDeps["ProverbsState"]]) -> StateSnapshotEvent:
    """Emit the current agent state as a StateSnapshotEvent so frontends can render it."""
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


# ============================================================================
# JIRA TOOLS - Project Management
# ============================================================================

async def jira_search_issues(jql_query: str, max_results: int = 50) -> dict:
    """
    Search Jira issues using JQL (Jira Query Language) via MCP server
    
    Args:
        jql_query: JQL query string (e.g., "project = PROJ AND status = Open")
        max_results: Maximum number of issues to return
    
    Returns:
        List of matching issues with details
    """
    try:
        print(f"🔍 Jira JQL Search via MCP: {jql_query}")
        
        mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:9000/mcp").rstrip("/")
        
        # Make the request to MCP server
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            response = await client.post(
                f"{mcp_url}/tool/call",
                json={
                    "tool": "jira_search_issues",
                    "args": {
                        "jql": jql_query,
                        "max_results": max_results
                    }
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                issues = data.get("content", [])
                
                return {
                    "issues": issues,
                    "total": len(issues),
                    "success": True,
                    "message": f"Found {len(issues)} issues"
                }
            else:
                print(f"❌ MCP API error: {response.status_code} - {response.text}")
                return {
                    "issues": [],
                    "total": 0,
                    "success": False,
                    "message": f"MCP API error: {response.status_code} - {response.text}"
                }
    except Exception as e:
        print(f"❌ Jira search error: {str(e)}")
        return {
            "issues": [],
            "total": 0,
            "success": False,
            "message": f"Error searching Jira: {str(e)}"
        }


async def jira_create_issue(
    project_key: str,
    summary: str,
    description: str,
    issue_type: str = "Task",
    priority: str = "Medium",
    assignee: Optional[str] = None
) -> dict:
    """
    Create a new Jira issue via MCP server
    
    Args:
        project_key: Project key (e.g., "PROJ")
        summary: Issue title/summary
        description: Detailed description
        issue_type: Type (Task, Bug, Story, Epic, etc.)
        priority: Priority level (Low, Medium, High, Critical)
        assignee: Username to assign (optional)
    
    Returns:
        Created issue details
    """
    try:
        print(f"📝 Creating Jira issue via MCP in {project_key}: {summary}")
        
        mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:9000/mcp").rstrip("/")
        
        # Make the request to MCP server
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            response = await client.post(
                f"{mcp_url}/tool/call",
                json={
                    "tool": "jira_create_issue",
                    "args": {
                        "project_key": project_key,
                        "summary": summary,
                        "description": description,
                        "issue_type": issue_type,
                        "priority": priority,
                        "assignee": assignee
                    }
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("content", [])
                if content and isinstance(content, list) and len(content) > 0:
                    result = content[0]
                    return {
                        "key": result.get("key"),
                        "success": True,
                        "message": f"Created issue {result.get('key')}: {summary}"
                    }
                else:
                    return {
                        "key": None,
                        "success": False,
                        "message": "No response from MCP server"
                    }
            else:
                print(f"❌ MCP API error: {response.status_code} - {response.text}")
                return {
                    "key": None,
                    "success": False,
                    "message": f"MCP API error: {response.status_code}"
                }
    except Exception as e:
        print(f"❌ Jira create issue error: {str(e)}")
        return {
            "key": None,
            "success": False,
            "message": f"Error creating issue: {str(e)}"
        }


async def jira_update_issue(
    issue_key: str,
    status: Optional[str] = None,
    assignee: Optional[str] = None,
    priority: Optional[str] = None,
    comment: Optional[str] = None,
    labels: Optional[List[str]] = None
) -> dict:
    """
    Update an existing Jira issue via MCP server
    
    Args:
        issue_key: Issue key (e.g., "PROJ-123")
        status: New status (optional)
        assignee: New assignee (optional)
        priority: New priority (optional)
        comment: Add a comment (optional)
        labels: Add/update labels (optional)
    
    Returns:
        Update confirmation
    """
    try:
        print(f"✏️ Updating Jira issue via MCP: {issue_key}")
        
        mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:9000/mcp").rstrip("/")
        
        # Make the request to MCP server
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            response = await client.post(
                f"{mcp_url}/tool/call",
                json={
                    "tool": "jira_update_issue",
                    "args": {
                        "issue_key": issue_key,
                        "status": status,
                        "assignee": assignee,
                        "priority": priority,
                        "comment": comment,
                        "labels": labels
                    }
                }
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "message": f"Updated issue {issue_key}"
                }
            else:
                print(f"❌ MCP API error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "message": f"MCP API error: {response.status_code} - {response.text}"
                }
    except Exception as e:
        print(f"❌ Jira update issue error: {str(e)}")
        return {
            "success": False,
            "message": f"Error updating issue: {str(e)}"
        }


async def jira_get_project_info(project_key: str) -> dict:
    """
    Get information about a Jira project via MCP server
    
    Args:
        project_key: Project key (e.g., "PROJ")
    
    Returns:
        Project details including name, description, issue types, etc.
    """
    try:
        print(f"📊 Getting Jira project info via MCP: {project_key}")
        
        mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:9000/mcp").rstrip("/")
        
        # Make the request to MCP server
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            response = await client.post(
                f"{mcp_url}/tool/call",
                json={
                    "tool": "jira_get_project",
                    "args": {
                        "project_key": project_key
                    }
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("content", [])
                if content and isinstance(content, list) and len(content) > 0:
                    project = content[0]
                    return {
                        "project": project,
                        "success": True,
                        "message": f"Retrieved project {project_key}"
                    }
                else:
                    return {
                        "project": None,
                        "success": False,
                        "message": "No response from MCP server"
                    }
            else:
                print(f"❌ MCP API error: {response.status_code} - {response.text}")
                return {
                    "project": None,
                    "success": False,
                    "message": f"MCP API error: {response.status_code}"
                }
    except Exception as e:
        print(f"❌ Jira get project info error: {str(e)}")
        return {
            "project": None,
            "success": False,
            "message": f"Error getting project info: {str(e)}"
        }


async def jira_get_issue(issue_key: str) -> dict:
    """
    Get detailed information about a specific Jira issue via MCP server
    
    Args:
        issue_key: Issue key (e.g., "PROJ-123")
    
    Returns:
        Issue details including status, assignee, description, comments, etc.
    """
    try:
        print(f"📄 Getting Jira issue via MCP: {issue_key}")
        
        mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:9000/mcp").rstrip("/")
        
        # Make the request to MCP server
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            response = await client.post(
                f"{mcp_url}/tool/call",
                json={
                    "tool": "jira_get_issue",
                    "args": {
                        "issue_key": issue_key
                    }
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("content", [])
                if content and isinstance(content, list) and len(content) > 0:
                    issue = content[0]
                    return {
                        "issue": issue,
                        "success": True,
                        "message": f"Retrieved issue {issue_key}"
                    }
                else:
                    return {
                        "issue": None,
                        "success": False,
                        "message": "No response from MCP server"
                    }
            else:
                print(f"❌ MCP API error: {response.status_code} - {response.text}")
                return {
                    "issue": None,
                    "success": False,
                    "message": f"MCP API error: {response.status_code}"
                }
    except Exception as e:
        print(f"❌ Jira get issue error: {str(e)}")
        return {
            "issue": None,
            "success": False,
            "message": f"Error getting issue: {str(e)}"
        }


# ============================================================================
# RAG TOOLS - LightRAG Integration
# ============================================================================

async def query_lightrag(query: str, mode: str = "hybrid", top_k: int = 5) -> dict:
    """
    Query local LightRAG instance
    
    Args:
        query: The natural language query
        mode: "local", "hybrid", or "global" search mode
        top_k: Number of top results to return
    
    Returns:
        Response containing answer and sources
    """
    host = os.getenv("LIGHT_RAG_HOST", "http://localhost:9621")
    url = f"{host}/query"
    
    print(f"🔌 Connecting to LightRAG at: {url}")
    print(f"📝 Query: {query} (mode: {mode})")
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "mode": mode,
        "top_k": top_k
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print(f"📤 Sending payload: {payload}")
            response = await client.post(url, headers=headers, json=payload)
            
            print(f"📥 LightRAG Status Code: {response.status_code}")
            print(f"📥 LightRAG Response: {response.text[:500]}")
            
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ LightRAG parsed response: {result}")
            
            if not result or "response" not in result:
                print("⚠️ No 'response' field in LightRAG result")
                return {
                    "response": "No relevant documentation found in local RAG",
                    "sources": [],
                    "success": False
                }
            
            # Extract sources from references
            sources = []
            if "references" in result and result["references"]:
                sources = [ref.get("file_path", ref.get("reference_id", "Unknown source")) 
                         for ref in result["references"]]
            
            print(f"✅ Extracted {len(sources)} sources from LightRAG")
            
            return {
                "response": result.get("response", ""),
                "sources": sources,
                "success": True
            }
    except Exception as e:
        print(f"❌ LightRAG Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "response": f"Error querying local RAG: {str(e)}",
            "sources": [],
            "success": False
        }


# ============================================================================
# RAG TOOLS - Tavily Web Search
# ============================================================================

async def search_web_tavily(query: str, max_results: int = 5) -> dict:
    """
    Search the web using Tavily API
    
    Args:
        query: Search query
        max_results: Maximum number of results
    
    Returns:
        Search results with sources
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "results": [],
            "success": False,
            "error": "Tavily API key not configured"
        }
    
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "include_answer": True
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            print(f"🌐 Tavily search results: {len(result.get('results', []))} results")
            
            return {
                "results": result.get("results", []),
                "success": True,
                "answer": result.get("answer", "")
            }
    except Exception as e:
        print(f"❌ Tavily Search Error: {e}")
        return {
            "results": [],
            "success": False,
            "error": str(e)
        }


__all__ = [
    # States
    "ProverbsState",
    "RAGState",
    # Proverbs/Demo tools
    "get_proverbs",
    "add_proverbs",
    "set_proverbs",
    "get_weather",
    "emit_state",
    # Jira tools
    "jira_search_issues",
    "jira_create_issue",
    "jira_update_issue",
    "jira_get_project_info",
    "jira_get_issue",
    # RAG tools
    "query_lightrag",
    "search_web_tavily",
]
