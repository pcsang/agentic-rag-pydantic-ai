"""
Advanced RAG System - Structured Architecture & Tools
=====================================================

Four-Layer Agentic RAG Architecture:

Layer 1: Document Processing & Assessment
  - assess_local_info(): Evaluates local RAG responses
  - Returns AssessmentResult with quality (none/partial/sufficient) and confidence
  - Determines if local information is sufficient or needs augmentation
  - Input: query, local_response, sources
  - Output: AssessmentResult(quality, confidence, reasoning)

Layer 2: RAG Agent Self-Grading  
  - grade_response_quality(): Scores response quality with explicit metrics
  - Returns GradingResult with three independent metrics:
    * Relevancy (0-1): How well response addresses the query
    * Faithfulness (0-1): How grounded response is in evidence
    * Context Quality (0-1): Depth and structure of information
  - Includes explicit needs_web_search flag for dynamic augmentation
  - Thresholds:
    * relevancy < 0.6 → consider web search
    * context_quality < 0.6 → consider web search
    * needs_web_search flag → explicit indicator

Layer 3: Local Retrieval Tool (LightRAG)
  - query_lightrag(): Interface to local document RAG at http://localhost:9621
  - Modes: local, global, hybrid, naive, mix, bypass
  - Timeout: 60 seconds
  - Returns: response text, sources list, success flag
  - No internet required - fully local operation

Layer 4: Web Search Integration (Tavily)
  - search_web_tavily(): Optional augmentation with web search
  - Used when local information is insufficient
  - Requires TAVILY_API_KEY environment variable
  - Returns: answer text, sources, success flag

Three-Strategy Decision Logic:
  - WEB_ONLY: quality == 'none' → web search only
  - LOCAL_ONLY: quality == 'sufficient' AND confidence >= 0.7 → local only
  - GENERATE_GRADE_AUGMENT: quality == 'partial' → grade and decide

Structured Types:
  - RAGDeps: Dependencies with question, context, query_id fields
  - AssessmentResult: Quality assessment from Layer 1
  - GradingResult: Explicit grading metrics from Layer 2
  - RAGState: Agent state tracking with grades and assessment history

Dynamic Context Augmentation Rules:
  - Augment if: relevancy < 0.6 OR context_quality < 0.6 OR confidence < 0.6
  - Augment if: needs_web_search flag is True from grading
  - Local threshold: 1000+ chars = sufficient (Layer 1)
  - Grading threshold: >= 0.7 quality across metrics (Layer 2)
"""

from pydantic import BaseModel, Field
from typing import List, Optional, TYPE_CHECKING
from pydantic_ai import RunContext

# The `pydantic_ai.ag_ui` module is an optional UI extra. When it's not
# installed the import would raise ModuleNotFoundError. Use TYPE_CHECKING so
# static type checkers still see the real type, and provide a tiny runtime
# fallback that supports subscription (StateDeps[T]).
if TYPE_CHECKING:
    from pydantic_ai.ag_ui import StateDeps  # type: ignore
else:
    class _StateDepsProxy:
        def __class_getitem__(cls, item):
            return item

    StateDeps = _StateDepsProxy
from ag_ui.core import EventType, StateSnapshotEvent
import httpx
import os
import json
from dataclasses import dataclass


# ============================================================================
# STRUCTURED DEPENDENCIES & STATE DEFINITIONS
# ============================================================================

@dataclass
class Deps:
    """
    Structured dependencies for RAG system using Pydantic for type safety
    Follows pydantic-ai pattern for passing context through the agent
    """
    question: str | None = None
    context: str | None = None
    query_id: str = ""

# Backward compatibility alias
RAGDeps = Deps


class ProverbsState(BaseModel):
    """List of the proverbs being written."""
    proverbs: List[str] = Field(
        default_factory=list,
        description='The list of already written proverbs',
    )


class RAGState(BaseModel):
    """Enhanced State for RAG Agent"""
    query_count: int = Field(default=0, description="Number of queries processed")
    last_query: str = Field(default="", description="Last query processed")
    last_sources: List[str] = Field(default_factory=list, description="Sources from last query")
    last_strategy: str = Field(default="", description="Last strategy used")
    last_grades: dict = Field(default_factory=dict, description="Last grading scores")
    last_assessment: dict = Field(default_factory=dict, description="Last assessment result")


class GradingResult(BaseModel):
    """Self-grading result structure"""
    relevancy: float = Field(..., ge=0, le=1, description="Relevancy score 0-1")
    faithfulness: float = Field(..., ge=0, le=1, description="Faithfulness to context 0-1")
    context_quality: float = Field(..., ge=0, le=1, description="Context quality 0-1")
    needs_web_search: bool = Field(..., description="Whether web search is needed")
    explanation: str = Field(..., description="Explanation of grades")


class AssessmentResult(BaseModel):
    """Local information assessment result"""
    quality: str = Field(..., description="Quality level: none, partial, sufficient")
    confidence: float = Field(..., ge=0, le=1, description="Confidence 0-1")
    reasoning: str = Field(..., description="Assessment reasoning")


# ============================================================================
# RAG HELPER FUNCTIONS - ASSESSMENT & GRADING
# ============================================================================

def assess_local_info(query: str, local_response: str, local_sources: List[str]) -> AssessmentResult:
    """
    Assess the quality of local RAG information.
    
    Layer 1: Document Processing & Assessment
    Evaluates if the local information is sufficient or needs augmentation.
    
    Args:
        query: The original query
        local_response: Response from local RAG
        local_sources: Sources returned by local RAG
    
    Returns:
        AssessmentResult with quality ('none', 'partial', 'sufficient'), confidence (0-1), and reasoning
    """
    print(f"🔍 Assessing local information for: {query[:50]}...")
    
    response_length = len(local_response.strip()) if local_response else 0
    sources_count = len(local_sources) if local_sources else 0
    
    # Check for negative indicators (signals of NO information)
    negative_indicators = [
        "no relevant",
        "not enough information",
        "unable to find",
        "i do not have enough information",
        "i don't have information",
        "no information",
        "cannot answer",
        "i don't have"
    ]
    
    has_negative_indicator = any(
        indicator in local_response.lower() 
        for indicator in negative_indicators
    )
    
    # === CASE 1: NO INFORMATION ===
    # Empty response, too short, or explicitly says no info
    if (not local_response or 
        response_length < 30 or 
        has_negative_indicator or
        (sources_count == 0 and response_length < 50)):
        
        return AssessmentResult(
            quality="none",
            confidence=0.0,
            reasoning="No relevant information found in local RAG"
        )
    
    # === CASE 2: SUFFICIENT INFORMATION ===
    # Good response with multiple sources or substantial length
    if response_length >= 500 and sources_count >= 2:
        return AssessmentResult(
            quality="sufficient",
            confidence=0.85,
            reasoning=f"Local RAG provides comprehensive information (length: {response_length} chars, sources: {sources_count})"
        )
    
    if response_length >= 1000:
        # Very long response is sufficient even with fewer sources
        return AssessmentResult(
            quality="sufficient",
            confidence=0.80,
            reasoning=f"Local RAG provides comprehensive information (length: {response_length} chars, sources: {sources_count})"
        )
    
    # === CASE 3: PARTIAL INFORMATION ===
    # Has some info but not comprehensive
    if response_length >= 100 and sources_count >= 1:
        confidence = 0.5 + (min(response_length, 500) / 500) * 0.25
        return AssessmentResult(
            quality="partial",
            confidence=min(0.75, confidence),
            reasoning=f"Local RAG provides partial information (length: {response_length} chars, sources: {sources_count})"
        )
    
    # Default: insufficient information
    return AssessmentResult(
        quality="partial",
        confidence=0.3,
        reasoning=f"Local RAG provides limited information (length: {response_length} chars, sources: {sources_count})"
    )


def grade_response_quality(query: str, context: str) -> GradingResult:
    """
    Grade the quality of a RAG response using self-grading mechanism.
    
    Layer 2: RAG Agent Self-Grading
    Evaluates responses based on three criteria and returns structured scores.
    
    The system evaluates its own responses based on three criteria:
    - Relevancy (0-1): How well the response addresses the query
    - Faithfulness (0-1): How grounded the response is in the provided context
    - Context Quality (0-1): Depth, structure, and completeness of information
    
    Args:
        query: The original user query
        context: The context/response to grade
    
    Returns:
        GradingResult with scores for each metric, needs_web_search flag, and explanation
        
    Example Output Format:
        {
            "Relevancy": 0.9,
            "Faithfulness": 0.95,
            "Context Quality": 0.85,
            "Needs Web Search": false,
            "Explanation": "Response directly addresses the question..."
        }
    
    Dynamic Context Augmentation Logic:
        if grades["Needs Web Search"]:
            web_results = await websearch_tool(query)
            # Augment context with web results
    """
    print(f"📊 Grading response quality...")
    
    # Calculate basic metrics
    query_words = set(query.lower().split())
    context_words = set(context.lower().split())
    
    # === RELEVANCY: How well the context addresses the query ===
    common_words = query_words & context_words
    relevancy = min(1.0, len(common_words) / max(len(query_words), 1) * 1.5)
    
    # === FAITHFULNESS: How grounded the context is in evidence ===
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
    
    # Faithfulness: evidence-based vs speculative language
    faithfulness = min(1.0, (positive_count * 0.1) - (negative_count * 0.1) + 0.5)
    faithfulness = max(0.0, min(1.0, faithfulness))
    
    # === CONTEXT_QUALITY: Depth and structure of information ===
    context_length = len(context.strip())
    has_structure = "\n" in context and "." in context
    
    context_quality = min(1.0, (context_length / 500) * 0.8)
    if has_structure:
        context_quality += 0.2
    context_quality = min(1.0, context_quality)
    
    # === NEEDS_WEB_SEARCH: Dynamic augmentation decision ===
    # Web search is needed if any metric is below threshold
    needs_web_search = (
        relevancy < 0.5 or          # Poor relevancy to query
        context_quality < 0.4 or    # Limited depth
        negative_count > positive_count or  # Speculative language
        context_length < 50         # Too short response
    )
    
    explanation = (
        f"Relevancy: {relevancy:.2f} (word overlap coverage), "
        f"Faithfulness: {faithfulness:.2f} (evidence indicators), "
        f"Context Quality: {context_quality:.2f} (depth/structure). "
        f"Web search {'recommended' if needs_web_search else 'not needed'}."
    )
    
    return GradingResult(
        relevancy=round(relevancy, 2),
        faithfulness=round(faithfulness, 2),
        context_quality=round(context_quality, 2),
        needs_web_search=needs_web_search,
        explanation=explanation
    )




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
# RAG TOOLS - Document Retrieval
# ============================================================================

async def retriever_tool(ctx: RunContext[Deps], question: str, mode: str = "hybrid", k: int = 5) -> List[str]:
    """
    Custom retrieval tool for accessing local documents via LightRAG.
    
    This function queries the LightRAG service at http://localhost:9621/query
    and returns the top-k most similar document chunks.
    
    Args:
        ctx: RunContext with Deps containing question and context
        question: The query to search for
        mode: LightRAG search mode (local, global, hybrid, etc.)
        k: Number of top results to return
    
    Returns:
        List of document content strings from the most relevant sources
        
    Example:
        docs = await retriever_tool(ctx, "What is an API?", mode="hybrid", k=3)
        # Returns: ["API is...", "APIs allow...", "REST API..."]
    """
    # Query LightRAG for documents
    result = await query_lightrag(question, mode=mode, top_k=k)
    
    if result.get("success"):
        # Return the response content as a list (for compatibility with similarity_search pattern)
        response_text = result.get("response", "")
        sources = result.get("sources", [])
        
        # Store sources in context for later use
        if ctx.deps:
            ctx.deps.context = response_text
        
        # Return as list of content chunks
        # In this case, we return the full response as one chunk, but you could split it
        return [response_text] if response_text else []
    else:
        return []


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
        async with httpx.AsyncClient(timeout=60.0) as client:
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
    except httpx.ReadTimeout:
        print(f"⏰ LightRAG Timeout: Query took longer than 60 seconds")
        return {
            "response": "LightRAG query timed out - service may be busy or query too complex",
            "sources": [],
            "success": False
        }
    except httpx.HTTPStatusError as e:
        print(f"❌ LightRAG HTTP Error: {e.response.status_code} - {e.response.text}")
        return {
            "response": f"LightRAG service error: {e.response.status_code}",
            "sources": [],
            "success": False
        }
    except Exception as e:
        print(f"❌ LightRAG Connection Error: {e}")
        return {
            "response": f"Could not connect to LightRAG service: {str(e)}",
            "sources": [],
            "success": False
        }


# ============================================================================
# RAG TOOLS - Web Search Integration
# ============================================================================

async def websearch_tool(question: str, max_results: int = 5) -> str:
    """
    Web search tool implemented using Tavily API.
    
    This function performs a web search and returns a QnA-style answer
    directly from the Tavily service.
    
    Args:
        question: The search query
        max_results: Maximum number of results to retrieve (default: 5)
    
    Returns:
        String containing the answer from web search, or error message
        
    Example:
        answer = await websearch_tool("Latest AI news 2025")
        # Returns: "Recent developments in AI include..."
    """
    result = await search_web_tavily(question, max_results=max_results)
    
    if result.get("success"):
        # Return the answer if available, otherwise concatenate result snippets
        if "answer" in result and result["answer"]:
            return result["answer"]
        else:
            # Combine results into a coherent answer
            results = result.get("results", [])
            if results:
                answer_parts = [r.get("content", "") for r in results[:3]]
                return "\n\n".join(filter(None, answer_parts))
            else:
                return "No web search results found."
    else:
        return f"Web search failed: {result.get('error', 'Unknown error')}"


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
    # States & Dependencies
    "ProverbsState",
    "RAGState",
    "Deps",
    "RAGDeps",  # Backward compatibility
    # Assessment & Grading
    "AssessmentResult",
    "GradingResult",
    "assess_local_info",
    "grade_response_quality",
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
    # RAG tools (new design pattern)
    "retriever_tool",      # Custom retrieval tool for local documents
    "websearch_tool",      # Web search using Tavily
    # RAG tools (original)
    "query_lightrag",
    "search_web_tavily",
]
