"""Compatibility shim re-exporting the refactored agents.

This module provides backward compatibility for code that previously imported
from `agent`. The implementation was split into three modules:
 - tools.py      (state definitions and tool functions)
 - rag_agent.py  (RAG agent with LightRAG + Tavily search)
 - jira_agent.py (Jira project management agent)

Other code can continue to import from `agent` and will get the agents
and commonly used symbols re-exported.
"""

# Export RAG agent as default
from rag_agent import rag_agent as agent, RAGState
from jira_agent import jira_agent, ProverbsState
from pydantic_ai.ag_ui import StateDeps

# For backward compatibility, export RAG tools
from tools import search_web_tavily as run_web_search

__all__ = [
    "agent",           # Default RAG agent for backward compatibility
    "rag_agent",       # Explicit RAG agent
    "jira_agent",      # Jira agent
    "RAGState",        # RAG state
    "ProverbsState",   # Jira state
    "StateDeps",       # State dependencies
    "run_web_search"   # Web search tool
]



