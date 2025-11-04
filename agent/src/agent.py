"""Compatibility shim re-exporting the refactored agent core.

This module used to contain the full agent implementation. The implementation
was split into three modules:
 - tools.py      (state and general tool functions)
 - mcp_jira.py   (MCP / Jira helpers)
 - agent_core.py (agent creation and wiring)

Other code can continue to import from `agent` (as before) and will get the
Agent instance and commonly used symbols re-exported from the new core.
"""

from agent_core import agent, ProverbsState, StateDeps, run_web_search

__all__ = ["agent", "ProverbsState", "StateDeps", "run_web_search"]



