from textwrap import dedent
from pydantic_ai import Agent, WebSearchTool
try:
    from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
except Exception:
    duckduckgo_search_tool = None
from pydantic_ai.ag_ui import StateDeps
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

import tools
import mcp_jira
import sys


# Create the Agent instance (core wiring)
agent = Agent(
    model=OpenAIResponsesModel('gpt-4.1-mini'),
    builtin_tools=[WebSearchTool()],
    # register DuckDuckGo common tool as a searchable tool if available
    tools=[duckduckgo_search_tool()] if duckduckgo_search_tool is not None else [],
    model_settings=OpenAIResponsesModelSettings(openai_include_web_search_sources=True),
    deps_type=StateDeps[tools.ProverbsState],
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
    """).strip(),
)


# Register tools defined in tools.py by decorating them on the Agent instance.
# We explicitly register the expected tool names to avoid accidental exports.
_tool_names = [
    "get_proverbs",
    "add_proverbs",
    "set_proverbs",
    "get_weather",
    "emit_state",
]

for _name in _tool_names:
    fn = getattr(tools, _name)
    # agent.tool can be used as a decorator factory; calling it with the function
    # returns a wrapped function and registers the tool on the agent.
    wrapped = agent.tool(fn)
    # Expose the decorated function at module level (so imports continue to work).
    globals()[_name] = wrapped


# Register MCP / Jira functions similarly. We register all jira_* functions from mcp_jira.
_jira_names = [n for n in dir(mcp_jira) if n.startswith("jira_")]
for _name in _jira_names:
    fn = getattr(mcp_jira, _name)
    wrapped = agent.tool(fn)
    globals()[_name] = wrapped


def run_web_search(query: str, force_use_web_search: bool = True):
    """Run the agent on `query` and print the textual output plus any web-search link references.

    Kept here as a convenient helper that uses the core `agent` instance.
    """
    prompt = query
    if force_use_web_search:
        prompt = "Please use the built-in web search tool and cite source URLs. " + query

    result = agent.run_sync(prompt)
    print("\n=== Agent Output ===")
    print(result.output)

    print("\n=== Raw result.response (debug) ===")
    try:
        print(repr(result.response))
    except Exception:
        try:
            print(result.response)
        except Exception:
            pass

    bt_calls = getattr(result.response, 'builtin_tool_calls', None)
    if not bt_calls:
        print("\n(no builtin tool calls returned) — the model did not invoke any built-in tools for this prompt.")
        print("Try making the request more explicitly: 'Use the web search tool and list sources for:' + your query")
        return result

    print("\n=== Builtin tool calls / sources ===")
    for call_part, return_part in bt_calls:
        print("\n-- Tool call:", getattr(call_part, 'tool_name', call_part))
        try:
            rp = getattr(return_part, 'content', return_part)
        except Exception:
            rp = getattr(return_part, '__dict__', return_part)

        urls = []
        if isinstance(rp, dict):
            for k in ('search_results', 'results', 'content', 'links', 'sources', 'source_urls'):
                if k in rp and isinstance(rp[k], list):
                    for item in rp[k]:
                        if isinstance(item, dict) and 'url' in item:
                            urls.append(item['url'])
                        elif isinstance(item, str) and item.startswith('http'):
                            urls.append(item)

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


# Re-export commonly used symbols for backwards compatibility with `from agent import ...`
ProverbsState = tools.ProverbsState

__all__ = [
    "agent",
    "ProverbsState",
    "StateDeps",
    "run_web_search",
]
