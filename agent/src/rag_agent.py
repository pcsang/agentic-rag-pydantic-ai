"""
RAG Agent for Local Documentation Search
"""

from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps
from tools import RAGState, query_lightrag, search_web_tavily

print("🤖 RAG AGENT INITIALIZING")

rag_agent = Agent(
    name="rag_agent",
    model="openai:gpt-4o-mini",
    system_prompt="You are a documentation assistant. Use search_documents tool for queries.",
    retries=1
)

print("✅ RAG Agent created")

@rag_agent.tool
async def search_documents(
    ctx: RunContext[StateDeps[RAGState]], 
    query: str,
    mode: str = "hybrid"
) -> str:
    """Search local docs via LightRAG"""
    print(f"\n🔍 SEARCH: {query}")
    
    ctx.deps.state.query_count += 1
    ctx.deps.state.last_query = query
    
    # Call LightRAG
    print("📡 Calling http://localhost:9621/query")
    result = await query_lightrag(query, mode=mode)
    
    print(f"✅ Success: {result.get('success')}")
    print(f"   Response: {len(result.get('response', ''))} chars")
    print(f"   Sources: {len(result.get('sources', []))}")
    
    if result.get("success") and result.get("response"):
        sources = result.get("sources", [])
        response_text = result.get("response", "")
        
        output = f"## 📚 Answer\n\n{response_text}\n\n"
        
        if sources:
            output += "### 📖 Sources\n"
            for i, src in enumerate(sources, 1):
                filename = src.split('/')[-1] if '/' in src else src
                output += f"- {filename}\n"
        
        ctx.deps.state.last_sources = sources
        return output
    
    else:
        # Try web search
        web_result = await search_web_tavily(query, max_results=3)
        
        if web_result.get("success"):
            output = f"## 🌐 Web Results\n\n{web_result.get('answer', '')}\n\n"
            return output
        
        return "❌ No information found"

__all__ = ["rag_agent", "RAGState"]
