"""
Multi-Agent Server - RAG Agent + Jira Agent
Serves both agents with automatic routing based on query content
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

print("\n" + "=" * 80)
print("🚀 MULTI-AGENT SERVER STARTING")
print("=" * 80)

# Check if API key is loaded
if os.getenv("OPENAI_API_KEY"):
    api_key_preview = os.getenv("OPENAI_API_KEY")[:10] + "..." if os.getenv("OPENAI_API_KEY") else "NOT SET"
    print(f"✅ OPENAI_API_KEY loaded: {api_key_preview}")
else:
    print("⚠️ WARNING: OPENAI_API_KEY not set!")
    print("   Agents will fail to initialize without a valid API key.")

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent RAG & Jira System",
    description="Intelligent routing between RAG and Jira agents",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✅ FastAPI app created with CORS middleware")

available_agents = []

# Try to load RAG agent
try:
    print("\n" + "=" * 80)
    print("🔄 Loading RAG Agent...")
    print("=" * 80)
    from rag_agent import rag_agent, RAGState
    from pydantic_ai.ag_ui import StateDeps as RAGStateDeps
    
    rag_app = rag_agent.to_ag_ui(deps=RAGStateDeps(RAGState()))
    app.mount("/rag", rag_app)
    available_agents.append("rag_agent")
    print("✅ RAG Agent loaded and mounted at /rag endpoint")
except Exception as e:
    print(f"⚠️ RAG Agent failed to load: {e}")
    import traceback
    traceback.print_exc()

# Try to load Jira agent  
try:
    print("\n" + "=" * 80)
    print("🔄 Loading Jira Agent...")
    print("=" * 80)
    from jira_agent import jira_agent, ProverbsState
    from pydantic_ai.ag_ui import StateDeps as JiraStateDeps
    
    jira_app = jira_agent.to_ag_ui(deps=JiraStateDeps(ProverbsState()))
    app.mount("/jira", jira_app)
    available_agents.append("jira_agent")
    print("✅ Jira Agent loaded and mounted at /jira endpoint")
except Exception as e:
    print(f"⚠️ Jira Agent failed to load: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print(f"📊 SERVER INITIALIZATION COMPLETE")
print(f"✅ Loaded {len(available_agents)} agent(s): {', '.join(available_agents)}")
print("=" * 80 + "\n")


# Add request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests to see routing"""
    print("\n" + "🌐" * 40)
    print(f"📨 INCOMING REQUEST:")
    print(f"   Method: {request.method}")
    print(f"   URL: {request.url}")
    print(f"   Path: {request.url.path}")
    print(f"   Client: {request.client.host if request.client else 'Unknown'}")
    
    # Determine which agent should handle this
    if "/rag" in request.url.path:
        print(f"   🤖 Routing to: RAG AGENT")
    elif "/jira" in request.url.path:
        print(f"   🎫 Routing to: JIRA AGENT")
    else:
        print(f"   📍 Routing to: Root/Health endpoint")
    
    print("🌐" * 40 + "\n")
    
    response = await call_next(request)
    return response


@app.get("/")
async def root():
    """Root endpoint showing available agents"""
    return {
        "message": "Multi-Agent RAG & Jira System",
        "version": "2.0.0",
        "available_agents": available_agents,
        "agents": {
            "rag_agent": {
                "endpoint": "/rag" if "rag_agent" in available_agents else "unavailable",
                "description": "LightRAG + Tavily web search for documentation and technical queries"
            },
            "jira_agent": {
                "endpoint": "/jira" if "jira_agent" in available_agents else "unavailable",
                "description": "Jira ticket management and project tracking"
            }
        }
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "agents": available_agents
    }

if __name__ == "__main__":
    import uvicorn
    print(f"\n🚀 Starting Multi-Agent Server with {len(available_agents)} agents")
    print(f"📍 Available agents: {', '.join(available_agents)}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
