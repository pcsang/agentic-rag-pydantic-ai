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

import logging
from datetime import datetime

# Setup logging for the main server
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/multi_agent_server.log'),
        logging.StreamHandler()
    ]
)

server_logger = logging.getLogger('MULTI_AGENT_SERVER')

print("\n" + "=" * 80)
print("🚀 MULTI-AGENT SERVER STARTING")
print("=" * 80)

server_logger.info("Multi-Agent Server startup initiated")
server_logger.info(f"Startup time: {datetime.now().isoformat()}")

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
    from rag_agent import handle_query as rag_handle_query
    
    # Create RAG endpoint
    @app.post("/rag")
    async def rag_endpoint(request: dict):
        """RAG agent endpoint using simplified OpenAI-based agent"""
        try:
            question = request.get("question") or request.get("query") or request.get("message", "")
            if not question:
                return {"error": "No question provided"}
            
            server_logger.info(f"📝 RAG Query: {question}")
            result = await rag_handle_query(question)
            server_logger.info(f"✅ RAG Response: {len(result.get('answer', ''))} chars, {len(result.get('sources', []))} sources")
            return result
        except Exception as e:
            server_logger.error(f"❌ RAG Error: {e}", exc_info=True)
            return {"error": str(e), "answer": f"Error: {str(e)}", "sources": []}
    
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


# Add request logging middleware with detailed debug info
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests with detailed debug information"""
    import time
    from urllib.parse import parse_qs
    
    request_start = time.time()
    request_id = id(request)  # Unique ID for tracking request lifecycle
    
    print("\n" + "=" * 80)
    print(f"🌐 NEW REQUEST RECEIVED [ID: {request_id}]")
    print("=" * 80)
    print(f"📨 REQUEST DETAILS:")
    print(f"   ├─ Method: {request.method}")
    print(f"   ├─ URL: {request.url}")
    print(f"   ├─ Path: {request.url.path}")
    print(f"   ├─ Query: {request.url.query}")
    print(f"   ├─ Client: {request.client.host if request.client else 'Unknown'}:{request.client.port if request.client else 'Unknown'}")
    print(f"   ├─ Headers: Content-Type={request.headers.get('content-type', 'not set')}")
    print(f"   └─ Timestamp: {datetime.now().isoformat()}")
    
    # Log headers for debugging CopilotKit routing
    print(f"\n📋 ALL HEADERS:")
    for header_name, header_value in request.headers.items():
        if header_name.lower() not in ['authorization']:  # Don't log auth tokens
            print(f"   {header_name}: {header_value}")
    
    # Determine which agent should handle this
    endpoint_target = "UNKNOWN"
    if "/rag" in request.url.path:
        endpoint_target = "RAG_AGENT (/rag)"
        print(f"\n🤖 ENDPOINT TARGET: {endpoint_target}")
        server_logger.info(f"[{request_id}] Request routed to RAG_AGENT: {request.url.path}")
    elif "/jira" in request.url.path:
        endpoint_target = "JIRA_AGENT (/jira)"
        print(f"\n🎫 ENDPOINT TARGET: {endpoint_target}")
        server_logger.info(f"[{request_id}] Request routed to JIRA_AGENT: {request.url.path}")
    else:
        endpoint_target = "ROOT/HEALTH"
        print(f"\n📍 ENDPOINT TARGET: {endpoint_target}")
        server_logger.info(f"[{request_id}] Request routed to ROOT/HEALTH: {request.url.path}")
    
    print("=" * 80 + "\n")
    
    # Call the endpoint
    try:
        response = await call_next(request)
        duration = time.time() - request_start
        
        # Log response
        print("=" * 80)
        print(f"✅ RESPONSE SENT [ID: {request_id}]")
        print("=" * 80)
        print(f"📤 RESPONSE DETAILS:")
        print(f"   ├─ Status Code: {response.status_code}")
        print(f"   ├─ Content-Type: {response.headers.get('content-type', 'not set')}")
        print(f"   ├─ Duration: {duration:.3f}s")
        print(f"   ├─ Endpoint: {endpoint_target}")
        print(f"   └─ Timestamp: {datetime.now().isoformat()}")
        print("=" * 80 + "\n")
        
        server_logger.info(f"[{request_id}] Response sent: {response.status_code} in {duration:.3f}s to {endpoint_target}")
        return response
        
    except Exception as e:
        duration = time.time() - request_start
        print("=" * 80)
        print(f"❌ REQUEST ERROR [ID: {request_id}]")
        print("=" * 80)
        print(f"⚠️ ERROR DETAILS:")
        print(f"   ├─ Error Type: {type(e).__name__}")
        print(f"   ├─ Error Message: {str(e)}")
        print(f"   ├─ Endpoint: {endpoint_target}")
        print(f"   ├─ Duration: {duration:.3f}s")
        print(f"   └─ Timestamp: {datetime.now().isoformat()}")
        print("=" * 80 + "\n")
        
        server_logger.error(f"[{request_id}] Error in {endpoint_target}: {str(e)}", exc_info=True)
        raise


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
