"""
RAG Agentic AI System with Self-Grading and Web Search Integration
Uses OpenAI API, local LightRAG, and Tavily web search
"""

from typing import Optional, List
import os
import json
import httpx
import asyncio
from pydantic import BaseModel, Field
from openai import OpenAI

# Optional: Configure logging if token is available
try:
    import logfire
    logfire_token = os.getenv('LOGFIRE_TOKEN')
    if logfire_token:
        logfire.configure(token=logfire_token)
except Exception:
    pass

# ============================================================================
# 1. TYPE DEFINITIONS
# ============================================================================

class GradingScore(BaseModel):
    """Response grading scores"""
    relevancy: float = Field(..., ge=0, le=1, description="Relevancy score 0-1")
    faithfulness: float = Field(..., ge=0, le=1, description="Faithfulness to context 0-1")
    context_quality: float = Field(..., ge=0, le=1, description="Context quality 0-1")
    needs_web_search: bool = Field(..., description="Whether web search is needed")
    reasoning: str = Field(..., description="Explanation of grading")


class RAGResponse(BaseModel):
    """Final RAG response"""
    answer: str = Field(..., description="The generated answer")
    sources: List[str] = Field(default_factory=list, description="Sources used")
    grading: GradingScore = Field(..., description="Self-grading scores")
    used_web_search: bool = Field(default=False, description="Whether web search was used")


# ============================================================================
# 2. LIGHT RAG INTEGRATION
# ============================================================================

class LightRAGClient:
    """Client for querying local LightRAG instance"""
    
    def __init__(self, host: str = "http://localhost:9621"):
        self.host = host
        self.query_endpoint = f"{host}/query"
    
    async def query(
        self, 
        query: str, 
        mode: str = "hybrid",
        top_k: int = 5
    ) -> dict:
        """
        Query local LightRAG instance
        
        Args:
            query: The natural language query
            mode: "local", "hybrid", or "global" search mode
            top_k: Number of top results to return
        
        Returns:
            Response containing answer and sources
        """
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
                response = await client.post(
                    self.query_endpoint,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                if not result or "response" not in result or not result["response"]:
                    return {
                        "response": "No relevant documentation found in local RAG",
                        "sources": [],
                        "success": False
                    }
                
                return {
                    "response": result.get("response", ""),
                    "sources": result.get("sources", []),
                    "success": True
                }
        except Exception as e:
            error_msg = str(e) if str(e) else "Connection error or invalid response from LightRAG"
            print(f"❌ LightRAG Query Error: {error_msg}")
            return {
                "response": f"Error querying local RAG: {error_msg}",
                "sources": [],
                "success": False
            }


# ============================================================================
# 3. WEB SEARCH INTEGRATION (TAVILY)
# ============================================================================

class TavilySearchClient:
    """Client for Tavily web search"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            print("⚠️ TAVILY_API_KEY not set. Web search will be unavailable.")
    
    async def search(self, query: str, max_results: int = 5) -> dict:
        """
        Search the web using Tavily API
        
        Args:
            query: Search query
            max_results: Maximum number of results
        
        Returns:
            Search results
        """
        if not self.api_key:
            return {
                "results": [],
                "success": False,
                "error": "Tavily API key not configured"
            }
        
        payload = {
            "api_key": self.api_key,
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


# ============================================================================
# 4. GRADING SYSTEM
# ============================================================================

async def grade_response(
    question: str,
    context: str,
    answer: str,
    model: str = "gpt-4.1-mini"
) -> GradingScore:
    """
    Grade the agent's response using OpenAI
    
    Evaluates:
    - Relevancy: Does the answer address the question?
    - Faithfulness: Is the answer grounded in the provided context?
    - Context Quality: Is the context sufficient?
    """
    
    client = OpenAI()
    
    grading_prompt = f"""
    You are an expert evaluator. Grade the following response on three criteria (0-1 scale):
    
    QUESTION: {question}
    
    CONTEXT PROVIDED: {context}
    
    ANSWER: {answer}
    
    Evaluate and respond with JSON:
    {{
        "relevancy": <0-1 score: How well does the answer address the question?>,
        "faithfulness": <0-1 score: How faithful is the answer to the provided context?>,
        "context_quality": <0-1 score: Is the context sufficient and appropriate?>,
        "needs_web_search": <boolean: Should web search be used for better coverage?>,
        "reasoning": "<Brief explanation of scores>"
    }}
    
    Return ONLY the JSON object, no other text.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=500,
            messages=[
                {"role": "user", "content": grading_prompt}
            ]
        )
        
        response_text = response.choices[0].message.content
        grading_data = json.loads(response_text)
        
        return GradingScore(**grading_data)
    
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse grading response: {e}")
        return GradingScore(
            relevancy=0.5,
            faithfulness=0.5,
            context_quality=0.5,
            needs_web_search=True,
            reasoning="Grading system error, defaulting to web search"
        )


# ============================================================================
# 5. AGENT IMPLEMENTATION
# ============================================================================

class RAGAgent:
    """Simple RAG agent using OpenAI API"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None
        self.system_prompt = """
    You are an expert RAG assistant. Your role is to:
    
    1. Answer user questions using the provided context from local documentation
    2. Provide accurate, well-structured answers grounded in the sources
    3. Include relevant examples and explanations
    4. Cite your sources appropriately
    
    When context is insufficient, the system will automatically augment with web search results.
    Always prioritize local documentation, but supplement with web results when needed.
    """
    
    @property
    def client(self):
        """Lazy load OpenAI client"""
        if self._client is None:
            self._client = OpenAI()
        return self._client
    
    async def query(self, prompt: str) -> str:
        """Query the OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


# Initialize clients
light_rag = LightRAGClient(host=os.getenv("LIGHT_RAG_HOST", "http://localhost:9621"))
tavily = TavilySearchClient()
rag_agent = RAGAgent(model=os.getenv("RAG_AGENT_MODEL", "gpt-4o-mini"))


# ============================================================================
# 6. HELPER FUNCTIONS
# ============================================================================

async def retrieve_from_local_rag(query: str) -> str:
    """
    Retrieve information from local LightRAG instance
    
    Args:
        query: The query to search for
    
    Returns:
        Retrieved context from documentation
    """
    print(f"\n🔍 Querying Local RAG: {query}")
    
    result = await light_rag.query(
        query=query,
        mode="hybrid",
        top_k=5
    )
    
    if result["success"]:
        print(f"✅ Retrieved from Local RAG (sources: {len(result['sources'])})")
        return result["response"]
    else:
        print(f"⚠️ Local RAG query failed: {result['response']}")
        return result["response"]


async def search_web(query: str) -> str:
    """
    Search the web using Tavily when local context is insufficient
    
    Args:
        query: The query to search for
    
    Returns:
        Web search results
    """
    print(f"\n🌐 Searching Web: {query}")
    
    result = await tavily.search(query=query, max_results=5)
    
    if result["success"]:
        formatted_results = "Web Search Results:\n"
        for i, item in enumerate(result["results"], 1):
            formatted_results += f"\n{i}. {item.get('title', 'No title')}\n"
            formatted_results += f"   {item.get('content', 'No content')}\n"
            formatted_results += f"   Source: {item.get('url', 'No URL')}\n"
        
        if result.get("answer"):
            formatted_results += f"\nDirect Answer: {result['answer']}\n"
        
        print(f"✅ Retrieved from Web (sources: {len(result['results'])})")
        return formatted_results
    else:
        print(f"⚠️ Web search failed: {result.get('error', 'Unknown error')}")
        return f"Web search failed: {result.get('error', 'Unknown error')}"


# ============================================================================
# 7. MAIN QUERY FUNCTION
# ============================================================================

async def query_rag_system(query: str) -> RAGResponse:
    """
    Query the RAG agentic system end-to-end
    
    Process:
    1. Query local LightRAG
    2. Generate initial answer
    3. Self-grade the response
    4. If grade indicates, augment with web search
    5. Generate final answer
    6. Return with grading scores
    """
    
    print(f"\n{'='*70}")
    print(f"🚀 RAG AGENTIC QUERY: {query}")
    print(f"{'='*70}\n")
    
    # Step 1: Query local RAG
    print("📚 Step 1: Querying Local RAG...")
    local_context = await light_rag.query(query=query, mode="hybrid", top_k=5)
    
    # Step 2: Generate initial answer
    print("\n🤖 Step 2: Generating Initial Answer...")
    prompt = f"""
User Question: {query}

Available Context:
{local_context.get('response', 'No context available')}

Please provide a comprehensive answer based on the context above.
If the context is insufficient, indicate that.
    """
    
    initial_answer = await rag_agent.query(prompt)
    print(f"Initial Answer: {initial_answer[:200]}...")
    
    # Step 3: Grade the response
    print("\n⭐ Step 3: Self-Grading...")
    grading = await grade_response(
        question=query,
        context=local_context.get('response', ''),
        answer=initial_answer
    )
    
    print(f"  Relevancy: {grading.relevancy:.2f}")
    print(f"  Faithfulness: {grading.faithfulness:.2f}")
    print(f"  Context Quality: {grading.context_quality:.2f}")
    print(f"  Needs Web Search: {grading.needs_web_search}")
    
    used_web_search = False
    web_results = {}
    
    # Step 4: Conditionally augment with web search
    if grading.needs_web_search:
        print("\n🌐 Step 4: Augmenting with Web Search...")
        web_results = await tavily.search(query=query, max_results=5)
        
        if web_results["success"]:
            used_web_search = True
            
            # Generate augmented answer
            augmented_prompt = f"""
User Question: {query}

Local Documentation Context:
{local_context.get('response', 'No local context')}

Web Search Results:
{json.dumps(web_results['results'][:3], indent=2)}

Please provide an improved answer that combines both local and web sources.
Cite which source each piece of information comes from.
            """
            
            final_answer = await rag_agent.query(augmented_prompt)
            print("✅ Generated augmented answer from local + web sources")
        else:
            final_answer = initial_answer
            print("⚠️ Web search failed, using initial answer")
    else:
        print("\n✅ Local context sufficient, no web search needed")
        final_answer = initial_answer
    
    # Prepare response
    sources = local_context.get("sources", [])
    if used_web_search and web_results.get("success"):
        sources.extend([r["url"] for r in web_results.get("results", [])[:3]])
    
    response = RAGResponse(
        answer=final_answer,
        sources=sources,
        grading=grading,
        used_web_search=used_web_search
    )
    
    print(f"\n{'='*70}")
    print("✅ QUERY COMPLETE")
    print(f"{'='*70}\n")
    
    return response


# ============================================================================
# 8. CLI INTERFACE
# ============================================================================

async def main():
    """Interactive CLI for RAG agentic system"""
    
    print("\n" + "="*70)
    print("🚀 RAG AGENTIC AI SYSTEM")
    print("="*70)
    print("Local RAG: http://localhost:9621")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            query = input("📝 Enter your question: ").strip()
            
            if query.lower() == "exit":
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            response = await query_rag_system(query)
            
            print(f"\n{'='*70}")
            print("📋 FINAL ANSWER:")
            print(f"{'='*70}")
            print(response.answer)
            
            print(f"\n📚 Sources Used:")
            for i, source in enumerate(response.sources, 1):
                print(f"  {i}. {source}")
            
            print(f"\n⭐ Grading Scores:")
            print(f"  Relevancy: {response.grading.relevancy:.2f}/1.0")
            print(f"  Faithfulness: {response.grading.faithfulness:.2f}/1.0")
            print(f"  Context Quality: {response.grading.context_quality:.2f}/1.0")
            print(f"  Web Search Used: {'Yes' if response.used_web_search else 'No'}")
            print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())