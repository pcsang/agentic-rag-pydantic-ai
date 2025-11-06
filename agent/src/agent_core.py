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

class LocalRAGAssessment(BaseModel):
    """Assessment of local RAG information quality"""
    has_info: bool = Field(..., description="Whether local RAG has any relevant information")
    info_quality: str = Field(..., description="Quality level: 'none', 'partial', 'sufficient'")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in the information 0-1")
    reasoning: str = Field(..., description="Explanation of the assessment")


class RAGDecision(BaseModel):
    """Decision on which RAG strategy to use"""
    strategy: str = Field(..., description="Strategy: 'local_only', 'web_only', 'generate_grade_augment'")
    reasoning: str = Field(..., description="Why this strategy was chosen")


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
    grading: Optional[GradingScore] = Field(None, description="Self-grading scores if applicable")
    used_web_search: bool = Field(default=False, description="Whether web search was used")
    strategy_used: str = Field(..., description="Strategy that was applied")
    local_assessment: LocalRAGAssessment = Field(..., description="Assessment of local information")


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
        top_k: int = 5  # Changed from 50 to 5 for better performance
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
                
                print(f"🔍 LightRAG raw response: {result}")  # Debug output
                
                # Check if we have a valid response structure
                if not result or "response" not in result:
                    return {
                        "response": "No relevant documentation found in local RAG",
                        "sources": [],
                        "success": False
                    }
                
                # Map LightRAG's 'references' to our 'sources' format
                sources = []
                if "references" in result and result["references"]:
                    sources = [ref.get("file_path", ref.get("reference_id", "Unknown source")) 
                             for ref in result["references"]]
                
                return {
                    "response": result.get("response", ""),
                    "sources": sources,
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
You are an expert evaluator. Grade the following response on three criteria (0-1 scale).

QUESTION: {question}

CONTEXT PROVIDED: {context}

ANSWER: {answer}

Evaluate and respond with ONLY a valid JSON object in this exact format:
{{
    "relevancy": 0.8,
    "faithfulness": 0.7,
    "context_quality": 0.6,
    "needs_web_search": false,
    "reasoning": "Brief explanation of scores"
}}

Requirements:
- relevancy: How well does the answer address the question? (0.0 to 1.0)
- faithfulness: How faithful is the answer to the provided context? (0.0 to 1.0) 
- context_quality: Is the context sufficient and appropriate? (0.0 to 1.0)
- needs_web_search: Should web search be used for better coverage? (true/false)
- reasoning: Brief explanation (one sentence)

Return ONLY the JSON object, no other text or formatting.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=500,
            messages=[
                {"role": "user", "content": grading_prompt}
            ]
        )
        
        response_text = response.choices[0].message.content
        print(f"🔍 Raw grading response: {response_text}")
        
        if not response_text or not response_text.strip():
            raise ValueError("Empty response from grading model")
        
        # Clean and extract JSON
        response_text = response_text.strip()
        
        # Try to extract JSON if wrapped in markdown or other text
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()
        
        # Find JSON object if mixed with other text
        if not response_text.startswith("{"):
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                response_text = response_text[json_start:json_end]
        
        grading_data = json.loads(response_text)
        return GradingScore(**grading_data)
    
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse grading response: {e}")
        print(f"   Raw response: {response_text[:200] if 'response_text' in locals() else 'No response'}")
        return GradingScore(
            relevancy=0.5,
            faithfulness=0.5,
            context_quality=0.5,
            needs_web_search=True,
            reasoning="JSON parsing error in grading system, defaulting to web search"
        )
    except Exception as e:
        print(f"❌ Grading system error: {e}")
        return GradingScore(
            relevancy=0.5,
            faithfulness=0.5,
            context_quality=0.5,
            needs_web_search=True,
            reasoning=f"Grading system error: {str(e)}, defaulting to web search"
        )


# ============================================================================
# 5. AGENT IMPLEMENTATION
# ============================================================================

class RAGAgent:
    """Simple RAG agent using OpenAI API"""
    
    def __init__(self, model: str = "gpt-5"):
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
            max_completion_tokens=2000,
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

async def assess_local_information(query: str, local_response: str, local_sources: List[str]) -> LocalRAGAssessment:
    """
    Assess the quality and sufficiency of local RAG information
    
    Args:
        query: The original query
        local_response: Response from local RAG
        local_sources: Sources returned by local RAG
    
    Returns:
        Assessment of local information quality
    """
    # Heuristic-based assessment
    has_meaningful_response = bool(local_response and 
                                 len(local_response.strip()) > 20 and
                                 "no relevant" not in local_response.lower() and
                                 "not enough information" not in local_response.lower())
    
    has_sources = bool(local_sources and len(local_sources) > 0)
    
    if not has_meaningful_response and not has_sources:
        return LocalRAGAssessment(
            has_info=False,
            info_quality="none",
            confidence_score=0.0,
            reasoning="No meaningful response or sources from local RAG"
        )
    
    # Check for partial/insufficient indicators
    insufficient_indicators = [
        "insufficient", "unclear", "more details", "please provide",
        "not specific", "limited information", "partial"
    ]
    
    is_partial = any(indicator in local_response.lower() for indicator in insufficient_indicators)
    response_length = len(local_response.strip())
    
    if is_partial or response_length < 100:
        return LocalRAGAssessment(
            has_info=True,
            info_quality="partial",
            confidence_score=0.4,
            reasoning=f"Local RAG has some info but seems incomplete (length: {response_length}, sources: {len(local_sources)})"
        )
    
    # Sufficient information
    return LocalRAGAssessment(
        has_info=True,
        info_quality="sufficient",
        confidence_score=0.8,
        reasoning=f"Local RAG provides comprehensive information (length: {response_length}, sources: {len(local_sources)})"
    )


async def decide_rag_strategy(assessment: LocalRAGAssessment) -> RAGDecision:
    """
    Decide which RAG strategy to use based on local information assessment
    
    Args:
        assessment: Assessment of local information
    
    Returns:
        Decision on RAG strategy
    """
    if assessment.info_quality == "none":
        return RAGDecision(
            strategy="web_only",
            reasoning="No local information available, skip to web search"
        )
    elif assessment.info_quality == "sufficient" and assessment.confidence_score >= 0.7:
        return RAGDecision(
            strategy="local_only",
            reasoning="Local information is sufficient and confident"
        )
    else:
        return RAGDecision(
            strategy="generate_grade_augment",
            reasoning="Local information is partial/unclear, need to generate, grade, and potentially augment"
        )


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
    Query the RAG agentic system with three-case decision logic:
    1. If local has sufficient info → use local only
    2. If local has NO info → skip to web search
    3. If local has some info but unclear → generate + grade + augment
    """
    
    print(f"\n{'='*70}")
    print(f"🚀 RAG AGENTIC QUERY: {query}")
    print(f"{'='*70}\n")
    
    # Step 1: Query local RAG and assess information quality
    print("📚 Step 1: Querying Local RAG...")
    local_context = await light_rag.query(query=query, mode="hybrid", top_k=5)
    
    local_response = local_context.get("response", "") if local_context.get("success") else ""
    local_sources = local_context.get("sources", [])
    
    print("🔍 Step 2: Assessing Local Information Quality...")
    assessment = await assess_local_information(query, local_response, local_sources)
    print(f"   Assessment: {assessment.info_quality} (confidence: {assessment.confidence_score:.2f})")
    print(f"   Reasoning: {assessment.reasoning}")
    
    # Step 3: Decide on RAG strategy
    decision = await decide_rag_strategy(assessment)
    print(f"📋 Step 3: Strategy Decision: {decision.strategy}")
    print(f"   Reasoning: {decision.reasoning}")
    
    # Initialize variables
    final_answer = ""
    used_web_search = False
    grading = None
    sources = local_sources.copy()
    
    # Step 4: Execute the chosen strategy
    if decision.strategy == "local_only":
        # Case 1: Local has sufficient info → use local only
        print("\n✅ Case 1: Using local information only")
        prompt = f"""
User Question: {query}

Local Documentation Context:
{local_response}

Please provide a comprehensive answer based on the local context above.
Since the local information is sufficient, focus on providing a complete response.
        """
        final_answer = await rag_agent.query(prompt)
        
    elif decision.strategy == "web_only":
        # Case 2: Local has NO info → skip to web search
        print("\n🌐 Case 2: No local info - using web search only")
        web_results = await tavily.search(query=query, max_results=5)
        
        if web_results.get("success"):
            used_web_search = True
            sources = [r["url"] for r in web_results.get("results", [])[:5]]
            
            prompt = f"""
User Question: {query}

Web Search Results:
{json.dumps(web_results.get('results', [])[:5], indent=2)}

Please provide a comprehensive answer based only on the web search results above.
Cite your sources appropriately.
            """
            final_answer = await rag_agent.query(prompt)
        else:
            final_answer = "I'm unable to find relevant information locally or on the web. Please try rephrasing your question or check back later."
            
    else:
        # Case 3: Local has some info but unclear → generate + grade + augment
        print("\n🤖 Case 3: Generate + Grade + Augment")
        
        # Generate initial answer from local context
        print("   3a. Generating initial answer from local context...")
        initial_prompt = f"""
User Question: {query}

Available Context:
{local_response}

Please provide an answer based on the context above. If the context is insufficient, clearly indicate what's missing.
        """
        initial_answer = await rag_agent.query(initial_prompt)
        print(f"   Initial Answer: {initial_answer[:150]}...")
        
        # Grade the initial response
        print("   3b. Grading the initial response...")
        grading = await grade_response(
            question=query,
            context=local_response,
            answer=initial_answer
        )
        
        print(f"      Relevancy: {grading.relevancy:.2f}")
        print(f"      Faithfulness: {grading.faithfulness:.2f}")
        print(f"      Context Quality: {grading.context_quality:.2f}")
        print(f"      Needs Web Search: {grading.needs_web_search}")
        
        # Decide whether to augment with web search
        threshold = 0.6
        needs_augmentation = (grading.needs_web_search or 
                            grading.relevancy < threshold or 
                            grading.context_quality < threshold)
        
        if needs_augmentation:
            print("   3c. Augmenting with web search...")
            web_results = await tavily.search(query=query, max_results=5)
            
            if web_results.get("success"):
                used_web_search = True
                web_sources = [r["url"] for r in web_results.get("results", [])[:5]]
                sources.extend(web_sources)
                
                augmented_prompt = f"""
User Question: {query}

Local Documentation Context:
{local_response}

Initial Answer from Local Context:
{initial_answer}

Web Search Results for Augmentation:
{json.dumps(web_results.get('results', [])[:5], indent=2)}

Please provide an improved answer that combines and synthesizes information from both local and web sources.
- Clearly indicate what information comes from local vs web sources
- Resolve any conflicts between sources
- Provide a comprehensive and well-cited response
                """
                final_answer = await rag_agent.query(augmented_prompt)
                print("   ✅ Generated augmented answer from local + web sources")
            else:
                final_answer = initial_answer
                print("   ⚠️ Web search failed, using initial local answer")
        else:
            print("   ✅ Initial answer quality sufficient, no augmentation needed")
            final_answer = initial_answer
    
    # Prepare final response
    response = RAGResponse(
        answer=final_answer,
        sources=sources,
        grading=grading,
        used_web_search=used_web_search,
        strategy_used=decision.strategy,
        local_assessment=assessment
    )
    
    print(f"\n{'='*70}")
    print("✅ QUERY COMPLETE")
    print(f"Strategy Used: {decision.strategy}")
    print(f"Web Search Used: {'Yes' if used_web_search else 'No'}")
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
            
            print(f"\n📊 System Information:")
            print(f"  Strategy Used: {response.strategy_used}")
            print(f"  Local Assessment: {response.local_assessment.info_quality} (confidence: {response.local_assessment.confidence_score:.2f})")
            print(f"  Web Search Used: {'Yes' if response.used_web_search else 'No'}")
            
            if response.grading:
                print(f"\n⭐ Grading Scores:")
                print(f"  Relevancy: {response.grading.relevancy:.2f}/1.0")
                print(f"  Faithfulness: {response.grading.faithfulness:.2f}/1.0")
                print(f"  Context Quality: {response.grading.context_quality:.2f}/1.0")
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