"""
Agentic RAG System with Multi-Agent Architecture

Multi-Agent Components:
- Router Agent: Analyzes local RAG quality and decides routing strategy
- Retriever Agents: Local RAG and Web Search (Tavily)
- Generator Agent: Produces answers from context
- Grader Agent: Evaluates response quality with detailed rubric
- Supervisor Agent: Orchestrates workflow and manages iterations

Strategies:
1. LOCAL_ONLY: Sufficient local info, no temporal keywords
2. WEB_ONLY: No local info available
3. GENERATE_GRADE_AUGMENT: Partial info OR temporal keywords detected (with iterative refinement)
"""

from typing import Optional, List, Dict, Any
import os
import json
import httpx
import asyncio
from pydantic import BaseModel, Field
from openai import OpenAI
import logging
from datetime import datetime
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import logfire
    logfire_token = os.getenv('LOGFIRE_TOKEN')
    if logfire_token:
        logfire.configure(token=logfire_token)
        logger.info("Logfire configured")
    else:
        logger.info("Standard logging only (Logfire token not configured)")
except Exception:
    logger.info("Standard logging only (Logfire not available)")


# Constants
TEMPORAL_KEYWORDS = [
    "latest", "current", "recent", "today", "2025", "2024", "now",
    "new", "updated", "this year", "this month", "cve-", 
    "breaking", "trending", "live"
]

NO_INFO_INDICATORS = [
    "i do not have", "i don't have", "no relevant",
    "not enough information", "unfortunately", "cannot find",
    "no information available", "no documentation found",
    "not in the documentation"
]

UNCLEAR_INDICATORS = [
    "insufficient", "unclear", "more details", "please provide",
    "not specific", "limited information", "partial", "may not be",
    "it depends", "context needed", "additional information"
]


# Type Definitions

class AgentRole(str, Enum):
    """Agent roles in the multi-agent system"""
    ROUTER = "router"
    RETRIEVER_LOCAL = "retriever_local"
    RETRIEVER_WEB = "retriever_web"
    GRADER = "grader"
    GENERATOR = "generator"
    SUPERVISOR = "supervisor"


class AgentAction(BaseModel):
    """Action that an agent wants to perform"""
    action: str = Field(..., description="Action name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    reasoning: str = Field(..., description="Why this action is needed")


class AgentState(BaseModel):
    """State shared across agents in the workflow"""
    query: str = Field(..., description="User's question")
    local_context: Optional[str] = None
    local_sources: List[str] = Field(default_factory=list)
    web_context: Optional[str] = None
    web_sources: List[str] = Field(default_factory=list)
    strategy: Optional[str] = None
    grading_result: Optional['GradingScore'] = None
    final_answer: Optional[str] = None
    confidence_score: float = 0.0
    iteration_count: int = 0
    max_iterations: int = 3


class LocalRAGAssessment(BaseModel):
    """Assessment of local RAG information quality by Router Agent"""
    has_info: bool = Field(..., description="Whether local RAG has any relevant information")
    info_quality: str = Field(..., description="Quality level: 'none', 'partial', 'sufficient'")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in the information 0-1")
    temporal_keywords_detected: bool = Field(default=False, description="Whether temporal keywords found")
    reasoning: str = Field(..., description="Explanation of the assessment")
    suggested_action: AgentAction = Field(..., description="Recommended next action")


class RAGDecision(BaseModel):
    """Decision on which RAG strategy to use by Router Agent"""
    strategy: str = Field(..., description="Strategy: 'local_only', 'web_only', 'generate_grade_augment'")
    reasoning: str = Field(..., description="Why this strategy was chosen")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in this decision")
    next_agent: AgentRole = Field(..., description="Which agent should act next")


class GradingScore(BaseModel):
    """Response grading scores from Grader Agent"""
    relevancy: float = Field(..., ge=0, le=1, description="Relevancy score 0-1")
    faithfulness: float = Field(..., ge=0, le=1, description="Faithfulness to context 0-1")
    context_quality: float = Field(..., ge=0, le=1, description="Context quality 0-1")
    needs_web_search: bool = Field(..., description="Whether web search is needed")
    needs_regeneration: bool = Field(default=False, description="Whether answer needs to be regenerated")
    reasoning: str = Field(..., description="Explanation of grading")
    improvement_suggestions: List[str] = Field(default_factory=list, description="How to improve the answer")


class RAGResponse(BaseModel):
    """Final RAG response"""
    answer: str = Field(..., description="The generated answer")
    sources: List[str] = Field(default_factory=list, description="Sources used")
    grading: Optional[GradingScore] = Field(None, description="Self-grading scores if applicable")
    used_web_search: bool = Field(default=False, description="Whether web search was used")
    strategy_used: str = Field(..., description="Strategy that was applied")
    local_assessment: Optional[LocalRAGAssessment] = Field(None, description="Assessment of local information")
    agent_workflow: List[str] = Field(default_factory=list, description="Sequence of agents involved")
    iterations: int = Field(default=1, description="Number of refinement iterations")


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
                
                logger.debug(f"LightRAG raw response: {result}")
                
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
            logger.error(f"LightRAG Query Error: {error_msg}")
            return {
                "response": f"Error querying local RAG: {error_msg}",
                "sources": [],
                "success": False
            }


# 3. WEB SEARCH INTEGRATION (TAVILY)

class TavilySearchClient:
    """Client for Tavily web search"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("TAVILY_API_KEY not set. Web search will be unavailable.")
    
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
            # Allow disabling SSL verification for testing/local environments by
            # setting the environment variable TAVILY_DISABLE_SSL=1. By default
            # verification is enabled.
            disable_ssl = os.getenv("TAVILY_DISABLE_SSL", "0") == "1"
            async with httpx.AsyncClient(timeout=30.0, verify=(not disable_ssl)) as client:
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
            logger.error(f"Tavily Search Error: {e}")
            return {
                "results": [],
                "success": False,
                "error": str(e)
            }


# 4. GRADING SYSTEM

async def grade_response(
    query: str, 
    answer: str, 
    context: str, 
    sources: List[str]
) -> GradingScore:
    """
    Grader Agent: Grade the quality and relevance of a generated response
    
    This agent evaluates:
    - Relevancy: How well the answer addresses the query
    - Faithfulness: How accurately it reflects the context
    - Context Quality: How good the retrieved context is
    - Need for Web Search: Whether additional web info is needed
    - Need for Regeneration: Whether answer should be regenerated
    
    Uses a detailed rubric with specific criteria and scoring.
    
    Args:
        query: Original question
        answer: Generated answer to grade
        context: Context used to generate the answer
        sources: Sources referenced
    
    Returns:
        Detailed grading with scores and improvement suggestions
    """
    
    grading_prompt = f"""You are a Grader Agent responsible for evaluating the quality of RAG responses.

Evaluate this response against a detailed rubric:

**Query:** {query}

**Generated Answer:**
{answer}

**Context Used:**
{context[:2000]}

**Sources:** {len(sources)} sources provided

**GRADING RUBRIC:**

1. **Relevancy (0-1):**
   - 1.0: Directly answers the question with all requested information
   - 0.7: Answers most of the question but missing minor details
   - 0.5: Partially relevant but missing key information
   - 0.3: Tangentially related but doesn't answer the question
   - 0.0: Completely irrelevant

2. **Faithfulness (0-1):**
   - 1.0: All statements supported by context, no hallucinations
   - 0.8: Mostly faithful with minor interpretations
   - 0.5: Some unsupported claims or interpretations
   - 0.3: Significant unsupported claims
   - 0.0: Mostly fabricated or contradicts context

3. **Context Quality (0-1):**
   - 1.0: Context is complete, relevant, and sufficient
   - 0.7: Context is good but could be more complete
   - 0.5: Context is partial or somewhat relevant
   - 0.3: Context is poor quality or barely relevant
   - 0.0: Context is irrelevant or missing

4. **Needs Web Search:** Determine if answer would benefit from web search:
   - TRUE if: answer is incomplete, context is insufficient, query about current events, temporal keywords present
   - FALSE if: answer is complete and well-supported by context

5. **Needs Regeneration:** Determine if answer should be regenerated:
   - TRUE if: faithfulness < 0.5, relevancy < 0.6, or significant quality issues
   - FALSE if: answer quality is acceptable

6. **Improvement Suggestions:** List 2-3 specific ways to improve the answer

**CRITICAL:** Return ONLY a JSON object with this structure:
{{
    "relevancy": <float 0-1>,
    "faithfulness": <float 0-1>,
    "context_quality": <float 0-1>,
    "needs_web_search": <boolean>,
    "needs_regeneration": <boolean>,
    "reasoning": "<detailed explanation of scores>",
    "improvement_suggestions": ["<suggestion 1>", "<suggestion 2>", "<suggestion 3>"]
}}

Analyze carefully and provide accurate scores."""
    
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a Grader Agent that evaluates RAG response quality. Return only valid JSON."},
                {"role": "user", "content": grading_prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"   Grader Agent response received")
        
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
        logger.error(f"Grader Agent: Failed to parse response: {e}")
        logger.error(f"   Raw response: {response_text[:200] if 'response_text' in locals() else 'No response'}")
        return GradingScore(
            relevancy=0.5,
            faithfulness=0.5,
            context_quality=0.5,
            needs_web_search=True,
            needs_regeneration=False,
            reasoning="JSON parsing error in grading system, defaulting to web search",
            improvement_suggestions=["Regenerate with clearer instructions"]
        )
    except Exception as e:
        logger.error(f"Grader Agent error: {e}")
        return GradingScore(
            relevancy=0.5,
            faithfulness=0.5,
            context_quality=0.5,
            needs_web_search=True,
            needs_regeneration=False,
            reasoning=f"Grading system error: {str(e)}, defaulting to web search",
            improvement_suggestions=["System error - retry needed"]
        )


# 5. HELPER FUNCTIONS

async def assess_local_information(query: str, local_response: str, local_sources: List[str]) -> LocalRAGAssessment:
    """
    Router Agent: Assess the quality and sufficiency of local RAG information
    
    This agent analyzes local RAG results and decides the routing strategy.
    
    Strategy:
    - NONE: No local info → WEB_ONLY
    - PARTIAL: Has some info but unclear → GENERATE_GRADE_AUGMENT
    - SUFFICIENT: Has clear, complete info → LOCAL_ONLY
    
    Args:
        query: The original query
        local_response: Response from local RAG
        local_sources: Sources returned by local RAG
    
    Returns:
        Assessment of local information quality with suggested action
    """
    
    
    response_lower = local_response.lower() if local_response else ""
    has_explicit_no_info = any(indicator in response_lower for indicator in NO_INFO_INDICATORS)
    
    if has_explicit_no_info:
        logger.info("   Router Agent: No info (explicit indicator detected)")
        return LocalRAGAssessment(
            has_info=False,
            info_quality="none",
            confidence_score=0.0,
            temporal_keywords_detected=False,
            reasoning="LightRAG explicitly indicated no relevant information available",
            suggested_action=AgentAction(
                action="web_search",
                parameters={"query": query},
                reasoning="Local RAG has no information, must search web"
            )
        )
    
    # Check if we have actual content
    has_sources = bool(local_sources and len(local_sources) > 0)
    has_meaningful_response = bool(local_response and len(local_response.strip()) > 20)
    
    # No sources AND no meaningful response = definitely no info
    if not has_sources and not has_meaningful_response:
        logger.info("   Router Agent: No info (no sources or response)")
        return LocalRAGAssessment(
            has_info=False,
            info_quality="none",
            confidence_score=0.0,
            temporal_keywords_detected=False,
            reasoning="No sources or meaningful response from local RAG",
            suggested_action=AgentAction(
                action="web_search",
                parameters={"query": query},
                reasoning="Empty local results, must search web"
            )
        )
    
    query_lower = query.lower()
    needs_current_data = any(keyword in query_lower for keyword in TEMPORAL_KEYWORDS)
    
    is_unclear = any(indicator in response_lower for indicator in UNCLEAR_INDICATORS)
    response_length = len(local_response.strip())
    
    # Determine quality level and routing action
    if is_unclear or response_length < 100 or needs_current_data:
        reason_parts = []
        if is_unclear:
            reason_parts.append("response indicates uncertainty")
        if response_length < 100:
            reason_parts.append(f"short response ({response_length} chars)")
        if needs_current_data:
            reason_parts.append("query requires current/temporal data")
        
        logger.info(f"   Router Agent: Partial ({', '.join(reason_parts)})")
        return LocalRAGAssessment(
            has_info=True,
            info_quality="partial",
            confidence_score=0.4,
            temporal_keywords_detected=needs_current_data,
            reasoning=f"Local RAG has info but: {', '.join(reason_parts)}. Sources: {len(local_sources)}",
            suggested_action=AgentAction(
                action="generate_grade_augment",
                parameters={
                    "local_context": local_response,
                    "local_sources": local_sources,
                    "needs_web": needs_current_data
                },
                reasoning="Partial local info requires generation, grading, and possible augmentation"
            )
        )
    
    # Sufficient information: has sources, meaningful response, no uncertainty
    logger.info(f"   Router Agent: Sufficient (length: {response_length}, sources: {len(local_sources)})")
    return LocalRAGAssessment(
        has_info=True,
        info_quality="sufficient",
        confidence_score=0.8,
        temporal_keywords_detected=needs_current_data,
        reasoning=f"Local RAG provides clear information (length: {response_length}, sources: {len(local_sources)})",
        suggested_action=AgentAction(
            action="use_local_only" if not needs_current_data else "generate_grade_augment",
            parameters={"local_context": local_response, "local_sources": local_sources},
            reasoning="High quality local info" if not needs_current_data else "Good local info but temporal query needs verification"
        )
    )


async def decide_rag_strategy(assessment: LocalRAGAssessment, query: str) -> RAGDecision:
    """
    Router Agent: Decide which RAG strategy to use based on local information assessment
    
    This is the core routing logic that determines the workflow path.
    
    Three strategies:
    1. LOCAL_ONLY: Local info is sufficient and confident (quality='sufficient', confidence >= 0.7, NO temporal keywords)
    2. WEB_ONLY: No local info available (quality='none')
    3. GENERATE_GRADE_AUGMENT: Local info partial/unclear OR temporal keywords present
    
    Args:
        assessment: Assessment of local information
        query: Original query to check for temporal keywords
    
    Returns:
        Decision on RAG strategy with next agent
    """
    if assessment.info_quality == "none":
        logger.info("   Router Decision: WEB_ONLY (no local info)")
        return RAGDecision(
            strategy="web_only",
            reasoning="No local information available, using web search only",
            confidence=0.9,
            next_agent=AgentRole.RETRIEVER_WEB
        )
    
    query_lower = query.lower()
    needs_current_data = any(keyword in query_lower for keyword in TEMPORAL_KEYWORDS)
    
    # Case 2: Sufficient local info AND no temporal requirements → Local Only
    if (assessment.info_quality == "sufficient" and 
        assessment.confidence_score >= 0.7 and 
        not needs_current_data):
        logger.info("   Router Decision: LOCAL_ONLY (sufficient + no temporal)")
        return RAGDecision(
            strategy="local_only",
            reasoning="Local information is sufficient and confident, no current data needed",
            confidence=0.9,
            next_agent=AgentRole.GENERATOR
        )
    
    # Case 3: Everything else → Generate + Grade + Augment
    reason_parts = []
    if assessment.info_quality == "partial":
        reason_parts.append("local info is partial/unclear")
    if needs_current_data:
        reason_parts.append("query requires current/temporal data")
    if assessment.confidence_score < 0.7:
        reason_parts.append(f"low confidence ({assessment.confidence_score:.2f})")
    
    reasoning = ", ".join(reason_parts) if reason_parts else "need verification and potential augmentation"
    logger.info(f"   Router Decision: GENERATE_GRADE_AUGMENT ({reasoning})")
    
    return RAGDecision(
        strategy="generate_grade_augment",
        reasoning=f"Will generate from local, grade quality, and augment if needed: {reasoning}",
        confidence=0.8,
        next_agent=AgentRole.GENERATOR
    )


class RAGAgent:
    """Generator Agent: Produces answers using OpenAI API"""
    
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model
        self._client = None
        self.system_prompt = """
You are an expert RAG (Retrieval-Augmented Generation) assistant specializing in answering questions using retrieved documentation.

## Core Responsibilities

1. **Answer questions accurately** using the provided context from local documentation or web search
2. **Ground all responses in sources** - NEVER fabricate information not in the context
3. **Provide clear, structured answers** with relevant examples
4. **Cite sources explicitly** to enable verification

## Response Guidelines

### When Using Local Documentation
- Prioritize information from the provided local context
- Quote or paraphrase relevant sections accurately
- Cite using format: [Local: document_name] or [Source: section_name]
- Provide code examples from the documentation when helpful
- Be thorough and comprehensive
- DO NOT mention web sources unless explicitly provided

### When Using Web Search Results
- Synthesize information from multiple web sources
- Cite each source clearly: [Web: Source 1], [Web: Source 2], etc.
- Include URLs for users to verify information
- Note the source for each claim or fact
- If sources conflict, present both perspectives with citations

### When Using Both Local + Web
- Clearly distinguish between local documentation and web sources
- Use format: [Local: doc_name] vs [Web: Source N]
- Resolve conflicts by noting differences: "Local docs indicate X, while recent web sources show Y"
- Prefer local docs for foundational concepts, web for current/temporal data

### Answer Structure
1. **Direct Answer**: Start with a clear, concise response to the question
2. **Detailed Explanation**: Provide context and comprehensive details from sources
3. **Examples**: Include relevant code snippets, use cases, or specific examples
4. **Citations**: Reference specific sources throughout (not just at the end)
5. **Additional Context**: Note any limitations, caveats, or related topics

## Quality Standards

- **Accuracy**: Only state what is supported by the provided sources
- **Completeness**: Address all parts of the user's question
- **Clarity**: Use clear language, proper markdown formatting
- **Relevance**: Focus strictly on information that answers the question
- **Honesty**: Explicitly state when information is uncertain or missing from sources

## Handling Edge Cases

- **Insufficient Context**: Clearly state "The provided context does not contain information about X"
- **Conflicting Information**: Present both perspectives with source citations
- **Outdated Information**: Note if documentation appears outdated (e.g., refers to old versions)
- **Partial Information**: Answer what you can, explicitly note what's missing

## Formatting Requirements

- Use markdown for better readability
- Include code blocks with appropriate syntax highlighting (```python, ```javascript, etc.)
- Use bullet points for lists
- Use headers (##, ###) to organize longer responses
- **Bold** key terms and important concepts
- Cite sources inline, not just at the end

## Critical Rules

1. **NO HALLUCINATION**: If information isn't in the context, say so explicitly
2. **CITE EVERYTHING**: Every claim should be traceable to a source
3. **BE EXPLICIT**: When uncertain, say "The context suggests..." or "Based on Source 1..."
4. **DISTINGUISH SOURCES**: Always make clear whether info is from local docs or web

Remember: Your primary value is **accuracy and trustworthiness**. When in doubt, be explicit about uncertainty rather than guessing.
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
rag_agent = RAGAgent(model=os.getenv("RAG_AGENT_MODEL", "gpt-4.1-mini"))


# 7. MAIN QUERY FUNCTION

async def query_rag_system(query: str) -> RAGResponse:
    """
    Multi-Agent RAG System with Supervisor orchestration
    
    Workflow:
    1. Router Agent: Assess local info and decide strategy
    2. Retriever Agents: Fetch context (local/web)
    3. Generator Agent: Create answer
    4. Grader Agent: Evaluate quality
    5. Supervisor Agent: Decide if refinement needed (up to 3 iterations)
    
    Three strategies:
    1. LOCAL_ONLY: Local has sufficient info, no temporal keywords
    2. WEB_ONLY: Local has NO info
    3. GENERATE_GRADE_AUGMENT: Local has some info but unclear OR temporal keywords
    """
    
    logger.info(f"\n{'='*70}")
    logger.info(f"MULTI-AGENT RAG QUERY: {query}")
    logger.info(f"{'='*70}\n")
    
    # Initialize agent workflow tracking
    agent_workflow = []
    iteration = 0
    max_iterations = 3
    
    # Step 1: Router Agent - Query local RAG and assess information quality
    agent_workflow.append(AgentRole.ROUTER.value)
    logger.info("Router Agent: Querying Local RAG...")
    local_context = await light_rag.query(query=query, mode="hybrid", top_k=50)
    
    local_response = local_context.get("response", "") if local_context.get("success") else ""
    local_sources = local_context.get("sources", [])
    
    logger.info("Router Agent: Assessing Local Information Quality...")
    assessment = await assess_local_information(query, local_response, local_sources)
    logger.info(f"   Assessment: {assessment.info_quality} (confidence: {assessment.confidence_score:.2f})")
    logger.info(f"   Reasoning: {assessment.reasoning}")
    
    # Router Agent - Decide on RAG strategy
    decision = await decide_rag_strategy(assessment, query)
    logger.info(f"Router Decision: {decision.strategy} (confidence: {decision.confidence:.2f})")
    logger.info(f"   Reasoning: {decision.reasoning}")
    logger.info(f"   Next Agent: {decision.next_agent.value}")
    
    # Initialize variables
    final_answer = ""
    used_web_search = False
    grading = None
    sources = local_sources.copy()
    
    # Step 2: Execute the chosen strategy with agent coordination
    if decision.strategy == "local_only":
        # Case 1: Local has sufficient info → use local only
        agent_workflow.append(AgentRole.GENERATOR.value)
        logger.info("\nCASE 1: LOCAL_ONLY - Generator Agent using local documentation")
        prompt = f"""
User Question: {query}

Local Documentation Context:
{local_response}

**Instructions:**
- The local documentation contains sufficient information to answer this question
- Provide a comprehensive, detailed answer based ONLY on the local context above
- Cite specific document sections or sources when making claims
- Use clear formatting with headers, bullet points, and code examples where appropriate
- Be thorough - the user trusts that local docs have complete information
- DO NOT mention web sources or external information

Provide your answer now:
        """
        final_answer = await rag_agent.query(prompt)
        
    elif decision.strategy == "web_only":
        # Case 2: Local has NO info → skip to web search
        agent_workflow.append(AgentRole.RETRIEVER_WEB.value)
        agent_workflow.append(AgentRole.GENERATOR.value)
        logger.info("\nCASE 2: WEB_ONLY - Retriever Agent performing web search")
        web_results = await tavily.search(query=query, max_results=5)
        
        if web_results.get("success"):
            used_web_search = True
            sources = [r["url"] for r in web_results.get("results", [])[:5]]
            
            # Format web results nicely
            web_context = ""
            for i, result in enumerate(web_results.get("results", [])[:5], 1):
                web_context += f"\n--- Source {i} ---\n"
                web_context += f"Title: {result.get('title', 'N/A')}\n"
                web_context += f"URL: {result.get('url', 'N/A')}\n"
                web_context += f"Content: {result.get('content', 'N/A')}\n"
            
            prompt = f"""
User Question: {query}

Web Search Results:
{web_context}

**Instructions:**
- The local documentation does NOT contain information about this topic
- Provide a comprehensive answer based ONLY on the web search results above
- Synthesize information from multiple sources when available
- Cite each source clearly using [Source 1], [Source 2], etc.
- Include relevant URLs for users to learn more
- Be clear about which source each piece of information comes from
- If results conflict, acknowledge both perspectives

Provide your answer now:
            """
            final_answer = await rag_agent.query(prompt)
        else:
            final_answer = "I'm unable to find relevant information. The local documentation doesn't cover this topic and web search is currently unavailable. Please try rephrasing your question or check back later."
            
    else:
        # Case 3: Local has some info but unclear → generate + grade + augment with iteration
        agent_workflow.append(AgentRole.GENERATOR.value)
        logger.info("\nCASE 3: GENERATE_GRADE_AUGMENT - Multi-agent workflow with iterations")
        
        # Supervisor Agent: Manage refinement iterations
        current_answer = ""
        current_context = local_response
        current_sources = sources.copy()
        
        for iteration in range(max_iterations):
            logger.info(f"\n   Iteration {iteration + 1}/{max_iterations}")
            
            # Generator Agent: Create answer from current context
            agent_workflow.append(AgentRole.GENERATOR.value)
            logger.info(f"   Generator Agent: Creating answer (iteration {iteration + 1})...")
            generation_prompt = f"""
User Question: {query}

Context (Local + Web):
{current_context}

**Instructions:**
- Generate a comprehensive answer based on the context provided
- If the context is unclear or incomplete, explicitly state what's missing
- Be honest about limitations in the available information
- Cite sources explicitly when making claims
- Use clear formatting

{"Previous attempt had issues. Improvements needed: " + ", ".join(grading.improvement_suggestions) if iteration > 0 and grading else ""}

Provide your answer:
            """
            current_answer = await rag_agent.query(generation_prompt)
            logger.info(f"   Generated answer: {current_answer[:150]}...")
            
            # Grader Agent: Evaluate the response
            agent_workflow.append(AgentRole.GRADER.value)
            logger.info(f"   Grader Agent: Evaluating response quality...")
            grading = await grade_response(
                query=query,
                answer=current_answer,
                context=current_context,
                sources=current_sources
            )
            
            logger.info(f"      Relevancy: {grading.relevancy:.2f}")
            logger.info(f"      Faithfulness: {grading.faithfulness:.2f}")
            logger.info(f"      Context Quality: {grading.context_quality:.2f}")
            logger.info(f"      Needs Web Search: {grading.needs_web_search}")
            logger.info(f"      Needs Regeneration: {grading.needs_regeneration}")
            
            # Supervisor Agent: Decide next action
            logger.info(f"   Supervisor Agent: Evaluating workflow state...")
            
            temporal_keywords_in_query = any(keyword in query.lower() for keyword in TEMPORAL_KEYWORDS)
            
            if temporal_keywords_in_query and not used_web_search:
                logger.info(f"   Supervisor: Temporal query detected, MUST do web search")
                agent_workflow.append(AgentRole.RETRIEVER_WEB.value)
                logger.info(f"   Supervisor: Routing to Web Retriever Agent for current data...")
                web_results = await tavily.search(query=query, max_results=5)
                
                if web_results.get("success"):
                    used_web_search = True
                    web_sources = [result.get("url", "") for result in web_results.get("results", [])]
                    web_context = "\n\n".join([
                        f"Source: {result.get('url', 'Unknown')}\n{result.get('content', '')}"
                        for result in web_results.get("results", [])
                    ])
                    
                    # Augment context with web results
                    current_context = f"{local_response}\n\n--- WEB SEARCH RESULTS ---\n\n{web_context}"
                    current_sources.extend(web_sources)
                    sources.extend(web_sources)
                    
                    logger.info(f"   Web search completed, augmented context with {len(web_sources)} sources")
                    continue  # Retry generation with augmented context
                else:
                    logger.warning("   Web search failed, continuing with current context")
            
            # Check if quality is acceptable
            quality_threshold = 0.6
            is_quality_acceptable = (
                grading.relevancy >= quality_threshold and
                grading.faithfulness >= quality_threshold and
                not grading.needs_regeneration
            )
            
            if is_quality_acceptable and not grading.needs_web_search:
                logger.info(f"   Supervisor: Quality acceptable, finalizing answer")
                final_answer = current_answer
                break
            
            # If needs web search and haven't searched yet (non-temporal case)
            if grading.needs_web_search and not used_web_search:
                agent_workflow.append(AgentRole.RETRIEVER_WEB.value)
                logger.info(f"   Supervisor: Routing to Web Retriever Agent...")
                web_results = await tavily.search(query=query, max_results=5)
                
                if web_results.get("success"):
                    used_web_search = True
                    web_sources = [result.get("url", "") for result in web_results.get("results", [])]
                    web_context = "\n\n".join([
                        f"Source: {result.get('url', 'Unknown')}\n{result.get('content', '')}"
                        for result in web_results.get("results", [])
                    ])
                    
                    # Augment context with web results
                    current_context = f"{local_response}\n\n--- WEB SEARCH RESULTS ---\n\n{web_context}"
                    current_sources.extend(web_sources)
                    sources.extend(web_sources)
                    
                    logger.info(f"   Web search completed, augmented context with {len(web_sources)} sources")
                    continue  # Retry generation with augmented context
                else:
                    logger.warning("   Web search failed, continuing with current context")
            
            # If needs regeneration but at max iterations
            if iteration == max_iterations - 1:
                logger.info(f"   Supervisor: Max iterations reached, using best available answer")
                final_answer = current_answer
                break
            
            # Otherwise, continue to next iteration with improvements
            logger.info(f"   Supervisor: Quality not sufficient, scheduling regeneration...")
        
        # If no answer was finalized, use the last generated answer
        if not final_answer:
            final_answer = current_answer
            
    # Return final response with agent workflow tracking
    logger.info(f"\n{'='*70}")
    logger.info(f"FINAL RESPONSE READY")
    logger.info(f"   Strategy: {decision.strategy}")
    logger.info(f"   Iterations: {iteration + 1}")
    logger.info(f"   Agent Workflow: {' → '.join(agent_workflow)}")
    logger.info(f"   Sources: {len(sources)}")
    logger.info(f"   Used Web Search: {used_web_search}")
    logger.info(f"{'='*70}\n")
    
    return RAGResponse(
        answer=final_answer,
        sources=sources,
        grading=grading,
        used_web_search=used_web_search,
        strategy_used=decision.strategy,
        local_assessment=assessment,
        agent_workflow=agent_workflow,
        iterations=iteration + 1
    )


# 8. EXPORT FOR FASTAPI INTEGRATION

# This function will be called by the FastAPI endpoint
async def handle_query(question: str) -> dict:
    """
    Handle RAG query from FastAPI endpoint
    
    Returns enhanced response with multi-agent workflow information
    
    Args:
        question: User's question
    
    Returns:
        Dictionary with answer, sources, metadata, and agent workflow
    """
    try:
        response = await query_rag_system(question)
        
        # Build beautiful metadata section
        metadata_lines = []
        
        # Add spacing before metadata
        metadata_lines.append("\n\n")
        
        # Agent workflow - compact format
        if response.agent_workflow:
            workflow_display = " → ".join(response.agent_workflow)
            iteration_text = f" (refined {response.iterations}x)" if response.iterations > 1 else ""
            metadata_lines.append(f"**� Workflow:** `{workflow_display}`{iteration_text}\n\n")
        
        # Quality scores - inline badges
        if response.grading:
            scores = []
            scores.append(f"**Relevancy** {int(response.grading.relevancy * 100)}%")
            scores.append(f"**Faithfulness** {int(response.grading.faithfulness * 100)}%")
            scores.append(f"**Quality** {int(response.grading.context_quality * 100)}%")
            metadata_lines.append(f"**📊 Scores:** {' · '.join(scores)}\n\n")
        
        # Sources - formatted list
        if response.sources:
            metadata_lines.append(f"**📚 Sources ({len(response.sources)}):**\n\n")
            
            for i, source in enumerate(response.sources[:10], 1):
                if source.startswith('http'):
                    # Web source - clickable link
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(source).netloc
                        metadata_lines.append(f"{i}. 🌐 [{domain}]({source})\n")
                    except:
                        metadata_lines.append(f"{i}. 🌐 {source}\n")
                else:
                    # Local document
                    metadata_lines.append(f"{i}. 📄 `{source}`\n")
            
            if len(response.sources) > 10:
                metadata_lines.append(f"\n*...and {len(response.sources) - 10} more sources*\n")
        
        # Combine answer with metadata
        final_answer = response.answer + "".join(metadata_lines)
        
        return {
            "answer": final_answer,
            "sources": response.sources,
            "strategy": response.strategy_used,
            "used_web_search": response.used_web_search,
            "agent_workflow": response.agent_workflow,
            "iterations": response.iterations,
            "local_assessment": {
                "quality": response.local_assessment.info_quality if response.local_assessment else None,
                "confidence": response.local_assessment.confidence_score if response.local_assessment else None,
                "reasoning": response.local_assessment.reasoning if response.local_assessment else None,
                "temporal_keywords_detected": response.local_assessment.temporal_keywords_detected if response.local_assessment else False,
            } if response.local_assessment else None,
            "grading": {
                "relevancy": response.grading.relevancy if response.grading else None,
                "faithfulness": response.grading.faithfulness if response.grading else None,
                "context_quality": response.grading.context_quality if response.grading else None,
                "needs_web_search": response.grading.needs_web_search if response.grading else None,
                "needs_regeneration": response.grading.needs_regeneration if response.grading else None,
                "reasoning": response.grading.reasoning if response.grading else None,
                "improvement_suggestions": response.grading.improvement_suggestions if response.grading else [],
            } if response.grading else None
        }
    except Exception as e:
        logger.error(f"Error handling query: {e}")
        import traceback
        traceback.print_exc()
        return {
            "answer": f"Error processing query: {str(e)}",
            "sources": [],
            "strategy": "ERROR",
            "used_web_search": False,
            "agent_workflow": [],
            "iterations": 0,
            "grading": None,
            "error": str(e)
        }


# 8. CLI INTERFACE

async def main():
    """Interactive CLI for RAG agentic system"""
    
    print("\n" + "="*70)
    print("RAG AGENTIC AI SYSTEM")
    print("="*70)
    print("Local RAG: http://localhost:9621")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            query = input("Enter your question: ").strip()
            
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
            
            print(f"\n📚 Sources Used ({len(response.sources)}):")
            for i, source in enumerate(response.sources, 1):
                print(f"  {i}. {source}")
            
            if response.grading:
                print(f"\n⭐ Grading Scores:")
                print(f"  Relevancy: {response.grading.relevancy:.2f}/1.0")
                print(f"  Faithfulness: {response.grading.faithfulness:.2f}/1.0")
                print(f"  Context Quality: {response.grading.context_quality:.2f}/1.0")
            
            print(f"\nStrategy: {response.strategy}")
            print(f"  Web Search Used: {'Yes' if response.used_web_search else 'No'}")
            print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
