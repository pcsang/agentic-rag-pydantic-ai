// State types for both agents

// RAG Agent State
export type RAGAgentState = {
  query_count: number;
  last_query: string;
  last_sources: string[];
  last_strategy: string;
}

// Jira Agent State  
export type JiraAgentState = {
  proverbs: string[];
}

// Default export for backward compatibility
export type AgentState = RAGAgentState;