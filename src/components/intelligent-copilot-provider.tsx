"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { CopilotPopup } from "@copilotkit/react-ui";
import { useState, useEffect } from "react";

/**
 * Intelligent CopilotKit Provider with automatic agent routing
 * Routes queries to appropriate agent based on content analysis
 */
export function CopilotKitProvider({ children }: { children: React.ReactNode }) {
  const [currentAgent, setCurrentAgent] = useState<"rag_agent" | "jira_agent">("rag_agent");
  const [availableAgents, setAvailableAgents] = useState<string[]>(["rag_agent"]);

  // Check available agents from backend
  useEffect(() => {
    const checkAgents = async () => {
      try {
        const response = await fetch('http://localhost:8000/');
        const data = await response.json();
        if (data.available_agents) {
          console.log("✅ Available agents:", data.available_agents);
          setAvailableAgents(data.available_agents);
        }
      } catch (error) {
        console.warn("⚠️ Could not check available agents");
        setAvailableAgents(["rag_agent"]);
      }
    };
    checkAgents();
  }, []);

  // Monitor input and auto-route based on content
  useEffect(() => {
    let lastValue = "";

    const checkInput = () => {
      const input = document.querySelector('textarea, [role="textbox"]') as HTMLTextAreaElement;
      
      if (input && input.value && input.value !== lastValue) {
        lastValue = input.value;
        const agent = determineAgent(input.value, availableAgents);
        
        if (agent !== currentAgent) {
          console.log(`🔄 Routing to ${agent}`);
          setCurrentAgent(agent);
        }
      }
    };

    const interval = setInterval(checkInput, 100);
    
    const handleInput = (e: Event) => {
      const target = e.target as HTMLTextAreaElement;
      if (target?.value) {
        const agent = determineAgent(target.value, availableAgents);
        if (agent !== currentAgent) {
          setCurrentAgent(agent);
        }
      }
    };

    document.addEventListener('input', handleInput);
    document.addEventListener('keyup', handleInput);

    return () => {
      clearInterval(interval);
      document.removeEventListener('input', handleInput);
      document.removeEventListener('keyup', handleInput);
    };
  }, [currentAgent, availableAgents]);

  return (
    <CopilotKit runtimeUrl="/api/copilotkit">
      {children}
      <CopilotPopup 
        labels={{
          title: currentAgent === "jira_agent" ? "🎫 Jira Assistant" : "📚 RAG Assistant",
          initial: currentAgent === "jira_agent" 
            ? "Ask about Jira tickets and project management..." 
            : "Ask about documentation and technical questions..."
        }}
      />
      {/* Agent indicator */}
      <div className="fixed bottom-24 right-6 bg-gradient-to-r from-blue-600 to-cyan-600 text-white px-4 py-2 rounded-full text-sm shadow-lg z-40 flex items-center gap-2">
        <span className="animate-pulse text-green-300">●</span>
        <span className="font-semibold">
          {currentAgent === "jira_agent" ? "🎫 Jira Agent" : "📚 RAG Agent"}
        </span>
      </div>
    </CopilotKit>
  );
}

/**
 * Determine which agent to use based on query content
 */
function determineAgent(query: string, available: string[]): "rag_agent" | "jira_agent" {
  const content = query.toLowerCase();
  
  // Jira keywords
  const jiraKeywords = [
    "jira", "ticket", "issue", "issues", "sprint", "epic", "story", "bug",
    "project", "board", "backlog", "workflow", "assignee", "assigned",
    "priority", "status", "task", "tasks", "subtask", "jql", "filter",
    "create issue", "update issue", "close issue", "resolve",
    "kanban", "scrum", "agile", "in jira", "from jira"
  ];

  const matchedJira = jiraKeywords.filter(kw => content.includes(kw));
  
  if (matchedJira.length > 0 && available.includes("jira_agent")) {
    console.log(`✅ Jira keywords: [${matchedJira.join(", ")}]`);
    return "jira_agent";
  }
  
  return "rag_agent";
}
