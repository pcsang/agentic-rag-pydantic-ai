"use client";

import React, { useState, useEffect } from "react";

interface AgenticHeaderProps {
  onToggleChat?: () => void;
  showCopilot?: boolean;
  currentAgent?: "rag_agent" | "jira_agent";
}

export default function AgenticHeader({ onToggleChat, showCopilot, currentAgent: propCurrentAgent }: AgenticHeaderProps) {
  const [currentAgent, setCurrentAgent] = useState<"rag_agent" | "jira_agent">(propCurrentAgent || "rag_agent");
  const [availableAgents, setAvailableAgents] = useState<string[]>(["rag_agent"]);

  // Update local state when prop changes
  useEffect(() => {
    if (propCurrentAgent) {
      setCurrentAgent(propCurrentAgent);
    }
  }, [propCurrentAgent]);

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

  // Auto-click CopilotPopup when showCopilot becomes true
  useEffect(() => {
    if (showCopilot) {
      const timer = setTimeout(() => {
        const toggleButton = document.querySelector('.copilotKitChatToggle') as HTMLButtonElement;
        if (toggleButton) {
          toggleButton.click();
        }
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [showCopilot]);

  // Handle sidebar toggle via CopilotKit APIs
  const handleChatToggle = () => {
    if (onToggleChat) {
      onToggleChat();
    }
  };

  return (
    <>
      <header className="agentic-header">
        <div className="brand">
          <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
            <defs>
              <linearGradient id="hdrg" x1="0" x2="1">
                <stop offset="0%" stopColor="var(--primary-dark)" />
                <stop offset="100%" stopColor="var(--primary)" />
              </linearGradient>
            </defs>
            <rect width="40" height="40" rx="8" fill="url(#hdrg)" />
            <g transform="translate(8,8)">
              <path d="M4 16 L12 4 L20 16" stroke="white" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" fill="none" />
              <path d="M6 12h8v6" stroke="white" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" fill="none" />
            </g>
          </svg>
          <div>
            <div className="title">Rag Agentic</div>
            <div className="subtitle">AI Copilot for RAG workflows</div>
          </div>
        </div>

        <div className="header-actions">
          {/* Agent Status Indicator */}
          <div className="flex items-center gap-3 px-4 py-2 bg-white/10 rounded-lg border border-white/20">
            <span className="animate-pulse text-green-300 text-xs">●</span>
            <span className="text-sm font-medium">
              {currentAgent === "jira_agent" ? "🎫 Jira Agent" : "📚 RAG Agent"}
            </span>
          </div>
          
          {/* Chat Toggle Button */}
          <button 
            onClick={handleChatToggle}
            className="action-btn flex items-center gap-2"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
            </svg>
            Chat
          </button>
          
          <div className="meta">v0.1</div>
          <button className="action-btn">Docs</button>
          <button className="action-btn">Kiet Ho</button>
        </div>
      </header>

      {/* Chat Popup with Auto-Open */}
      {/* CopilotSidebar integration moved to main layout */}
    </>
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
