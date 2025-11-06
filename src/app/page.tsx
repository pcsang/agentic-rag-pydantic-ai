"use client";

import { WeatherCard } from "@/components/weather";
import { MoonCard } from "@/components/moon";
import ToolCallRenderer from "@/components/tool-renderer";
import AgenticHeader from "@/components/agentic-header";
import LeftSidebar from "@/components/left-sidebar";
import { useCopilotAction } from "@copilotkit/react-core";
import { CopilotKitCSSProperties, CopilotPopup } from "@copilotkit/react-ui";
import { CopilotKit } from "@copilotkit/react-core";
import { useState, useEffect } from "react";

export default function CopilotKitPage() {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit">
      <MainContent />
    </CopilotKit>
  );
}

function MainContent() {
  // Initialize themeColor from CSS variable when available, with SSR-safe fallback
  const [themeColor, setThemeColor] = useState(() => {
    try {
      if (typeof window !== "undefined") {
        const computed = getComputedStyle(document.documentElement).getPropertyValue("--primary");
        if (computed) return computed.trim();
      }
    } catch (e) {
      // ignore
    }
    return "#3b82f6"; // fallback to professional blue primary
  });

  // 🪁 Frontend Actions: https://docs.copilotkit.ai/pydantic-ai/frontend-actions
  useCopilotAction({
    name: "setThemeColor",
    parameters: [{
      name: "themeColor",
      description: "The theme color to set. Make sure to pick nice colors.",
      required: true, 
    }],
    handler({ themeColor }) {
      setThemeColor(themeColor);
    },
  });

  return (
    <main style={{ "--copilot-kit-primary-color": themeColor } as CopilotKitCSSProperties}>
      <YourMainContent themeColor={themeColor} />
    </main>
  );
}

function YourMainContent({ themeColor }: { themeColor: string }) {
  // State tracking - simplified without useCoAgent to avoid conflicts with dynamic routing
  const [queryCount, setQueryCount] = useState(0);
  const [showCopilot, setShowCopilot] = useState(false);
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

  //🪁 Generative UI: https://docs.copilotkit.ai/pydantic-ai/generative-ui
  // Render builtin web-search tool calls (name must match the tool registered server-side)
  useCopilotAction({
    name: "web_search",
    description: "Render frontend view for web search tool calls.",
    available: "frontend",
    render: ({ status, args }) => {
      return <ToolCallRenderer name="web_search" args={args} status={status} />
    },
  }, [themeColor]);

  useCopilotAction({
    name: "get_weather",
    description: "Get the weather for a given location.",
    // Mark this action as a render-only frontend action so the UI will render
    // the tool call when the agent invokes the `get_weather` tool.
    available: "frontend",
    parameters: [
      { name: "location", type: "string", required: true },
    ],
    render: ({ status, args }) => {
      // `status` can be e.g. 'pending' or 'complete' depending on the tool call lifecycle
      return <WeatherCard location={args.location} themeColor={themeColor} />
    },
  }, [themeColor]);

  // 🪁 Human In the Loop: https://docs.copilotkit.ai/pydantic-ai/human-in-the-loop
  useCopilotAction({
    name: "go_to_moon",
    description: "Go to the moon on request.",
    renderAndWaitForResponse: ({ respond, status}) => {
      return <MoonCard themeColor={themeColor} status={status} respond={respond} />
    },
  }, [themeColor]);

  // Remove useCoAgentStateRender to avoid agent mismatch errors
  // The agent state is managed server-side and displayed through tool renders

  return (
    <>
      <div className="h-screen flex flex-col">
        <AgenticHeader 
          onToggleChat={() => setShowCopilot(!showCopilot)}
          showCopilot={showCopilot}
          currentAgent={currentAgent}
        />
        <div className="flex flex-1 relative">
          {/* Left Sidebar */}
          <LeftSidebar />
          
          {/* Main Content Area with sidebar spacing */}
          <div className={`flex-1 ml-80 lg:ml-80 md:ml-24 sm:ml-0 transition-all duration-300 ${showCopilot ? 'mr-96' : 'mr-6'}`}>
            <div className="app-content">
              <div className="content-card">
                <div className="text-center">
                  <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                    Multi-Agent RAG System
                  </h1>
                  <p className="text-lg text-slate-600 dark:text-slate-400 mb-8">
                    Ask about documentation or Jira tickets
                  </p>
                  <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-slate-800 dark:to-slate-700 rounded-xl p-6 max-w-2xl mx-auto border border-slate-200 dark:border-slate-600">
                    <div className="text-sm text-slate-700 dark:text-slate-300">
                      <p className="font-semibold mb-3">🤖 Intelligent Agent Routing</p>
                      The system automatically routes your questions to the appropriate agent:
                      <div className="grid md:grid-cols-2 gap-4 mt-4">
                        <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-600">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-lg">📚</span>
                            <strong className="text-blue-600 dark:text-blue-400">RAG Agent</strong>
                          </div>
                          <p className="text-xs text-slate-600 dark:text-slate-400">
                            Technical documentation, LightRAG + Web search
                          </p>
                        </div>
                        <div className="bg-white dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-600">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-lg">🎫</span>
                            <strong className="text-indigo-600 dark:text-indigo-400">Jira Agent</strong>
                          </div>
                          <p className="text-xs text-slate-600 dark:text-slate-400">
                            Ticket management, JQL queries
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* CopilotPopup for custom chatbox UI */}
      <CopilotPopup
        instructions="You are assisting the user as best as you can. Answer in the best way possible given the data you have."
        labels={{
          title: currentAgent === "jira_agent" ? "🎫 Jira Assistant" : "📚 RAG Assistant",
          initial: currentAgent === "jira_agent" 
            ? "Ask about Jira tickets and project management..." 
            : "Ask about documentation and technical questions..."
        }}
      />
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
