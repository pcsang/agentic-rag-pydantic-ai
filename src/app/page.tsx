"use client";

import { CopilotKitCSSProperties, CopilotChat } from "@copilotkit/react-ui";
import { CopilotKit } from "@copilotkit/react-core";

export default function CopilotKitPage() {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit">
      <MainContent />
    </CopilotKit>
  );
}

function MainContent() {
  return (
    <main style={{ "--copilot-kit-primary-color": "#3b82f6" } as CopilotKitCSSProperties}>
      <div className="h-screen w-screen flex flex-col bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-slate-800">
        {/* Professional Header */}
        <header className="bg-gradient-to-r from-blue-700 via-blue-600 to-indigo-600 dark:from-blue-900 dark:via-slate-800 dark:to-slate-900 shadow-lg">
          <div className="max-w-full px-8 py-6 flex items-center justify-center">
            <div className="flex flex-col items-center gap-2 text-center">
              <div className="w-12 h-12 bg-white dark:bg-slate-700 rounded-lg flex items-center justify-center shadow-md">
                <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  📚
                </span>
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white tracking-tight">RAG Agentic</h1>
                <p className="text-sm text-blue-100 dark:text-slate-300 font-medium">Multi-Agent System</p>
              </div>
            </div>
          </div>
        </header>

        {/* Fullscreen Chat */}
        <div className="flex-1 overflow-hidden">
          <CopilotChat
            instructions="You are a RAG system proxy. Your ONLY job is to:

1. Call query_rag tool for EVERY user question
2. Return the COMPLETE tool response WORD-FOR-WORD, including:
   - All answer text
   - ALL metadata sections (workflow, scores, sources)
   - ALL source references and links
3. DO NOT summarize, paraphrase, or modify ANY part
4. DO NOT add your own commentary
5. DO NOT omit sources or metadata
6. Show EVERYTHING from the tool response

You are a pass-through. Return tool output verbatim."
            labels={{
              title: "Ask a question...",
              initial: "Ask about documentation and technical questions..."
            }}
            className="h-full w-full"
          />
        </div>
      </div>
    </main>
  );
}
