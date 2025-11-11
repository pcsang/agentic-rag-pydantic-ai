import {
  CopilotRuntime,
  copilotRuntimeNextJSAppRouterEndpoint,
  OpenAIAdapter,
} from "@copilotkit/runtime";
import { NextRequest } from "next/server";

// Define RAG query action using CopilotKit format
const ragQueryAction = {
  name: "query_rag",
  description: "Query the RAG system. CRITICAL: Return the COMPLETE response including all metadata, sources, and links. DO NOT summarize or modify the response in any way.",
  parameters: [
    {
      name: "question",
      type: "string" as const,
      description: "The user's question to answer",
      required: true,
    },
  ],
  handler: async (args: { [key: string]: string }) => {
    const question = args.question;
    console.log("🔍 RAG Query:", question);
    try {
      const response = await fetch("http://localhost:8000/rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      
      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("✅ RAG Response received");
      console.log("   Answer length:", data.answer?.length || 0);
      console.log("   Sources count:", data.sources?.length || 0);
      console.log("   Workflow:", data.agent_workflow?.join(" → ") || "none");
      
      // Return the COMPLETE answer with all metadata
      return data.answer;
    } catch (error) {
      console.error("❌ RAG Error:", error);
      return `Error querying RAG system: ${error instanceof Error ? error.message : 'Unknown error'}`;
    }
  },
};

// Create runtime with RAG action
const runtime = new CopilotRuntime({
  actions: [ragQueryAction],
});

// Service adapter
const serviceAdapter = new OpenAIAdapter({
  model: process.env.OPENAI_MODEL || "gpt-4o-mini",
});

console.log("✅ CopilotRuntime initialized with RAG action");

// Build Next.js API route handlers
export const POST = async (req: NextRequest) => {
  try {
    console.log("📨 POST /api/copilotkit");
    
    const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
      runtime,
      serviceAdapter,
      endpoint: "/api/copilotkit",
    });
 
    return await handleRequest(req);
  } catch (error) {
    console.error("❌ Error in /api/copilotkit:", error);
    console.error("Stack:", error instanceof Error ? error.stack : 'No stack trace');
    throw error;
  }
};

export const GET = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });
 
  return handleRequest(req);
};
