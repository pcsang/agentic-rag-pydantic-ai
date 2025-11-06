import {
  CopilotRuntime,
  copilotRuntimeNextJSAppRouterEndpoint,
  OpenAIAdapter,
} from "@copilotkit/runtime";
import { HttpAgent } from "@ag-ui/client";
import { NextRequest } from "next/server";

// Create OpenAI adapter for non-agent components
const serviceAdapter = new OpenAIAdapter({
  model: "gpt-4o-mini",
});

// Create SINGLE runtime with both agents registered
const runtime = new CopilotRuntime({
  agents: {
    rag_agent: new HttpAgent({ url: "http://localhost:8000/rag/" }),
    jira_agent: new HttpAgent({ url: "http://localhost:8000/jira/" }),
  },
});

console.log("✅ CopilotRuntime initialized with rag_agent and jira_agent");

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
