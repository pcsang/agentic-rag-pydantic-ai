"use client";

import { ProverbsCard } from "@/components/proverbs";
import { WeatherCard } from "@/components/weather";
import { MoonCard } from "@/components/moon";
import ToolCallRenderer from "@/components/tool-renderer";
import ReferenceChip from "@/components/reference-chip";
import AgenticHeader from "@/components/agentic-header";
import { AgentState } from "@/lib/types";
import { useCoAgent, useCopilotAction, useCoAgentStateRender } from "@copilotkit/react-core";
import { CopilotKitCSSProperties } from "@copilotkit/react-ui";
import CopilotPopupUI from "./copilot-popup";
import { useState } from "react";

export default function CopilotKitPage() {
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
    return "#06b6d4"; // fallback to new cyan primary
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
      <AgenticHeader />
      <YourMainContent themeColor={themeColor} />
      <CopilotPopupUI />
    </main>
  );
}

function YourMainContent({ themeColor }: { themeColor: string }) {
  // 🪁 Shared State: https://docs.copilotkit.ai/pydantic-ai/shared-state
  const { state, setState } = useCoAgent<AgentState>({
    name: "my_agent",
    initialState: {
      proverbs: [
        "CopilotKit may be new, but its the best thing since sliced bread.",
      ],
    },
  })

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
      return <WeatherCard location={args.location} themeColor={themeColor} status={status} />
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

  // 🪁 Agentic Generative UI: render the agent's state inside the chat.
  // This will show the `proverbs` list managed by the server-side Pydantic AI agent
  // served as the agent named `my_agent`.
  useCoAgentStateRender<{ proverbs?: string[] }>({
    name: "my_agent",
    render: ({ state }) => {
      return (
        <div className="mt-4 text-white max-w-md w-full p-3 rounded-md bg-white/5">
          <div className="font-semibold">Agent State — Proverbs</div>
          <div className="flex flex-col gap-2 mt-2">
            {state.proverbs && state.proverbs.length > 0 ? (
              state.proverbs.map((p, i) => (
                <div key={i} className="text-sm">
                  {i + 1}. {p}
                </div>
              ))
            ) : (
              <div className="text-sm text-gray-200">No proverbs yet.</div>
            )}
          </div>
        </div>
      );
    },
  });

  return (
    <div
      style={{ backgroundColor: themeColor }}
      className="h-screen flex justify-center items-center flex-col transition-colors duration-300"
    >
      <ProverbsCard state={state} setState={setState} />
    </div>
  );
}
