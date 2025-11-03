import React from "react";

export function ToolCallRenderer({ name, args, status }: { name: string, args?: any, status?: string }) {
  return (
    <div className="rounded-md bg-white/10 p-3 mt-3 text-sm text-white max-w-md w-full">
      <div className="font-medium">Tool: {name}</div>
      <div className="text-xs text-gray-200">Status: {status ?? "pending"}</div>
      <pre className="mt-2 text-xs text-gray-100 overflow-auto">
        {JSON.stringify(args ?? {}, null, 2)}
      </pre>
    </div>
  );
}

export default ToolCallRenderer;
