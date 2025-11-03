import React from "react";

export default function AgenticHeader() {
  return (
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
        <div className="meta">v0.1</div>
        <button className="action-btn">Docs</button>
        <button className="action-btn">Kiet Ho</button>
      </div>
    </header>
  );
}
