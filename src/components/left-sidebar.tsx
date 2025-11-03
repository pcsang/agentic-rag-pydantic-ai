"use client";
import React, { useEffect, useState } from "react";

type RecentChat = { id?: string; title: string; time?: string };

const MENU = [
  { key: "dashboard", label: "Dashboard", icon: DashboardIcon },
  { key: "chat", label: "Chat", icon: ChatIcon },
  { key: "documents", label: "Documents", icon: DocumentsIcon },
  { key: "analytics", label: "Analytics", icon: AnalyticsIcon },
];

export default function LeftSidebar({ className = "" }: { className?: string }) {
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try {
      const v = localStorage.getItem("rag_sidebar_collapsed");
      return v === "1";
    } catch (e) {
      return false;
    }
  });
  const [active, setActive] = useState<string>("chat");
  const [recent, setRecent] = useState<RecentChat[]>([]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem("rag_chat_history");
      if (raw) {
        const parsed = JSON.parse(raw);
        // parsed may be an array of strings or objects
        const items: RecentChat[] = Array.isArray(parsed)
          ? parsed
              .slice()
              .reverse()
              .slice(0, 50)
              .map((it: any) =>
                typeof it === "string"
                  ? { title: it, time: undefined }
                  : { title: it.title ?? String(it.message ?? it.text ?? "Untitled"), time: it.time }
              )
          : [];
        setRecent(items.slice(0, 3));
      }
    } catch (e) {
      // ignore parse errors
      setRecent([]);
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("rag_sidebar_collapsed", collapsed ? "1" : "0");
    } catch (e) {}
  }, [collapsed]);

  return (
    <aside className={`left-sidebar ${collapsed ? "collapsed" : "expanded"} ${className}`}>
      <div className="sidebar-top">
        <div className="logo-wrap" onClick={() => setActive("dashboard") } role="button" tabIndex={0}>
          <div className="logo-icon" aria-hidden>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="2" y="2" width="20" height="20" rx="4" fill="url(#g)" />
              <defs>
                <linearGradient id="g" x1="0" x2="1">
                  <stop offset="0" stopColor="var(--primary)" />
                  <stop offset="1" stopColor="var(--primary-dark)" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          {!collapsed && (
            <div className="logo-text">
              <div className="brand">RAG</div>
              <div className="tag">Agentic</div>
            </div>
          )}
        </div>

        <button
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className={`collapse-btn ${collapsed ? "rotated" : ""}`}
          onClick={() => setCollapsed((c) => !c)}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M9 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>

      <nav className="sidebar-nav">
        {MENU.map((m) => (
          <div
            key={m.key}
            role="button"
            tabIndex={0}
            className={`nav-item ${active === m.key ? "active" : ""}`}
            onClick={() => setActive(m.key)}
          >
            <div className="nav-icon">
              <m.icon />
            </div>
            {!collapsed && <div className="nav-label">{m.label}</div>}
          </div>
        ))}
      </nav>

      <div className="sidebar-recent">
        {!collapsed && <div className="section-title">Recent</div>}
        <div className="recent-list">
          {recent.length === 0 ? (
            <div className="recent-empty">No recent chats</div>
          ) : (
            recent.map((r, i) => (
              <div key={i} className="recent-item" title={r.title}>
                <div className="recent-dot" aria-hidden />
                {!collapsed ? (
                  <div className="recent-title">{truncate(r.title, 40)}</div>
                ) : (
                  <div className="recent-icon">{r.title.charAt(0).toUpperCase()}</div>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      <div className="sidebar-footer">
        <button className="footer-btn">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M12 2v4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M5 8h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          {!collapsed && <span>Settings</span>}
        </button>
        <button className="footer-btn danger">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M3 6h18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M10 6v12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M14 6v12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          {!collapsed && <span>Logout</span>}
        </button>
      </div>
    </aside>
  );
}

function truncate(s: string, n: number) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function IconWrapper({ children }: { children: React.ReactNode }) {
  return <div style={{ width: 20, height: 20 }}>{children}</div>;
}

function DashboardIcon() {
  return (
    <IconWrapper>
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <rect x="3" y="3" width="8" height="8" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
        <rect x="13" y="3" width="8" height="4" rx="1" stroke="currentColor" strokeWidth="1.5" />
        <rect x="13" y="9" width="8" height="12" rx="1" stroke="currentColor" strokeWidth="1.5" />
      </svg>
    </IconWrapper>
  );
}

function ChatIcon() {
  return (
    <IconWrapper>
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </IconWrapper>
  );
}

function DocumentsIcon() {
  return (
    <IconWrapper>
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M14 2v6h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </IconWrapper>
  );
}

function AnalyticsIcon() {
  return (
    <IconWrapper>
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <path d="M3 3v18h18" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M7 13v4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M12 9v8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M17 5v12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </IconWrapper>
  );
}
