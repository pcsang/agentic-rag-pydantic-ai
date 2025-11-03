import React from "react";

export function ReferenceChip({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      className="reference-chip"
      href={href}
      target="_blank"
      rel="noopener noreferrer"
    >
      <span className="chip-text">{children}</span>
      <svg className="chip-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="14" height="14" fill="currentColor" style={{ marginLeft: 6 }}>
        <path d="M14 3h7v7h-2V6.41l-9.29 9.3-1.41-1.42 9.3-9.29H14V3z" />
        <path d="M5 5h5V3H3v7h2V5z" />
      </svg>
    </a>
  );
}

export default ReferenceChip;
