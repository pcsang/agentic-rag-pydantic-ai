import type { Metadata } from "next";
import "./globals.css";
import "@copilotkit/react-ui/styles.css";

export const metadata: Metadata = {
  title: "RAG & Jira Multi-Agent System",
  description: "Intelligent routing between RAG and Jira agents",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={"antialiased"}>
        {children}
      </body>
    </html>
  );
}
