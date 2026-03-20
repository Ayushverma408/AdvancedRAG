import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SurgAI — Surgical Reference Assistant",
  description: "AI-powered surgical reference powered by Fischer's Mastery of Surgery",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
