"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

interface Session {
  id: string;
  name: string | null;
  messageCount: number;
  lastMessage: string | null;
  createdAt: string;
  updatedAt: string;
}

function timeAgo(dateStr: string) {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export default function DashboardPage() {
  const router = useRouter();
  const [user, setUser] = useState<{ name: string; email: string } | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("/api/auth/me").then((r) => r.json()),
      fetch("/api/sessions").then((r) => r.json()),
    ]).then(([u, s]) => {
      setUser(u.user);
      setSessions(s.sessions ?? []);
      setLoading(false);
    });
  }, []);

  const logout = async () => {
    await fetch("/api/auth/logout", { method: "POST" });
    router.push("/");
    router.refresh();
  };

  return (
    <div className="min-h-screen">
      {/* Top nav */}
      <header className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-800 bg-slate-950/90 px-6 py-4 backdrop-blur">
        <Link href="/" className="text-lg font-bold text-white">
          SurgAI
        </Link>
        <div className="flex items-center gap-4">
          {user && (
            <span className="text-sm text-slate-400">
              {user.name}
            </span>
          )}
          <button onClick={logout} className="btn-secondary text-sm py-1.5 px-3">
            Sign out
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-6 py-10">
        {/* Action bar */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">
              {user ? `Welcome back, ${user.name.split(" ")[0]}` : "Dashboard"}
            </h1>
            <p className="mt-1 text-sm text-slate-500">Your chat sessions with the surgical reference.</p>
          </div>
          <a
            href="/chat"
            className="btn-primary flex items-center gap-2"
          >
            <span>+</span>
            New chat
          </a>
        </div>

        {/* Sessions */}
        {loading ? (
          <div className="flex items-center justify-center py-20 text-slate-500">Loading…</div>
        ) : sessions.length === 0 ? (
          <div className="card flex flex-col items-center gap-4 py-16 text-center">
            <div className="text-4xl">💬</div>
            <p className="text-slate-400">No sessions yet. Start a new chat to begin.</p>
            <a href="/chat" className="btn-primary">
              Start your first chat
            </a>
          </div>
        ) : (
          <div className="space-y-3">
            {sessions.map((s) => (
              <Link
                key={s.id}
                href={`/dashboard/session/${s.id}`}
                className="card flex items-start justify-between gap-4 transition-colors hover:border-slate-700 hover:bg-slate-800/50"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="truncate font-medium text-white">
                      {s.name ?? "Untitled session"}
                    </span>
                    <span className="shrink-0 rounded-full bg-slate-800 px-2 py-0.5 text-xs text-slate-400">
                      {s.messageCount} msg{s.messageCount !== 1 ? "s" : ""}
                    </span>
                  </div>
                  {s.lastMessage && (
                    <p className="mt-1 truncate text-sm text-slate-500">{s.lastMessage}</p>
                  )}
                </div>
                <span className="shrink-0 text-xs text-slate-600">{timeAgo(s.updatedAt)}</span>
              </Link>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
