"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";

interface Step {
  id: string;
  type: string;
  name: string | null;
  output: string | null;
  createdAt: string;
}

interface SessionDetail {
  id: string;
  name: string | null;
  createdAt: string;
  steps: Step[];
}

function formatDate(d: string) {
  return new Date(d).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function SessionPage() {
  const { id } = useParams<{ id: string }>();
  const [session, setSession] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    fetch(`/api/sessions/${id}`)
      .then((r) => {
        if (r.status === 404) { setNotFound(true); return null; }
        return r.json();
      })
      .then((data) => {
        if (data) setSession(data.session);
        setLoading(false);
      });
  }, [id]);

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 flex items-center gap-4 border-b border-slate-800 bg-slate-950/90 px-6 py-4 backdrop-blur">
        <Link href="/dashboard" className="text-slate-400 hover:text-white">
          ← Dashboard
        </Link>
        <span className="text-slate-700">/</span>
        <span className="truncate font-medium text-white">
          {session?.name ?? "Session transcript"}
        </span>
      </header>

      <main className="mx-auto max-w-3xl px-6 py-10">
        {loading ? (
          <div className="py-20 text-center text-slate-500">Loading…</div>
        ) : notFound ? (
          <div className="py-20 text-center text-slate-500">Session not found.</div>
        ) : session ? (
          <>
            <div className="mb-6 text-sm text-slate-500">
              Started {formatDate(session.createdAt)} · {session.steps.length} messages
            </div>
            <div className="space-y-4">
              {session.steps.map((step) => {
                const isUser = step.type === "user_message";
                return (
                  <div
                    key={step.id}
                    className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}
                  >
                    <div
                      className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm ${
                        isUser
                          ? "rounded-tr-sm bg-indigo-600 text-white"
                          : "rounded-tl-sm bg-slate-800 text-slate-100"
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{step.output ?? ""}</p>
                      <p className={`mt-1.5 text-xs ${isUser ? "text-indigo-300" : "text-slate-500"}`}>
                        {formatDate(step.createdAt)}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="mt-8 text-center">
              <a href="/chat" className="btn-primary">
                Continue in chat
              </a>
            </div>
          </>
        ) : null}
      </main>
    </div>
  );
}
