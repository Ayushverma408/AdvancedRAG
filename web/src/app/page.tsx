import Link from "next/link";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      {/* Nav */}
      <nav className="flex items-center justify-between border-b border-slate-800 px-6 py-4">
        <span className="text-lg font-bold tracking-tight text-white">SurgAI</span>
        <div className="flex gap-3">
          <Link href="/login" className="btn-secondary text-sm py-2 px-4">
            Sign In
          </Link>
          <Link href="/signup" className="btn-primary text-sm py-2 px-4">
            Get Started
          </Link>
        </div>
      </nav>

      {/* Hero */}
      <main className="flex flex-1 flex-col items-center justify-center px-6 py-24 text-center">
        <div className="mb-4 inline-flex items-center rounded-full border border-indigo-500/30 bg-indigo-500/10 px-4 py-1.5 text-xs font-medium text-indigo-400">
          Powered by HyDE · Fischer&apos;s Surgery · GPT-4o
        </div>
        <h1 className="mx-auto max-w-3xl text-5xl font-bold leading-tight tracking-tight text-white">
          Your surgical reference,{" "}
          <span className="bg-gradient-to-r from-indigo-400 to-violet-400 bg-clip-text text-transparent">
            instantly answered.
          </span>
        </h1>
        <p className="mx-auto mt-6 max-w-xl text-lg text-slate-400">
          Ask anything about Fischer&apos;s Mastery of Surgery. Get precise, cited answers in seconds —
          0.93 faithfulness score on RAGAS evaluation.
        </p>
        <div className="mt-10 flex gap-4">
          <Link href="/signup" className="btn-primary px-8 py-3 text-base">
            Start for free
          </Link>
          <Link href="/login" className="btn-secondary px-8 py-3 text-base">
            Sign in
          </Link>
        </div>

        {/* Features */}
        <div className="mt-24 grid max-w-3xl grid-cols-1 gap-6 text-left sm:grid-cols-3">
          {[
            {
              icon: "🔬",
              title: "Medical-grade accuracy",
              body: "HyDE + hybrid retrieval + cross-encoder reranking. Built for surgical vocabulary.",
            },
            {
              icon: "📄",
              title: "Page-level citations",
              body: "Every answer references exact pages from Fischer's Surgery with attached figures.",
            },
            {
              icon: "💬",
              title: "Session history",
              body: "All your conversations are saved. Pick up right where you left off.",
            },
          ].map((f) => (
            <div key={f.title} className="card">
              <div className="mb-3 text-2xl">{f.icon}</div>
              <h3 className="mb-2 font-semibold text-white">{f.title}</h3>
              <p className="text-sm text-slate-400">{f.body}</p>
            </div>
          ))}
        </div>
      </main>

      <footer className="border-t border-slate-800 px-6 py-4 text-center text-xs text-slate-600">
        SurgAI — internal use only
      </footer>
    </div>
  );
}
