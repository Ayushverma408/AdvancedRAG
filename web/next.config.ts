import type { NextConfig } from "next";

const config: NextConfig = {
  async rewrites() {
    const chainlitUrl = process.env.CHAINLIT_URL ?? "http://localhost:8000";
    return [
      { source: "/chat", destination: chainlitUrl },
      { source: "/chat/:path*", destination: `${chainlitUrl}/:path*` },
    ];
  },
};

export default config;
