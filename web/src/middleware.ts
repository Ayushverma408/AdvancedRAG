import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";

const PROTECTED_API = ["/api/sessions", "/api/auth/me"];
const PROTECTED_PAGES = ["/dashboard"];

function getSecret() {
  return new TextEncoder().encode(process.env.JWT_SECRET ?? "");
}

export async function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  const isProtectedApi = PROTECTED_API.some((p) => pathname.startsWith(p));
  const isProtectedPage = PROTECTED_PAGES.some((p) => pathname.startsWith(p));

  if (!isProtectedApi && !isProtectedPage) return NextResponse.next();

  const token = req.cookies.get("auth_token")?.value;

  if (!token) {
    if (isProtectedApi) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    return NextResponse.redirect(new URL("/login", req.url));
  }

  try {
    const { payload } = await jwtVerify(token, getSecret());
    const headers = new Headers(req.headers);
    headers.set("x-user-id", payload.userId as string);
    headers.set("x-user-email", payload.email as string);
    headers.set("x-user-name", payload.name as string);
    return NextResponse.next({ request: { headers } });
  } catch {
    if (isProtectedApi) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    const res = NextResponse.redirect(new URL("/login", req.url));
    res.cookies.delete("auth_token");
    return res;
  }
}

export const config = {
  matcher: ["/dashboard/:path*", "/api/sessions/:path*", "/api/auth/me"],
};
