import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { authService } from "@/services/auth.service";

const COOKIE = "auth_token";
const COOKIE_OPTS = {
  httpOnly: true,
  secure: process.env.NODE_ENV === "production",
  sameSite: "lax" as const,
  maxAge: 60 * 60 * 24 * 7,
  path: "/",
};

const signupSchema = z.object({
  name: z.string().min(2, "Name must be at least 2 characters"),
  email: z.string().email("Invalid email"),
  password: z.string().min(8, "Password must be at least 8 characters"),
});

const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(1),
});

export class AuthController {
  async signup(req: NextRequest) {
    try {
      const input = signupSchema.parse(await req.json());
      const { user, token } = await authService.signup(input);
      const res = NextResponse.json({ user }, { status: 201 });
      res.cookies.set(COOKIE, token, COOKIE_OPTS);
      return res;
    } catch (e: unknown) {
      if (e instanceof z.ZodError) {
        return NextResponse.json({ error: e.errors[0].message }, { status: 400 });
      }
      const err = e as Error & { code?: string };
      if (err.code === "EMAIL_EXISTS") {
        return NextResponse.json({ error: err.message }, { status: 409 });
      }
      return NextResponse.json({ error: "Internal server error" }, { status: 500 });
    }
  }

  async login(req: NextRequest) {
    try {
      const input = loginSchema.parse(await req.json());
      const { user, token } = await authService.login(input);
      const res = NextResponse.json({ user });
      res.cookies.set(COOKIE, token, COOKIE_OPTS);
      return res;
    } catch (e: unknown) {
      if (e instanceof z.ZodError) {
        return NextResponse.json({ error: e.errors[0].message }, { status: 400 });
      }
      const err = e as Error & { code?: string };
      if (err.code === "INVALID_CREDS") {
        return NextResponse.json({ error: err.message }, { status: 401 });
      }
      return NextResponse.json({ error: "Internal server error" }, { status: 500 });
    }
  }

  async logout() {
    const res = NextResponse.json({ ok: true });
    res.cookies.delete(COOKIE);
    return res;
  }

  async me(req: NextRequest) {
    return NextResponse.json({
      user: {
        id: req.headers.get("x-user-id"),
        email: req.headers.get("x-user-email"),
        name: req.headers.get("x-user-name"),
      },
    });
  }
}

export const authController = new AuthController();
