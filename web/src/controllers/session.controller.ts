import { NextRequest, NextResponse } from "next/server";
import { sessionService } from "@/services/session.service";

export class SessionController {
  async list(req: NextRequest) {
    try {
      const userId = req.headers.get("x-user-id")!;
      const sessions = await sessionService.listByUser(userId);
      return NextResponse.json({ sessions });
    } catch {
      return NextResponse.json({ error: "Internal server error" }, { status: 500 });
    }
  }

  async get(req: NextRequest, id: string) {
    try {
      const userId = req.headers.get("x-user-id")!;
      const session = await sessionService.getById(id, userId);
      return NextResponse.json({ session });
    } catch (e: unknown) {
      const err = e as Error & { code?: string };
      if (err.code === "NOT_FOUND") {
        return NextResponse.json({ error: "Session not found" }, { status: 404 });
      }
      return NextResponse.json({ error: "Internal server error" }, { status: 500 });
    }
  }
}

export const sessionController = new SessionController();
