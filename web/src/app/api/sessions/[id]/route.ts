import { sessionController } from "@/controllers/session.controller";
import type { NextRequest } from "next/server";

export const GET = (req: NextRequest, { params }: { params: { id: string } }) =>
  sessionController.get(req, params.id);
