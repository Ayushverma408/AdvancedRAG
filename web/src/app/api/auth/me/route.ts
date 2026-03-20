import { authController } from "@/controllers/auth.controller";
import type { NextRequest } from "next/server";

export const GET = (req: NextRequest) => authController.me(req);
