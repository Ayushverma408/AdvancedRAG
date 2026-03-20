import { authController } from "@/controllers/auth.controller";
import type { NextRequest } from "next/server";

export const POST = (req: NextRequest) => authController.signup(req);
