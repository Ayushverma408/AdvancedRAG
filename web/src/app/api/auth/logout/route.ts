import { authController } from "@/controllers/auth.controller";

export const POST = () => authController.logout();
