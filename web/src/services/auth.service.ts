import bcrypt from "bcryptjs";
import { prisma } from "@/lib/prisma";
import { signToken } from "@/lib/jwt";
import type { SignupInput, LoginInput } from "@/models/user";

export class AuthService {
  async signup(input: SignupInput) {
    const exists = await prisma.user.findUnique({ where: { email: input.email } });
    if (exists) throw Object.assign(new Error("Email already registered"), { code: "EMAIL_EXISTS" });

    const passwordHash = await bcrypt.hash(input.password, 12);
    const user = await prisma.user.create({
      data: { email: input.email, name: input.name, passwordHash },
    });

    const token = await signToken({ userId: user.id, email: user.email, name: user.name });
    return { user: { id: user.id, email: user.email, name: user.name }, token };
  }

  async login(input: LoginInput) {
    const user = await prisma.user.findUnique({ where: { email: input.email } });
    if (!user) throw Object.assign(new Error("Invalid credentials"), { code: "INVALID_CREDS" });

    const valid = await bcrypt.compare(input.password, user.passwordHash);
    if (!valid) throw Object.assign(new Error("Invalid credentials"), { code: "INVALID_CREDS" });

    const token = await signToken({ userId: user.id, email: user.email, name: user.name });
    return { user: { id: user.id, email: user.email, name: user.name }, token };
  }
}

export const authService = new AuthService();
