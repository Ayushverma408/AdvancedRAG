import { prisma } from "@/lib/prisma";

export class SessionService {
  async listByUser(userId: string) {
    const threads = await prisma.chainlitThread.findMany({
      where: { userId },
      include: {
        steps: {
          where: { type: { in: ["user_message", "assistant_message"] } },
          orderBy: { createdAt: "desc" },
          take: 1,
        },
        _count: {
          select: {
            steps: { where: { type: { in: ["user_message", "assistant_message"] } } },
          },
        },
      },
      orderBy: { updatedAt: "desc" },
    });

    return threads.map((t) => ({
      id: t.id,
      name: t.name,
      createdAt: t.createdAt,
      updatedAt: t.updatedAt,
      messageCount: t._count.steps,
      lastMessage: t.steps[0]?.output ?? null,
    }));
  }

  async getById(threadId: string, userId: string) {
    const thread = await prisma.chainlitThread.findFirst({
      where: { id: threadId, userId },
      include: {
        steps: {
          where: { type: { in: ["user_message", "assistant_message"] } },
          orderBy: { createdAt: "asc" },
        },
      },
    });
    if (!thread) throw Object.assign(new Error("Session not found"), { code: "NOT_FOUND" });
    return thread;
  }
}

export const sessionService = new SessionService();
