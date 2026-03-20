export interface Session {
  id: string;
  name: string | null;
  createdAt: Date;
  updatedAt: Date;
  messageCount: number;
  lastMessage: string | null;
}

export interface SessionDetail {
  id: string;
  name: string | null;
  createdAt: Date;
  steps: Step[];
}

export interface Step {
  id: string;
  type: string;
  name: string | null;
  output: string | null;
  createdAt: Date;
}
