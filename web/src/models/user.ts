export interface User {
  id: string;
  email: string;
  name: string;
  createdAt: Date;
}

export interface SignupInput {
  name: string;
  email: string;
  password: string;
}

export interface LoginInput {
  email: string;
  password: string;
}
