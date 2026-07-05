export interface Team {
  id: string;
  name: string;
  description?: string | null;
  created_by?: string | null;
  archived_at?: string | null;
  created_at?: string;
  my_role?: "admin" | "member";
  boards?: number;
  tasks?: number;
  members?: number;
}

export interface TeamMember {
  id: string;
  team_id: string;
  user_id: string;
  role: "admin" | "member";
  full_name?: string | null;
  avatar_url?: string | null;
  created_at?: string;
}

export interface TeamInvite {
  id: string;
  team_id: string;
  email: string;
  role: "admin" | "member";
  status: "pending" | "accepted" | "declined";
  expires_at?: string;
  created_at?: string;
}

export interface Board {
  id: string;
  team_id?: string | null;
  owner_id?: string;
  artist_id?: string | null;
  name: string;
  description?: string | null;
  archived?: boolean;
  position?: number;
  task_count?: number;
}
