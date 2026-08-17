export interface Board {
  id: string;
  /** Owning organization id (same edge as artists.team_id); null = personal board. */
  team_id?: string | null;
  owner_id?: string;
  artist_id?: string | null;
  name: string;
  description?: string | null;
  archived?: boolean;
  position?: number;
  task_count?: number;
  /** Team boards only: when true, only the owner, org admins and member_user_ids can see it. */
  restricted?: boolean;
  member_user_ids?: string[];
}
