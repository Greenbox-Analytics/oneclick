export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "13.0.5"
  }
  public: {
    Tables: {
      artists: {
        Row: {
          additional_epk: string | null
          additional_linktree: string | null
          additional_press_kit: string | null
          avatar_url: string | null
          bio: string | null
          created_at: string
          custom_links: Json | null
          dsp_apple_music: string | null
          dsp_soundcloud: string | null
          dsp_spotify: string | null
          email: string
          genres: string[] | null
          has_contract: boolean | null
          id: string
          name: string
          social_instagram: string | null
          social_tiktok: string | null
          social_youtube: string | null
          updated_at: string
          user_id: string
          custom_social_links: Json | null
          custom_dsp_links: Json | null
          linked_user_id: string | null
          verified: boolean
          verified_at: string | null
        }
        Insert: {
          additional_epk?: string | null
          additional_linktree?: string | null
          additional_press_kit?: string | null
          avatar_url?: string | null
          bio?: string | null
          created_at?: string
          custom_links?: Json | null
          dsp_apple_music?: string | null
          dsp_soundcloud?: string | null
          dsp_spotify?: string | null
          email: string
          genres?: string[] | null
          has_contract?: boolean | null
          id?: string
          name: string
          social_instagram?: string | null
          social_tiktok?: string | null
          social_youtube?: string | null
          updated_at?: string
          user_id?: string
          custom_social_links?: Json | null
          custom_dsp_links?: Json | null
          linked_user_id?: string | null
          verified?: boolean
          verified_at?: string | null
        }
        Update: {
          additional_epk?: string | null
          additional_linktree?: string | null
          additional_press_kit?: string | null
          avatar_url?: string | null
          bio?: string | null
          created_at?: string
          custom_links?: Json | null
          dsp_apple_music?: string | null
          dsp_soundcloud?: string | null
          dsp_spotify?: string | null
          email?: string
          genres?: string[] | null
          has_contract?: boolean | null
          id?: string
          name?: string
          social_instagram?: string | null
          social_tiktok?: string | null
          social_youtube?: string | null
          updated_at?: string
          user_id?: string
          custom_social_links?: Json | null
          custom_dsp_links?: Json | null
          linked_user_id?: string | null
          verified?: boolean
          verified_at?: string | null
        }
        Relationships: []
      }
      profiles: {
        Row: {
          avatar_url: string | null
          company: string | null
          first_name: string | null
          full_name: string | null
          given_name: string | null
          id: string
          industry: string | null
          last_name: string | null
          onboarding_completed: boolean
          phone: string | null
          updated_at: string | null
          username: string | null
          walkthrough_completed: boolean
          website: string | null
        }
        Insert: {
          avatar_url?: string | null
          company?: string | null
          first_name?: string | null
          full_name?: string | null
          given_name?: string | null
          id: string
          industry?: string | null
          last_name?: string | null
          onboarding_completed?: boolean
          phone?: string | null
          updated_at?: string | null
          username?: string | null
          walkthrough_completed?: boolean
          website?: string | null
        }
        Update: {
          avatar_url?: string | null
          company?: string | null
          first_name?: string | null
          full_name?: string | null
          given_name?: string | null
          id?: string
          industry?: string | null
          last_name?: string | null
          onboarding_completed?: boolean
          phone?: string | null
          updated_at?: string | null
          username?: string | null
          walkthrough_completed?: boolean
          website?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "profiles_id_fkey"
            columns: ["id"]
            isOneToOne: true
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
        ]
      }
      user_onboarding: {
        Row: {
          user_id: string
          oneclick_completed: boolean
          zoe_completed: boolean
          splitsheet_completed: boolean
          artists_completed: boolean
          workspace_completed: boolean
          portfolio_completed: boolean
          created_at: string
        }
        Insert: {
          user_id: string
          oneclick_completed?: boolean
          zoe_completed?: boolean
          splitsheet_completed?: boolean
          artists_completed?: boolean
          workspace_completed?: boolean
          portfolio_completed?: boolean
          created_at?: string
        }
        Update: {
          user_id?: string
          oneclick_completed?: boolean
          zoe_completed?: boolean
          splitsheet_completed?: boolean
          artists_completed?: boolean
          workspace_completed?: boolean
          portfolio_completed?: boolean
          created_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "user_onboarding_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: true
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
        ]
      }
      project_files: {
        Row: {
          content_hash: string | null
          created_at: string
          file_name: string
          file_size: number | null
          file_type: string | null
          file_url: string
          folder_category: string
          id: string
          project_id: string
          updated_at: string
        }
        Insert: {
          content_hash?: string | null
          created_at?: string
          file_name: string
          file_size?: number | null
          file_type?: string | null
          file_url: string
          folder_category: string
          id?: string
          project_id: string
          updated_at?: string
        }
        Update: {
          content_hash?: string | null
          created_at?: string
          file_name?: string
          file_size?: number | null
          file_type?: string | null
          file_url?: string
          folder_category?: string
          id?: string
          project_id?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "project_files_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      projects: {
        Row: {
          artist_id: string
          created_at: string
          description: string | null
          id: string
          name: string
          updated_at: string
          about_content: Json
        }
        Insert: {
          artist_id: string
          created_at?: string
          description?: string | null
          id?: string
          name: string
          updated_at?: string
          about_content?: Json
        }
        Update: {
          artist_id?: string
          created_at?: string
          description?: string | null
          id?: string
          name?: string
          updated_at?: string
          about_content?: Json
        }
        Relationships: [
          {
            foreignKeyName: "projects_artist_id_fkey"
            columns: ["artist_id"]
            isOneToOne: false
            referencedRelation: "artists"
            referencedColumns: ["id"]
          },
        ]
      }
      works_registry: {
        Row: {
          id: string; user_id: string; artist_id: string; project_id: string
          title: string; work_type: string; custom_work_type: string | null
          isrc: string | null; iswc: string | null
          upc: string | null; release_date: string | null; status: string
          notes: string | null; created_at: string; updated_at: string
        }
        Insert: {
          id?: string; user_id?: string; artist_id: string; project_id: string
          title: string; work_type?: string; custom_work_type?: string | null
          isrc?: string | null; iswc?: string | null
          upc?: string | null; release_date?: string | null; status?: string
          notes?: string | null; created_at?: string; updated_at?: string
        }
        Update: {
          id?: string; user_id?: string; artist_id?: string; project_id?: string
          title?: string; work_type?: string; custom_work_type?: string | null
          isrc?: string | null; iswc?: string | null
          upc?: string | null; release_date?: string | null; status?: string
          notes?: string | null; created_at?: string; updated_at?: string
        }
        Relationships: []
      }
      ownership_stakes: {
        Row: {
          id: string; work_id: string; user_id: string; stake_type: string
          holder_name: string; holder_role: string; percentage: number
          holder_email: string | null; holder_ipi: string | null
          publisher_or_label: string | null; notes: string | null
          created_at: string; updated_at: string
        }
        Insert: {
          id?: string; work_id: string; user_id?: string; stake_type: string
          holder_name: string; holder_role: string; percentage: number
          holder_email?: string | null; holder_ipi?: string | null
          publisher_or_label?: string | null; notes?: string | null
          created_at?: string; updated_at?: string
        }
        Update: {
          id?: string; work_id?: string; user_id?: string; stake_type?: string
          holder_name?: string; holder_role?: string; percentage?: number
          holder_email?: string | null; holder_ipi?: string | null
          publisher_or_label?: string | null; notes?: string | null
          created_at?: string; updated_at?: string
        }
        Relationships: []
      }
      licensing_rights: {
        Row: {
          id: string; work_id: string; user_id: string; license_type: string
          licensee_name: string; licensee_email: string | null; territory: string
          start_date: string; end_date: string | null; terms: string | null
          status: string; created_at: string; updated_at: string
        }
        Insert: {
          id?: string; work_id: string; user_id?: string; license_type: string
          licensee_name: string; licensee_email?: string | null; territory?: string
          start_date: string; end_date?: string | null; terms?: string | null
          status?: string; created_at?: string; updated_at?: string
        }
        Update: {
          id?: string; work_id?: string; user_id?: string; license_type?: string
          licensee_name?: string; licensee_email?: string | null; territory?: string
          start_date?: string; end_date?: string | null; terms?: string | null
          status?: string; created_at?: string; updated_at?: string
        }
        Relationships: []
      }
      registry_agreements: {
        Row: {
          id: string; work_id: string; user_id: string; agreement_type: string
          title: string; description: string | null; effective_date: string
          parties: Json; file_id: string | null; document_hash: string | null
          created_at: string
        }
        Insert: {
          id?: string; work_id: string; user_id?: string; agreement_type: string
          title: string; description?: string | null; effective_date: string
          parties?: Json; file_id?: string | null; document_hash?: string | null
          created_at?: string
        }
        Update: {
          id?: string; work_id?: string; user_id?: string; agreement_type?: string
          title?: string; description?: string | null; effective_date?: string
          parties?: Json; file_id?: string | null; document_hash?: string | null
          created_at?: string
        }
        Relationships: []
      }
      registry_collaborators: {
        Row: {
          id: string; work_id: string; stake_id: string | null; invited_by: string
          collaborator_user_id: string | null; email: string; name: string; role: string
          status: string; invite_token: string
          expires_at: string; invited_at: string; responded_at: string | null
        }
        Insert: {
          id?: string; work_id: string; stake_id?: string | null; invited_by?: string
          collaborator_user_id?: string | null; email: string; name: string; role: string
          status?: string; invite_token?: string
          expires_at?: string; invited_at?: string; responded_at?: string | null
        }
        Update: {
          id?: string; work_id?: string; stake_id?: string | null; invited_by?: string
          collaborator_user_id?: string | null; email?: string; name?: string; role?: string
          status?: string; invite_token?: string
          expires_at?: string; invited_at?: string; responded_at?: string | null
        }
        Relationships: []
      }
      team_cards: {
        Row: {
          id: string; user_id: string; display_name: string; first_name: string
          last_name: string; email: string; avatar_url: string | null; bio: string | null
          phone: string | null; website: string | null; company: string | null
          industry: string | null; social_links: Json; dsp_links: Json; custom_links: Json
          visible_fields: Json; created_at: string; updated_at: string
        }
        Insert: {
          id?: string; user_id: string; display_name: string; first_name: string
          last_name: string; email: string; avatar_url?: string | null; bio?: string | null
          phone?: string | null; website?: string | null; company?: string | null
          industry?: string | null; social_links?: Json; dsp_links?: Json; custom_links?: Json
          visible_fields?: Json; created_at?: string; updated_at?: string
        }
        Update: {
          id?: string; user_id?: string; display_name?: string; first_name?: string
          last_name?: string; email?: string; avatar_url?: string | null; bio?: string | null
          phone?: string | null; website?: string | null; company?: string | null
          industry?: string | null; social_links?: Json; dsp_links?: Json; custom_links?: Json
          visible_fields?: Json; created_at?: string; updated_at?: string
        }
        Relationships: []
      }
      notes: {
        Row: {
          id: string; user_id: string; folder_id: string | null
          artist_id: string | null; project_id: string | null
          title: string; content: Json; pinned: boolean
          created_at: string; updated_at: string
        }
        Insert: {
          id?: string; user_id?: string; folder_id?: string | null
          artist_id?: string | null; project_id?: string | null
          title?: string; content?: Json; pinned?: boolean
          created_at?: string; updated_at?: string
        }
        Update: {
          id?: string; user_id?: string; folder_id?: string | null
          artist_id?: string | null; project_id?: string | null
          title?: string; content?: Json; pinned?: boolean
          created_at?: string; updated_at?: string
        }
        Relationships: []
      }
      note_folders: {
        Row: {
          id: string; user_id: string; artist_id: string | null; project_id: string | null
          name: string; parent_folder_id: string | null; sort_order: number
          created_at: string; updated_at: string
        }
        Insert: {
          id?: string; user_id?: string; artist_id?: string | null; project_id?: string | null
          name: string; parent_folder_id?: string | null; sort_order?: number
          created_at?: string; updated_at?: string
        }
        Update: {
          id?: string; user_id?: string; artist_id?: string | null; project_id?: string | null
          name?: string; parent_folder_id?: string | null; sort_order?: number
          created_at?: string; updated_at?: string
        }
        Relationships: []
      }
      project_members: {
        Row: {
          id: string
          project_id: string
          user_id: string
          role: string
          invited_by: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          project_id: string
          user_id: string
          role: string
          invited_by?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          project_id?: string
          user_id?: string
          role?: string
          invited_by?: string | null
          created_at?: string
          updated_at?: string
        }
        Relationships: []
      }
      pending_project_invites: {
        Row: {
          id: string
          project_id: string
          email: string
          role: string
          invited_by: string
          created_at: string
          expires_at: string
        }
        Insert: {
          id?: string
          project_id: string
          email: string
          role: string
          invited_by: string
          created_at?: string
          expires_at?: string
        }
        Update: {
          id?: string
          project_id?: string
          email?: string
          role?: string
          invited_by?: string
          created_at?: string
          expires_at?: string
        }
        Relationships: []
      }
      work_files: {
        Row: {
          id: string
          work_id: string
          file_id: string
          created_at: string
        }
        Insert: {
          id?: string
          work_id: string
          file_id: string
          created_at?: string
        }
        Update: {
          id?: string
          work_id?: string
          file_id?: string
          created_at?: string
        }
        Relationships: []
      }
      work_audio_links: {
        Row: {
          id: string
          work_id: string
          audio_file_id: string
          created_at: string
        }
        Insert: {
          id?: string
          work_id: string
          audio_file_id: string
          created_at?: string
        }
        Update: {
          id?: string
          work_id?: string
          audio_file_id?: string
          created_at?: string
        }
        Relationships: []
      }
      registry_notifications: {
        Row: {
          id: string; user_id: string; work_id: string | null; type: string
          title: string; message: string; read: boolean; metadata: Json; created_at: string
        }
        Insert: {
          id?: string; user_id?: string; work_id?: string | null; type: string
          title: string; message: string; read?: boolean; metadata?: Json; created_at?: string
        }
        Update: {
          id?: string; user_id?: string; work_id?: string | null; type?: string
          title?: string; message?: string; read?: boolean; metadata?: Json; created_at?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {},
  },
} as const
