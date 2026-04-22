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
      artist_credentials: {
        Row: {
          artist_id: string
          created_at: string
          id: string
          login_identifier: string
          notes: string | null
          password_ciphertext: string
          platform_name: string
          updated_at: string
          url: string | null
          user_id: string
        }
        Insert: {
          artist_id: string
          created_at?: string
          id?: string
          login_identifier: string
          notes?: string | null
          password_ciphertext: string
          platform_name: string
          updated_at?: string
          url?: string | null
          user_id: string
        }
        Update: {
          artist_id?: string
          created_at?: string
          id?: string
          login_identifier?: string
          notes?: string | null
          password_ciphertext?: string
          platform_name?: string
          updated_at?: string
          url?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "artist_credentials_artist_id_fkey"
            columns: ["artist_id"]
            isOneToOne: false
            referencedRelation: "artists"
            referencedColumns: ["id"]
          },
        ]
      }
      artists: {
        Row: {
          additional_linktree: string | null
          additional_press_kit: string | null
          additional_website: string | null
          avatar_url: string | null
          bio: string | null
          created_at: string
          custom_dsp_links: Json | null
          custom_links: Json | null
          custom_social_links: Json | null
          dsp_apple_music: string | null
          dsp_soundcloud: string | null
          dsp_spotify: string | null
          email: string
          genres: string[] | null
          has_contract: boolean | null
          id: string
          linked_user_id: string | null
          name: string
          social_instagram: string | null
          social_tiktok: string | null
          social_youtube: string | null
          updated_at: string
          user_id: string | null
          verified: boolean
          verified_at: string | null
        }
        Insert: {
          additional_linktree?: string | null
          additional_press_kit?: string | null
          additional_website?: string | null
          avatar_url?: string | null
          bio?: string | null
          created_at?: string
          custom_dsp_links?: Json | null
          custom_links?: Json | null
          custom_social_links?: Json | null
          dsp_apple_music?: string | null
          dsp_soundcloud?: string | null
          dsp_spotify?: string | null
          email: string
          genres?: string[] | null
          has_contract?: boolean | null
          id?: string
          linked_user_id?: string | null
          name: string
          social_instagram?: string | null
          social_tiktok?: string | null
          social_youtube?: string | null
          updated_at?: string
          user_id?: string | null
          verified?: boolean
          verified_at?: string | null
        }
        Update: {
          additional_linktree?: string | null
          additional_press_kit?: string | null
          additional_website?: string | null
          avatar_url?: string | null
          bio?: string | null
          created_at?: string
          custom_dsp_links?: Json | null
          custom_links?: Json | null
          custom_social_links?: Json | null
          dsp_apple_music?: string | null
          dsp_soundcloud?: string | null
          dsp_spotify?: string | null
          email?: string
          genres?: string[] | null
          has_contract?: boolean | null
          id?: string
          linked_user_id?: string | null
          name?: string
          social_instagram?: string | null
          social_tiktok?: string | null
          social_youtube?: string | null
          updated_at?: string
          user_id?: string | null
          verified?: boolean
          verified_at?: string | null
        }
        Relationships: []
      }
      audio_files: {
        Row: {
          content_hash: string | null
          created_at: string
          file_name: string
          file_path: string
          file_size: number | null
          file_type: string | null
          file_url: string
          folder_id: string
          id: string
        }
        Insert: {
          content_hash?: string | null
          created_at?: string
          file_name: string
          file_path: string
          file_size?: number | null
          file_type?: string | null
          file_url: string
          folder_id: string
          id?: string
        }
        Update: {
          content_hash?: string | null
          created_at?: string
          file_name?: string
          file_path?: string
          file_size?: number | null
          file_type?: string | null
          file_url?: string
          folder_id?: string
          id?: string
        }
        Relationships: [
          {
            foreignKeyName: "audio_files_folder_id_fkey"
            columns: ["folder_id"]
            isOneToOne: false
            referencedRelation: "audio_folders"
            referencedColumns: ["id"]
          },
        ]
      }
      audio_folders: {
        Row: {
          artist_id: string
          created_at: string
          id: string
          name: string
          parent_id: string | null
        }
        Insert: {
          artist_id: string
          created_at?: string
          id?: string
          name: string
          parent_id?: string | null
        }
        Update: {
          artist_id?: string
          created_at?: string
          id?: string
          name?: string
          parent_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "audio_folders_artist_id_fkey"
            columns: ["artist_id"]
            isOneToOne: false
            referencedRelation: "artists"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "audio_folders_parent_id_fkey"
            columns: ["parent_id"]
            isOneToOne: false
            referencedRelation: "audio_folders"
            referencedColumns: ["id"]
          },
        ]
      }
      board_columns: {
        Row: {
          artist_id: string | null
          color: string | null
          created_at: string | null
          id: string
          position: number
          title: string
          updated_at: string | null
          user_id: string
        }
        Insert: {
          artist_id?: string | null
          color?: string | null
          created_at?: string | null
          id?: string
          position?: number
          title: string
          updated_at?: string | null
          user_id: string
        }
        Update: {
          artist_id?: string | null
          color?: string | null
          created_at?: string | null
          id?: string
          position?: number
          title?: string
          updated_at?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "board_columns_artist_id_fkey"
            columns: ["artist_id"]
            isOneToOne: false
            referencedRelation: "artists"
            referencedColumns: ["id"]
          },
        ]
      }
      board_task_artists: {
        Row: {
          artist_id: string
          created_at: string | null
          id: string
          task_id: string
        }
        Insert: {
          artist_id: string
          created_at?: string | null
          id?: string
          task_id: string
        }
        Update: {
          artist_id?: string
          created_at?: string | null
          id?: string
          task_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "board_task_artists_artist_id_fkey"
            columns: ["artist_id"]
            isOneToOne: false
            referencedRelation: "artists"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "board_task_artists_task_id_fkey"
            columns: ["task_id"]
            isOneToOne: false
            referencedRelation: "board_tasks"
            referencedColumns: ["id"]
          },
        ]
      }
      board_task_comments: {
        Row: {
          content: string
          created_at: string | null
          id: string
          task_id: string
          updated_at: string | null
          user_id: string
        }
        Insert: {
          content: string
          created_at?: string | null
          id?: string
          task_id: string
          updated_at?: string | null
          user_id: string
        }
        Update: {
          content?: string
          created_at?: string | null
          id?: string
          task_id?: string
          updated_at?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "board_task_comments_task_id_fkey"
            columns: ["task_id"]
            isOneToOne: false
            referencedRelation: "board_tasks"
            referencedColumns: ["id"]
          },
        ]
      }
      board_task_contracts: {
        Row: {
          created_at: string | null
          id: string
          project_file_id: string
          task_id: string
        }
        Insert: {
          created_at?: string | null
          id?: string
          project_file_id: string
          task_id: string
        }
        Update: {
          created_at?: string | null
          id?: string
          project_file_id?: string
          task_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "board_task_contracts_project_file_id_fkey"
            columns: ["project_file_id"]
            isOneToOne: false
            referencedRelation: "project_files"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "board_task_contracts_task_id_fkey"
            columns: ["task_id"]
            isOneToOne: false
            referencedRelation: "board_tasks"
            referencedColumns: ["id"]
          },
        ]
      }
      board_task_projects: {
        Row: {
          created_at: string | null
          id: string
          project_id: string
          task_id: string
        }
        Insert: {
          created_at?: string | null
          id?: string
          project_id: string
          task_id: string
        }
        Update: {
          created_at?: string | null
          id?: string
          project_id?: string
          task_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "board_task_projects_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "board_task_projects_task_id_fkey"
            columns: ["task_id"]
            isOneToOne: false
            referencedRelation: "board_tasks"
            referencedColumns: ["id"]
          },
        ]
      }
      board_tasks: {
        Row: {
          artist_id: string | null
          assignee_name: string | null
          color: string | null
          column_id: string | null
          completed_at: string | null
          created_at: string | null
          description: string | null
          due_date: string | null
          external_id: string | null
          external_provider: string | null
          external_url: string | null
          id: string
          is_parent: boolean | null
          labels: string[] | null
          last_synced_at: string | null
          parent_task_id: string | null
          position: number
          priority: string | null
          project_id: string | null
          start_date: string | null
          sync_hash: string | null
          title: string
          updated_at: string | null
          user_id: string
        }
        Insert: {
          artist_id?: string | null
          assignee_name?: string | null
          color?: string | null
          column_id?: string | null
          completed_at?: string | null
          created_at?: string | null
          description?: string | null
          due_date?: string | null
          external_id?: string | null
          external_provider?: string | null
          external_url?: string | null
          id?: string
          is_parent?: boolean | null
          labels?: string[] | null
          last_synced_at?: string | null
          parent_task_id?: string | null
          position?: number
          priority?: string | null
          project_id?: string | null
          start_date?: string | null
          sync_hash?: string | null
          title: string
          updated_at?: string | null
          user_id: string
        }
        Update: {
          artist_id?: string | null
          assignee_name?: string | null
          color?: string | null
          column_id?: string | null
          completed_at?: string | null
          created_at?: string | null
          description?: string | null
          due_date?: string | null
          external_id?: string | null
          external_provider?: string | null
          external_url?: string | null
          id?: string
          is_parent?: boolean | null
          labels?: string[] | null
          last_synced_at?: string | null
          parent_task_id?: string | null
          position?: number
          priority?: string | null
          project_id?: string | null
          start_date?: string | null
          sync_hash?: string | null
          title?: string
          updated_at?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "board_tasks_artist_id_fkey"
            columns: ["artist_id"]
            isOneToOne: false
            referencedRelation: "artists"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "board_tasks_column_id_fkey"
            columns: ["column_id"]
            isOneToOne: false
            referencedRelation: "board_columns"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "board_tasks_parent_task_id_fkey"
            columns: ["parent_task_id"]
            isOneToOne: false
            referencedRelation: "board_tasks"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "board_tasks_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      contacts: {
        Row: {
          created_at: string
          email: string | null
          id: string
          name: string
          notes: string | null
          phone: string | null
          role: string | null
          stripe_account_id: string | null
          updated_at: string
          user_id: string
        }
        Insert: {
          created_at?: string
          email?: string | null
          id?: string
          name: string
          notes?: string | null
          phone?: string | null
          role?: string | null
          stripe_account_id?: string | null
          updated_at?: string
          user_id: string
        }
        Update: {
          created_at?: string
          email?: string | null
          id?: string
          name?: string
          notes?: string | null
          phone?: string | null
          role?: string | null
          stripe_account_id?: string | null
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      drive_sync_mappings: {
        Row: {
          created_at: string | null
          drive_file_id: string
          drive_folder_id: string | null
          drive_modified_at: string | null
          id: string
          last_synced_at: string | null
          local_modified_at: string | null
          project_file_id: string | null
          project_id: string | null
          sync_direction: string
          user_id: string
        }
        Insert: {
          created_at?: string | null
          drive_file_id: string
          drive_folder_id?: string | null
          drive_modified_at?: string | null
          id?: string
          last_synced_at?: string | null
          local_modified_at?: string | null
          project_file_id?: string | null
          project_id?: string | null
          sync_direction: string
          user_id: string
        }
        Update: {
          created_at?: string | null
          drive_file_id?: string
          drive_folder_id?: string | null
          drive_modified_at?: string | null
          id?: string
          last_synced_at?: string | null
          local_modified_at?: string | null
          project_file_id?: string | null
          project_id?: string | null
          sync_direction?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "drive_sync_mappings_project_file_id_fkey"
            columns: ["project_file_id"]
            isOneToOne: false
            referencedRelation: "project_files"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "drive_sync_mappings_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      file_shares: {
        Row: {
          contact_id: string | null
          file_id: string
          file_name: string
          file_source: string
          id: string
          link_expires_at: string
          message: string | null
          recipient_email: string
          recipient_name: string | null
          shared_at: string
          status: string
          user_id: string
        }
        Insert: {
          contact_id?: string | null
          file_id: string
          file_name: string
          file_source: string
          id?: string
          link_expires_at: string
          message?: string | null
          recipient_email: string
          recipient_name?: string | null
          shared_at?: string
          status?: string
          user_id: string
        }
        Update: {
          contact_id?: string | null
          file_id?: string
          file_name?: string
          file_source?: string
          id?: string
          link_expires_at?: string
          message?: string | null
          recipient_email?: string
          recipient_name?: string | null
          shared_at?: string
          status?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "file_shares_contact_id_fkey"
            columns: ["contact_id"]
            isOneToOne: false
            referencedRelation: "contacts"
            referencedColumns: ["id"]
          },
        ]
      }
      integration_connections: {
        Row: {
          access_token_encrypted: string
          created_at: string | null
          id: string
          provider: string
          provider_metadata: Json | null
          provider_user_id: string | null
          provider_workspace_id: string | null
          refresh_token_encrypted: string | null
          scopes: string[] | null
          status: string
          token_expires_at: string | null
          updated_at: string | null
          user_id: string
        }
        Insert: {
          access_token_encrypted: string
          created_at?: string | null
          id?: string
          provider: string
          provider_metadata?: Json | null
          provider_user_id?: string | null
          provider_workspace_id?: string | null
          refresh_token_encrypted?: string | null
          scopes?: string[] | null
          status?: string
          token_expires_at?: string | null
          updated_at?: string | null
          user_id: string
        }
        Update: {
          access_token_encrypted?: string
          created_at?: string | null
          id?: string
          provider?: string
          provider_metadata?: Json | null
          provider_user_id?: string | null
          provider_workspace_id?: string | null
          refresh_token_encrypted?: string | null
          scopes?: string[] | null
          status?: string
          token_expires_at?: string | null
          updated_at?: string | null
          user_id?: string
        }
        Relationships: []
      }
      licensing_rights: {
        Row: {
          created_at: string
          end_date: string | null
          id: string
          license_type: string
          licensee_email: string | null
          licensee_name: string
          start_date: string
          status: string
          terms: string | null
          territory: string | null
          updated_at: string
          user_id: string
          work_id: string
        }
        Insert: {
          created_at?: string
          end_date?: string | null
          id?: string
          license_type: string
          licensee_email?: string | null
          licensee_name: string
          start_date: string
          status?: string
          terms?: string | null
          territory?: string | null
          updated_at?: string
          user_id: string
          work_id: string
        }
        Update: {
          created_at?: string
          end_date?: string | null
          id?: string
          license_type?: string
          licensee_email?: string | null
          licensee_name?: string
          start_date?: string
          status?: string
          terms?: string | null
          territory?: string | null
          updated_at?: string
          user_id?: string
          work_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "licensing_rights_work_id_fkey"
            columns: ["work_id"]
            isOneToOne: false
            referencedRelation: "works_registry"
            referencedColumns: ["id"]
          },
        ]
      }
      mailing_list: {
        Row: {
          created_at: string | null
          email: string
          id: string
          name: string | null
        }
        Insert: {
          created_at?: string | null
          email: string
          id?: string
          name?: string | null
        }
        Update: {
          created_at?: string | null
          email?: string
          id?: string
          name?: string | null
        }
        Relationships: []
      }
      note_folders: {
        Row: {
          artist_id: string | null
          created_at: string
          id: string
          name: string
          parent_folder_id: string | null
          project_id: string | null
          sort_order: number
          updated_at: string
          user_id: string
        }
        Insert: {
          artist_id?: string | null
          created_at?: string
          id?: string
          name: string
          parent_folder_id?: string | null
          project_id?: string | null
          sort_order?: number
          updated_at?: string
          user_id: string
        }
        Update: {
          artist_id?: string | null
          created_at?: string
          id?: string
          name?: string
          parent_folder_id?: string | null
          project_id?: string | null
          sort_order?: number
          updated_at?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "note_folders_artist_id_fkey"
            columns: ["artist_id"]
            isOneToOne: false
            referencedRelation: "artists"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "note_folders_parent_folder_id_fkey"
            columns: ["parent_folder_id"]
            isOneToOne: false
            referencedRelation: "note_folders"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "note_folders_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      notes: {
        Row: {
          artist_id: string | null
          content: Json
          created_at: string
          folder_id: string | null
          id: string
          pinned: boolean
          project_id: string | null
          title: string
          updated_at: string
          user_id: string
        }
        Insert: {
          artist_id?: string | null
          content?: Json
          created_at?: string
          folder_id?: string | null
          id?: string
          pinned?: boolean
          project_id?: string | null
          title?: string
          updated_at?: string
          user_id: string
        }
        Update: {
          artist_id?: string | null
          content?: Json
          created_at?: string
          folder_id?: string | null
          id?: string
          pinned?: boolean
          project_id?: string | null
          title?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "notes_artist_id_fkey"
            columns: ["artist_id"]
            isOneToOne: false
            referencedRelation: "artists"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "notes_folder_id_fkey"
            columns: ["folder_id"]
            isOneToOne: false
            referencedRelation: "note_folders"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "notes_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      notification_settings: {
        Row: {
          channel_id: string | null
          created_at: string | null
          enabled: boolean | null
          event_type: string
          id: string
          provider: string
          user_id: string
        }
        Insert: {
          channel_id?: string | null
          created_at?: string | null
          enabled?: boolean | null
          event_type: string
          id?: string
          provider: string
          user_id: string
        }
        Update: {
          channel_id?: string | null
          created_at?: string | null
          enabled?: boolean | null
          event_type?: string
          id?: string
          provider?: string
          user_id?: string
        }
        Relationships: []
      }
      ownership_stakes: {
        Row: {
          created_at: string
          holder_email: string | null
          holder_ipi: string | null
          holder_name: string
          holder_role: string
          id: string
          notes: string | null
          percentage: number
          publisher_or_label: string | null
          stake_type: string
          updated_at: string
          user_id: string
          work_id: string
        }
        Insert: {
          created_at?: string
          holder_email?: string | null
          holder_ipi?: string | null
          holder_name: string
          holder_role: string
          id?: string
          notes?: string | null
          percentage: number
          publisher_or_label?: string | null
          stake_type: string
          updated_at?: string
          user_id: string
          work_id: string
        }
        Update: {
          created_at?: string
          holder_email?: string | null
          holder_ipi?: string | null
          holder_name?: string
          holder_role?: string
          id?: string
          notes?: string | null
          percentage?: number
          publisher_or_label?: string | null
          stake_type?: string
          updated_at?: string
          user_id?: string
          work_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "ownership_stakes_work_id_fkey"
            columns: ["work_id"]
            isOneToOne: false
            referencedRelation: "works_registry"
            referencedColumns: ["id"]
          },
        ]
      }
      payments: {
        Row: {
          amount: number
          contact_id: string | null
          created_at: string
          currency: string
          id: string
          metadata: Json | null
          party_name: string
          status: string
          stripe_payment_intent_id: string | null
          stripe_transfer_id: string | null
          updated_at: string
          user_id: string
        }
        Insert: {
          amount: number
          contact_id?: string | null
          created_at?: string
          currency?: string
          id?: string
          metadata?: Json | null
          party_name: string
          status?: string
          stripe_payment_intent_id?: string | null
          stripe_transfer_id?: string | null
          updated_at?: string
          user_id: string
        }
        Update: {
          amount?: number
          contact_id?: string | null
          created_at?: string
          currency?: string
          id?: string
          metadata?: Json | null
          party_name?: string
          status?: string
          stripe_payment_intent_id?: string | null
          stripe_transfer_id?: string | null
          updated_at?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "payments_contact_id_fkey"
            columns: ["contact_id"]
            isOneToOne: false
            referencedRelation: "contacts"
            referencedColumns: ["id"]
          },
        ]
      }
      pending_project_invites: {
        Row: {
          created_at: string | null
          email: string
          expires_at: string | null
          id: string
          invited_by: string
          last_email_attempt_at: string | null
          last_email_error: string | null
          project_id: string
          role: string
        }
        Insert: {
          created_at?: string | null
          email: string
          expires_at?: string | null
          id?: string
          invited_by: string
          last_email_attempt_at?: string | null
          last_email_error?: string | null
          project_id: string
          role: string
        }
        Update: {
          created_at?: string | null
          email?: string
          expires_at?: string | null
          id?: string
          invited_by?: string
          last_email_attempt_at?: string | null
          last_email_error?: string | null
          project_id?: string
          role?: string
        }
        Relationships: [
          {
            foreignKeyName: "pending_project_invites_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      profiles: {
        Row: {
          avatar_url: string | null
          company: string | null
          first_name: string | null
          full_name: string | null
          given_name: string | null
          id: string
          last_name: string | null
          onboarding_completed: boolean | null
          phone: string | null
          role: string | null
          updated_at: string | null
          walkthrough_completed: boolean | null
          website: string | null
        }
        Insert: {
          avatar_url?: string | null
          company?: string | null
          first_name?: string | null
          full_name?: string | null
          given_name?: string | null
          id: string
          last_name?: string | null
          onboarding_completed?: boolean | null
          phone?: string | null
          role?: string | null
          updated_at?: string | null
          walkthrough_completed?: boolean | null
          website?: string | null
        }
        Update: {
          avatar_url?: string | null
          company?: string | null
          first_name?: string | null
          full_name?: string | null
          given_name?: string | null
          id?: string
          last_name?: string | null
          onboarding_completed?: boolean | null
          phone?: string | null
          role?: string | null
          updated_at?: string | null
          walkthrough_completed?: boolean | null
          website?: string | null
        }
        Relationships: []
      }
      project_audio_links: {
        Row: {
          audio_file_id: string
          created_at: string
          project_id: string
        }
        Insert: {
          audio_file_id: string
          created_at?: string
          project_id: string
        }
        Update: {
          audio_file_id?: string
          created_at?: string
          project_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "project_audio_links_audio_file_id_fkey"
            columns: ["audio_file_id"]
            isOneToOne: false
            referencedRelation: "audio_files"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "project_audio_links_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      project_files: {
        Row: {
          content_hash: string | null
          contract_markdown: string | null
          created_at: string
          file_name: string
          file_path: string | null
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
          contract_markdown?: string | null
          created_at?: string
          file_name: string
          file_path?: string | null
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
          contract_markdown?: string | null
          created_at?: string
          file_name?: string
          file_path?: string | null
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
      project_members: {
        Row: {
          created_at: string | null
          id: string
          invited_by: string | null
          project_id: string
          role: string
          updated_at: string | null
          user_id: string
        }
        Insert: {
          created_at?: string | null
          id?: string
          invited_by?: string | null
          project_id: string
          role: string
          updated_at?: string | null
          user_id: string
        }
        Update: {
          created_at?: string | null
          id?: string
          invited_by?: string | null
          project_id?: string
          role?: string
          updated_at?: string | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "project_members_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      project_notification_settings: {
        Row: {
          created_at: string | null
          enabled: boolean | null
          event_type: string
          id: string
          project_id: string
        }
        Insert: {
          created_at?: string | null
          enabled?: boolean | null
          event_type: string
          id?: string
          project_id: string
        }
        Update: {
          created_at?: string | null
          enabled?: boolean | null
          event_type?: string
          id?: string
          project_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "project_notification_settings_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      projects: {
        Row: {
          about_content: Json
          artist_id: string
          created_at: string
          description: string | null
          drive_folder_id: string | null
          id: string
          name: string
          slack_channel_id: string | null
          updated_at: string
        }
        Insert: {
          about_content?: Json
          artist_id: string
          created_at?: string
          description?: string | null
          drive_folder_id?: string | null
          id?: string
          name: string
          slack_channel_id?: string | null
          updated_at?: string
        }
        Update: {
          about_content?: Json
          artist_id?: string
          created_at?: string
          description?: string | null
          drive_folder_id?: string | null
          id?: string
          name?: string
          slack_channel_id?: string | null
          updated_at?: string
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
      registry_agreements: {
        Row: {
          agreement_type: string
          created_at: string
          description: string | null
          document_hash: string | null
          effective_date: string
          file_id: string | null
          id: string
          parties: Json
          title: string
          user_id: string
          work_id: string
        }
        Insert: {
          agreement_type: string
          created_at?: string
          description?: string | null
          document_hash?: string | null
          effective_date: string
          file_id?: string | null
          id?: string
          parties?: Json
          title: string
          user_id: string
          work_id: string
        }
        Update: {
          agreement_type?: string
          created_at?: string
          description?: string | null
          document_hash?: string | null
          effective_date?: string
          file_id?: string | null
          id?: string
          parties?: Json
          title?: string
          user_id?: string
          work_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "registry_agreements_file_id_fkey"
            columns: ["file_id"]
            isOneToOne: false
            referencedRelation: "project_files"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "registry_agreements_work_id_fkey"
            columns: ["work_id"]
            isOneToOne: false
            referencedRelation: "works_registry"
            referencedColumns: ["id"]
          },
        ]
      }
      registry_collaborators: {
        Row: {
          collaborator_user_id: string | null
          email: string
          expires_at: string
          id: string
          invite_token: string
          invited_at: string
          invited_by: string
          name: string
          responded_at: string | null
          role: string
          stake_id: string | null
          status: string
          work_id: string
        }
        Insert: {
          collaborator_user_id?: string | null
          email: string
          expires_at?: string
          id?: string
          invite_token?: string
          invited_at?: string
          invited_by: string
          name: string
          responded_at?: string | null
          role: string
          stake_id?: string | null
          status?: string
          work_id: string
        }
        Update: {
          collaborator_user_id?: string | null
          email?: string
          expires_at?: string
          id?: string
          invite_token?: string
          invited_at?: string
          invited_by?: string
          name?: string
          responded_at?: string | null
          role?: string
          stake_id?: string | null
          status?: string
          work_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "registry_collaborators_stake_id_fkey"
            columns: ["stake_id"]
            isOneToOne: false
            referencedRelation: "ownership_stakes"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "registry_collaborators_work_id_fkey"
            columns: ["work_id"]
            isOneToOne: false
            referencedRelation: "works_registry"
            referencedColumns: ["id"]
          },
        ]
      }
      registry_notifications: {
        Row: {
          created_at: string
          id: string
          message: string
          metadata: Json
          read: boolean
          title: string
          type: string
          user_id: string
          work_id: string | null
        }
        Insert: {
          created_at?: string
          id?: string
          message: string
          metadata?: Json
          read?: boolean
          title: string
          type: string
          user_id: string
          work_id?: string | null
        }
        Update: {
          created_at?: string
          id?: string
          message?: string
          metadata?: Json
          read?: boolean
          title?: string
          type?: string
          user_id?: string
          work_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "registry_notifications_work_id_fkey"
            columns: ["work_id"]
            isOneToOne: false
            referencedRelation: "works_registry"
            referencedColumns: ["id"]
          },
        ]
      }
      royalty_calculation_contracts: {
        Row: {
          calculation_id: string
          contract_id: string
        }
        Insert: {
          calculation_id: string
          contract_id: string
        }
        Update: {
          calculation_id?: string
          contract_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "royalty_calculation_contracts_calculation_id_fkey"
            columns: ["calculation_id"]
            isOneToOne: false
            referencedRelation: "royalty_calculations"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "royalty_calculation_contracts_contract_id_fkey"
            columns: ["contract_id"]
            isOneToOne: false
            referencedRelation: "project_files"
            referencedColumns: ["id"]
          },
        ]
      }
      royalty_calculations: {
        Row: {
          created_at: string
          id: string
          project_id: string
          results: Json
          royalty_statement_id: string
          updated_at: string
          user_id: string
        }
        Insert: {
          created_at?: string
          id?: string
          project_id: string
          results: Json
          royalty_statement_id: string
          updated_at?: string
          user_id: string
        }
        Update: {
          created_at?: string
          id?: string
          project_id?: string
          results?: Json
          royalty_statement_id?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "royalty_calculations_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "royalty_calculations_royalty_statement_id_fkey"
            columns: ["royalty_statement_id"]
            isOneToOne: false
            referencedRelation: "project_files"
            referencedColumns: ["id"]
          },
        ]
      }
      slack_notifications: {
        Row: {
          channel_id: string
          created_at: string | null
          id: string
          is_read: boolean | null
          message_text: string
          project_id: string | null
          sender_avatar_url: string | null
          sender_name: string
          slack_ts: string
          user_id: string
        }
        Insert: {
          channel_id: string
          created_at?: string | null
          id?: string
          is_read?: boolean | null
          message_text: string
          project_id?: string | null
          sender_avatar_url?: string | null
          sender_name: string
          slack_ts: string
          user_id: string
        }
        Update: {
          channel_id?: string
          created_at?: string | null
          id?: string
          is_read?: boolean | null
          message_text?: string
          project_id?: string | null
          sender_avatar_url?: string | null
          sender_name?: string
          slack_ts?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "slack_notifications_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      sync_log: {
        Row: {
          created_at: string | null
          direction: string
          entity_id: string | null
          entity_type: string
          error_message: string | null
          external_id: string | null
          id: string
          metadata: Json | null
          provider: string
          status: string
          user_id: string
        }
        Insert: {
          created_at?: string | null
          direction: string
          entity_id?: string | null
          entity_type: string
          error_message?: string | null
          external_id?: string | null
          id?: string
          metadata?: Json | null
          provider: string
          status: string
          user_id: string
        }
        Update: {
          created_at?: string | null
          direction?: string
          entity_id?: string | null
          entity_type?: string
          error_message?: string | null
          external_id?: string | null
          id?: string
          metadata?: Json | null
          provider?: string
          status?: string
          user_id?: string
        }
        Relationships: []
      }
      team_cards: {
        Row: {
          avatar_url: string | null
          bio: string | null
          company: string | null
          created_at: string
          custom_links: Json
          display_name: string
          dsp_links: Json
          email: string
          first_name: string
          id: string
          last_name: string
          phone: string | null
          role: string | null
          social_links: Json
          updated_at: string
          user_id: string
          visible_fields: Json
          website: string | null
        }
        Insert: {
          avatar_url?: string | null
          bio?: string | null
          company?: string | null
          created_at?: string
          custom_links?: Json
          display_name: string
          dsp_links?: Json
          email: string
          first_name: string
          id?: string
          last_name: string
          phone?: string | null
          role?: string | null
          social_links?: Json
          updated_at?: string
          user_id: string
          visible_fields?: Json
          website?: string | null
        }
        Update: {
          avatar_url?: string | null
          bio?: string | null
          company?: string | null
          created_at?: string
          custom_links?: Json
          display_name?: string
          dsp_links?: Json
          email?: string
          first_name?: string
          id?: string
          last_name?: string
          phone?: string | null
          role?: string | null
          social_links?: Json
          updated_at?: string
          user_id?: string
          visible_fields?: Json
          website?: string | null
        }
        Relationships: []
      }
      user_onboarding: {
        Row: {
          artists_completed: boolean
          created_at: string
          oneclick_completed: boolean
          portfolio_completed: boolean
          profile_completed: boolean
          project_detail_completed: boolean
          registry_completed: boolean
          splitsheet_completed: boolean
          user_id: string
          work_detail_completed: boolean
          workspace_completed: boolean
          zoe_completed: boolean
        }
        Insert: {
          artists_completed?: boolean
          created_at?: string
          oneclick_completed?: boolean
          portfolio_completed?: boolean
          profile_completed?: boolean
          project_detail_completed?: boolean
          registry_completed?: boolean
          splitsheet_completed?: boolean
          user_id: string
          work_detail_completed?: boolean
          workspace_completed?: boolean
          zoe_completed?: boolean
        }
        Update: {
          artists_completed?: boolean
          created_at?: string
          oneclick_completed?: boolean
          portfolio_completed?: boolean
          profile_completed?: boolean
          project_detail_completed?: boolean
          registry_completed?: boolean
          splitsheet_completed?: boolean
          user_id?: string
          work_detail_completed?: boolean
          workspace_completed?: boolean
          zoe_completed?: boolean
        }
        Relationships: []
      }
      work_audio_links: {
        Row: {
          audio_file_id: string
          created_at: string | null
          id: string
          work_id: string
        }
        Insert: {
          audio_file_id: string
          created_at?: string | null
          id?: string
          work_id: string
        }
        Update: {
          audio_file_id?: string
          created_at?: string | null
          id?: string
          work_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "work_audio_links_audio_file_id_fkey"
            columns: ["audio_file_id"]
            isOneToOne: false
            referencedRelation: "audio_files"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "work_audio_links_work_id_fkey"
            columns: ["work_id"]
            isOneToOne: false
            referencedRelation: "works_registry"
            referencedColumns: ["id"]
          },
        ]
      }
      work_files: {
        Row: {
          created_at: string | null
          file_id: string
          id: string
          work_id: string
        }
        Insert: {
          created_at?: string | null
          file_id: string
          id?: string
          work_id: string
        }
        Update: {
          created_at?: string | null
          file_id?: string
          id?: string
          work_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "work_files_file_id_fkey"
            columns: ["file_id"]
            isOneToOne: false
            referencedRelation: "project_files"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "work_files_work_id_fkey"
            columns: ["work_id"]
            isOneToOne: false
            referencedRelation: "works_registry"
            referencedColumns: ["id"]
          },
        ]
      }
      works_registry: {
        Row: {
          artist_id: string
          created_at: string
          custom_work_type: string | null
          id: string
          isrc: string | null
          iswc: string | null
          notes: string | null
          project_id: string
          release_date: string | null
          status: string
          title: string
          upc: string | null
          updated_at: string
          user_id: string
          work_type: string
        }
        Insert: {
          artist_id: string
          created_at?: string
          custom_work_type?: string | null
          id?: string
          isrc?: string | null
          iswc?: string | null
          notes?: string | null
          project_id: string
          release_date?: string | null
          status?: string
          title: string
          upc?: string | null
          updated_at?: string
          user_id: string
          work_type?: string
        }
        Update: {
          artist_id?: string
          created_at?: string
          custom_work_type?: string | null
          id?: string
          isrc?: string | null
          iswc?: string | null
          notes?: string | null
          project_id?: string
          release_date?: string | null
          status?: string
          title?: string
          upc?: string | null
          updated_at?: string
          user_id?: string
          work_type?: string
        }
        Relationships: [
          {
            foreignKeyName: "works_registry_artist_id_fkey"
            columns: ["artist_id"]
            isOneToOne: false
            referencedRelation: "artists"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "works_registry_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      workspace_settings: {
        Row: {
          accent_color: string | null
          board_grouping: string | null
          board_period: string
          calendar_view: string | null
          created_at: string | null
          custom_period_days: number | null
          id: string
          timezone: string | null
          updated_at: string | null
          use_24h_time: boolean | null
          user_id: string
        }
        Insert: {
          accent_color?: string | null
          board_grouping?: string | null
          board_period?: string
          calendar_view?: string | null
          created_at?: string | null
          custom_period_days?: number | null
          id?: string
          timezone?: string | null
          updated_at?: string | null
          use_24h_time?: boolean | null
          user_id: string
        }
        Update: {
          accent_color?: string | null
          board_grouping?: string | null
          board_period?: string
          calendar_view?: string | null
          created_at?: string | null
          custom_period_days?: number | null
          id?: string
          timezone?: string | null
          updated_at?: string | null
          use_24h_time?: boolean | null
          user_id?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      execute_readonly_query: { Args: { query_text: string }; Returns: Json }
      get_project_role: { Args: { p_id: string }; Returns: string }
      get_user_id_by_email: { Args: { lookup_email: string }; Returns: string }
      is_project_member: { Args: { p_id: string }; Returns: boolean }
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
