


SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;


COMMENT ON SCHEMA "public" IS 'standard public schema';



CREATE EXTENSION IF NOT EXISTS "pg_graphql" WITH SCHEMA "graphql";






CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "supabase_vault" WITH SCHEMA "vault";






CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA "extensions";






CREATE OR REPLACE FUNCTION "public"."delete_orphan_calculation"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
  -- Delete the parent calculation. 
  -- This will cascade delete other junction rows for this calculation, 
  -- which will re-trigger this function, but the DELETE on the parent will be a no-op for subsequent calls.
  DELETE FROM public.royalty_calculations WHERE id = OLD.calculation_id;
  RETURN OLD;
END;
$$;


ALTER FUNCTION "public"."delete_orphan_calculation"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."execute_readonly_query"("query_text" "text") RETURNS "jsonb"
    LANGUAGE "plpgsql" SECURITY DEFINER
    AS $$
DECLARE
  result JSONB;
BEGIN
  -- Validate that the query is read-only (SELECT only)
  IF query_text !~* '^\s*SELECT' THEN
    RAISE EXCEPTION 'Only SELECT queries are allowed';
  END IF;
  
  -- Check for dangerous keywords
  IF query_text ~* '(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE)' THEN
    RAISE EXCEPTION 'Query contains forbidden keywords';
  END IF;
  
  -- Execute the query and return results as JSONB
  EXECUTE format('SELECT jsonb_agg(row_to_json(t)) FROM (%s) t', query_text) INTO result;
  
  RETURN COALESCE(result, '[]'::jsonb);
END;
$$;


ALTER FUNCTION "public"."execute_readonly_query"("query_text" "text") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."handle_new_user"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
begin
  insert into public.profiles (id, full_name, avatar_url)
  values (new.id, new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'avatar_url');
  return new;
end;
$$;


ALTER FUNCTION "public"."handle_new_user"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."handle_new_user_onboarding"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
begin
  insert into public.user_onboarding (user_id)
  values (new.id);
  return new;
end;
$$;


ALTER FUNCTION "public"."handle_new_user_onboarding"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."update_updated_at_column"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    SET "search_path" TO 'public'
    AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."update_updated_at_column"() OWNER TO "postgres";

SET default_tablespace = '';

SET default_table_access_method = "heap";


CREATE TABLE IF NOT EXISTS "public"."artists" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "name" "text" NOT NULL,
    "email" "text" NOT NULL,
    "bio" "text",
    "genres" "text"[] DEFAULT '{}'::"text"[],
    "avatar_url" "text",
    "social_instagram" "text",
    "social_tiktok" "text",
    "social_youtube" "text",
    "dsp_spotify" "text",
    "dsp_apple_music" "text",
    "dsp_soundcloud" "text",
    "additional_epk" "text",
    "additional_press_kit" "text",
    "additional_linktree" "text",
    "custom_links" "jsonb" DEFAULT '[]'::"jsonb",
    "has_contract" boolean DEFAULT false,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "custom_social_links" "jsonb" DEFAULT '[]'::"jsonb",
    "custom_dsp_links" "jsonb" DEFAULT '[]'::"jsonb",
    "user_id" "uuid"
);


ALTER TABLE "public"."artists" OWNER TO "postgres";


COMMENT ON COLUMN "public"."artists"."custom_social_links" IS 'Custom social media links (e.g., Twitter, Threads, etc.) stored as JSON array: [{"id": "1", "label": "Twitter", "url": "https://twitter.com/username"}]';



COMMENT ON COLUMN "public"."artists"."custom_dsp_links" IS 'Custom DSP/streaming platform links (e.g., Tidal, Deezer, etc.) stored as JSON array: [{"id": "1", "label": "Tidal", "url": "https://tidal.com/artist"}]';



CREATE TABLE IF NOT EXISTS "public"."audio_files" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "folder_id" "uuid" NOT NULL,
    "file_name" "text" NOT NULL,
    "file_url" "text" NOT NULL,
    "file_path" "text" NOT NULL,
    "file_size" bigint,
    "file_type" "text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."audio_files" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."audio_folders" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "artist_id" "uuid" NOT NULL,
    "name" "text" NOT NULL,
    "parent_id" "uuid",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."audio_folders" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."board_columns" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "artist_id" "uuid",
    "title" "text" NOT NULL,
    "position" integer DEFAULT 0 NOT NULL,
    "color" "text",
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."board_columns" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."board_task_artists" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "task_id" "uuid" NOT NULL,
    "artist_id" "uuid" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."board_task_artists" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."board_task_comments" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "task_id" "uuid" NOT NULL,
    "user_id" "uuid" NOT NULL,
    "content" "text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."board_task_comments" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."board_task_contracts" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "task_id" "uuid" NOT NULL,
    "project_file_id" "uuid" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."board_task_contracts" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."board_task_projects" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "task_id" "uuid" NOT NULL,
    "project_id" "uuid" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."board_task_projects" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."board_tasks" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "column_id" "uuid",
    "user_id" "uuid" NOT NULL,
    "title" "text" NOT NULL,
    "description" "text",
    "position" integer DEFAULT 0 NOT NULL,
    "priority" "text",
    "due_date" "date",
    "artist_id" "uuid",
    "project_id" "uuid",
    "assignee_name" "text",
    "labels" "text"[],
    "external_id" "text",
    "external_provider" "text",
    "external_url" "text",
    "last_synced_at" timestamp with time zone,
    "sync_hash" "text",
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"(),
    "start_date" "date",
    "color" "text",
    "parent_task_id" "uuid",
    "is_parent" boolean DEFAULT false,
    "completed_at" timestamp with time zone,
    CONSTRAINT "board_tasks_priority_check" CHECK (("priority" = ANY (ARRAY['low'::"text", 'medium'::"text", 'high'::"text", 'urgent'::"text"])))
);


ALTER TABLE "public"."board_tasks" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."contacts" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "name" "text" NOT NULL,
    "email" "text",
    "phone" "text",
    "role" "text",
    "notes" "text",
    "stripe_account_id" "text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."contacts" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."drive_sync_mappings" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "project_file_id" "uuid",
    "project_id" "uuid",
    "drive_file_id" "text" NOT NULL,
    "drive_folder_id" "text",
    "sync_direction" "text" NOT NULL,
    "last_synced_at" timestamp with time zone,
    "drive_modified_at" timestamp with time zone,
    "local_modified_at" timestamp with time zone,
    "created_at" timestamp with time zone DEFAULT "now"(),
    CONSTRAINT "drive_sync_mappings_sync_direction_check" CHECK (("sync_direction" = ANY (ARRAY['to_drive'::"text", 'from_drive'::"text", 'bidirectional'::"text"])))
);


ALTER TABLE "public"."drive_sync_mappings" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."file_shares" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "contact_id" "uuid",
    "recipient_email" "text" NOT NULL,
    "recipient_name" "text",
    "file_name" "text" NOT NULL,
    "file_source" "text" NOT NULL,
    "file_id" "text" NOT NULL,
    "message" "text",
    "shared_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "link_expires_at" timestamp with time zone NOT NULL,
    "status" "text" DEFAULT 'sent'::"text" NOT NULL
);


ALTER TABLE "public"."file_shares" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."integration_connections" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "provider" "text" NOT NULL,
    "access_token_encrypted" "text" NOT NULL,
    "refresh_token_encrypted" "text",
    "token_expires_at" timestamp with time zone,
    "provider_user_id" "text",
    "provider_workspace_id" "text",
    "provider_metadata" "jsonb" DEFAULT '{}'::"jsonb",
    "scopes" "text"[],
    "status" "text" DEFAULT 'active'::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"(),
    CONSTRAINT "integration_connections_provider_check" CHECK (("provider" = ANY (ARRAY['google_drive'::"text", 'slack'::"text", 'notion'::"text", 'monday'::"text"]))),
    CONSTRAINT "integration_connections_status_check" CHECK (("status" = ANY (ARRAY['active'::"text", 'expired'::"text", 'revoked'::"text"])))
);


ALTER TABLE "public"."integration_connections" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."mailing_list" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "email" "text" NOT NULL,
    "name" "text",
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."mailing_list" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."notification_settings" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "provider" "text" NOT NULL,
    "event_type" "text" NOT NULL,
    "enabled" boolean DEFAULT true,
    "channel_id" "text",
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."notification_settings" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."payments" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "contact_id" "uuid",
    "party_name" "text" NOT NULL,
    "amount" numeric(10,2) NOT NULL,
    "currency" "text" DEFAULT 'usd'::"text" NOT NULL,
    "status" "text" DEFAULT 'pending'::"text" NOT NULL,
    "stripe_payment_intent_id" "text",
    "stripe_transfer_id" "text",
    "metadata" "jsonb",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."payments" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."profiles" (
    "id" "uuid" NOT NULL,
    "updated_at" timestamp with time zone,
    "full_name" "text",
    "website" "text",
    "company" "text",
    "phone" "text",
    "first_name" "text",
    "last_name" "text",
    "given_name" "text",
    "industry" "text",
    "onboarding_completed" boolean DEFAULT false,
    "walkthrough_completed" boolean DEFAULT false
);


ALTER TABLE "public"."profiles" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."project_audio_links" (
    "project_id" "uuid" NOT NULL,
    "audio_file_id" "uuid" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."project_audio_links" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."project_files" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "project_id" "uuid" NOT NULL,
    "folder_category" "text" NOT NULL,
    "file_name" "text" NOT NULL,
    "file_url" "text" NOT NULL,
    "file_size" bigint,
    "file_type" "text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "file_path" "text",
    "contract_markdown" "text",
    CONSTRAINT "project_files_folder_category_check" CHECK (("folder_category" = ANY (ARRAY['contract'::"text", 'split_sheet'::"text", 'royalty_statement'::"text", 'other'::"text"])))
);


ALTER TABLE "public"."project_files" OWNER TO "postgres";


COMMENT ON COLUMN "public"."project_files"."folder_category" IS 'File category: contract, split_sheet, royalty_statement, or other';



COMMENT ON COLUMN "public"."project_files"."file_url" IS 'Public URL for accessing the file';



COMMENT ON COLUMN "public"."project_files"."file_path" IS 'Storage path for the file in Supabase Storage';



COMMENT ON COLUMN "public"."project_files"."contract_markdown" IS 'Full markdown text of contract PDF, used for full-document LLM context. Populated during upload or lazily on first access.';



CREATE TABLE IF NOT EXISTS "public"."projects" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "artist_id" "uuid" NOT NULL,
    "name" "text" NOT NULL,
    "description" "text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."projects" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."royalty_calculation_contracts" (
    "calculation_id" "uuid" NOT NULL,
    "contract_id" "uuid" NOT NULL
);


ALTER TABLE "public"."royalty_calculation_contracts" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."royalty_calculations" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "royalty_statement_id" "uuid" NOT NULL,
    "project_id" "uuid" NOT NULL,
    "user_id" "uuid" NOT NULL,
    "results" "jsonb" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."royalty_calculations" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."sync_log" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "provider" "text" NOT NULL,
    "direction" "text" NOT NULL,
    "entity_type" "text" NOT NULL,
    "entity_id" "uuid",
    "external_id" "text",
    "status" "text" NOT NULL,
    "error_message" "text",
    "metadata" "jsonb" DEFAULT '{}'::"jsonb",
    "created_at" timestamp with time zone DEFAULT "now"(),
    CONSTRAINT "sync_log_direction_check" CHECK (("direction" = ANY (ARRAY['push'::"text", 'pull'::"text", 'bidirectional'::"text"]))),
    CONSTRAINT "sync_log_status_check" CHECK (("status" = ANY (ARRAY['success'::"text", 'conflict'::"text", 'error'::"text"])))
);


ALTER TABLE "public"."sync_log" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."user_onboarding" (
    "user_id" "uuid" NOT NULL,
    "oneclick_completed" boolean DEFAULT false NOT NULL,
    "zoe_completed" boolean DEFAULT false NOT NULL,
    "splitsheet_completed" boolean DEFAULT false NOT NULL,
    "artists_completed" boolean DEFAULT false NOT NULL,
    "workspace_completed" boolean DEFAULT false NOT NULL,
    "portfolio_completed" boolean DEFAULT false NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."user_onboarding" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."workspace_settings" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "uuid" NOT NULL,
    "board_period" "text" DEFAULT 'monthly'::"text" NOT NULL,
    "custom_period_days" integer,
    "accent_color" "text",
    "board_grouping" "text" DEFAULT 'column'::"text",
    "use_24h_time" boolean DEFAULT false,
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"(),
    "calendar_view" "text" DEFAULT 'month'::"text",
    "timezone" "text"
);


ALTER TABLE "public"."workspace_settings" OWNER TO "postgres";


ALTER TABLE ONLY "public"."artists"
    ADD CONSTRAINT "artists_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."audio_files"
    ADD CONSTRAINT "audio_files_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."audio_folders"
    ADD CONSTRAINT "audio_folders_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."board_columns"
    ADD CONSTRAINT "board_columns_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."board_task_artists"
    ADD CONSTRAINT "board_task_artists_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."board_task_artists"
    ADD CONSTRAINT "board_task_artists_task_id_artist_id_key" UNIQUE ("task_id", "artist_id");



ALTER TABLE ONLY "public"."board_task_comments"
    ADD CONSTRAINT "board_task_comments_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."board_task_contracts"
    ADD CONSTRAINT "board_task_contracts_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."board_task_contracts"
    ADD CONSTRAINT "board_task_contracts_task_id_project_file_id_key" UNIQUE ("task_id", "project_file_id");



ALTER TABLE ONLY "public"."board_task_projects"
    ADD CONSTRAINT "board_task_projects_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."board_task_projects"
    ADD CONSTRAINT "board_task_projects_task_id_project_id_key" UNIQUE ("task_id", "project_id");



ALTER TABLE ONLY "public"."board_tasks"
    ADD CONSTRAINT "board_tasks_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."contacts"
    ADD CONSTRAINT "contacts_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."drive_sync_mappings"
    ADD CONSTRAINT "drive_sync_mappings_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."file_shares"
    ADD CONSTRAINT "file_shares_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."integration_connections"
    ADD CONSTRAINT "integration_connections_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."integration_connections"
    ADD CONSTRAINT "integration_connections_user_id_provider_key" UNIQUE ("user_id", "provider");



ALTER TABLE ONLY "public"."mailing_list"
    ADD CONSTRAINT "mailing_list_email_key" UNIQUE ("email");



ALTER TABLE ONLY "public"."mailing_list"
    ADD CONSTRAINT "mailing_list_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."notification_settings"
    ADD CONSTRAINT "notification_settings_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."notification_settings"
    ADD CONSTRAINT "notification_settings_user_id_provider_event_type_key" UNIQUE ("user_id", "provider", "event_type");



ALTER TABLE ONLY "public"."payments"
    ADD CONSTRAINT "payments_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."profiles"
    ADD CONSTRAINT "profiles_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."project_audio_links"
    ADD CONSTRAINT "project_audio_links_pkey" PRIMARY KEY ("project_id", "audio_file_id");



ALTER TABLE ONLY "public"."project_files"
    ADD CONSTRAINT "project_files_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."projects"
    ADD CONSTRAINT "projects_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."royalty_calculation_contracts"
    ADD CONSTRAINT "royalty_calculation_contracts_pkey" PRIMARY KEY ("calculation_id", "contract_id");



ALTER TABLE ONLY "public"."royalty_calculations"
    ADD CONSTRAINT "royalty_calculations_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."sync_log"
    ADD CONSTRAINT "sync_log_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."user_onboarding"
    ADD CONSTRAINT "user_onboarding_pkey" PRIMARY KEY ("user_id");



ALTER TABLE ONLY "public"."workspace_settings"
    ADD CONSTRAINT "workspace_settings_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."workspace_settings"
    ADD CONSTRAINT "workspace_settings_user_id_key" UNIQUE ("user_id");



CREATE INDEX "idx_artists_user_id" ON "public"."artists" USING "btree" ("user_id");



CREATE INDEX "idx_audio_files_folder_id" ON "public"."audio_files" USING "btree" ("folder_id");



CREATE UNIQUE INDEX "idx_audio_files_unique_name" ON "public"."audio_files" USING "btree" ("folder_id", "file_name");



CREATE INDEX "idx_audio_folders_artist_id" ON "public"."audio_folders" USING "btree" ("artist_id");



CREATE INDEX "idx_audio_folders_parent_id" ON "public"."audio_folders" USING "btree" ("parent_id");



CREATE UNIQUE INDEX "idx_audio_folders_unique_name" ON "public"."audio_folders" USING "btree" ("artist_id", "parent_id", "name") WHERE ("parent_id" IS NOT NULL);



CREATE UNIQUE INDEX "idx_audio_folders_unique_name_root" ON "public"."audio_folders" USING "btree" ("artist_id", "name") WHERE ("parent_id" IS NULL);



CREATE INDEX "idx_board_columns_user" ON "public"."board_columns" USING "btree" ("user_id");



CREATE INDEX "idx_board_task_artists_task" ON "public"."board_task_artists" USING "btree" ("task_id");



CREATE INDEX "idx_board_task_comments_task" ON "public"."board_task_comments" USING "btree" ("task_id");



CREATE INDEX "idx_board_task_contracts_task" ON "public"."board_task_contracts" USING "btree" ("task_id");



CREATE INDEX "idx_board_task_projects_task" ON "public"."board_task_projects" USING "btree" ("task_id");



CREATE INDEX "idx_board_tasks_column" ON "public"."board_tasks" USING "btree" ("column_id");



CREATE INDEX "idx_board_tasks_due_date" ON "public"."board_tasks" USING "btree" ("user_id", "due_date");



CREATE INDEX "idx_board_tasks_is_parent" ON "public"."board_tasks" USING "btree" ("user_id", "is_parent");



CREATE INDEX "idx_board_tasks_parent" ON "public"."board_tasks" USING "btree" ("parent_task_id");



CREATE INDEX "idx_board_tasks_start_date" ON "public"."board_tasks" USING "btree" ("user_id", "start_date");



CREATE INDEX "idx_board_tasks_user" ON "public"."board_tasks" USING "btree" ("user_id");



CREATE INDEX "idx_contacts_user_id" ON "public"."contacts" USING "btree" ("user_id");



CREATE INDEX "idx_drive_sync_user" ON "public"."drive_sync_mappings" USING "btree" ("user_id");



CREATE INDEX "idx_file_shares_contact_id" ON "public"."file_shares" USING "btree" ("contact_id");



CREATE INDEX "idx_file_shares_user_id" ON "public"."file_shares" USING "btree" ("user_id");



CREATE INDEX "idx_integration_connections_user_provider" ON "public"."integration_connections" USING "btree" ("user_id", "provider");



CREATE INDEX "idx_payments_contact_id" ON "public"."payments" USING "btree" ("contact_id");



CREATE INDEX "idx_payments_status" ON "public"."payments" USING "btree" ("status");



CREATE INDEX "idx_payments_user_id" ON "public"."payments" USING "btree" ("user_id");



CREATE INDEX "idx_project_audio_links_audio_file_id" ON "public"."project_audio_links" USING "btree" ("audio_file_id");



CREATE INDEX "idx_project_files_folder_category" ON "public"."project_files" USING "btree" ("folder_category");



CREATE INDEX "idx_project_files_project_id" ON "public"."project_files" USING "btree" ("project_id");



CREATE INDEX "idx_sync_log_user" ON "public"."sync_log" USING "btree" ("user_id", "created_at" DESC);



CREATE UNIQUE INDEX "project_files_project_id_normalized_file_name_unique" ON "public"."project_files" USING "btree" ("project_id", "lower"("btrim"("file_name")));



CREATE OR REPLACE TRIGGER "cleanup_calculation_on_contract_delete" AFTER DELETE ON "public"."royalty_calculation_contracts" FOR EACH ROW EXECUTE FUNCTION "public"."delete_orphan_calculation"();



CREATE OR REPLACE TRIGGER "update_artists_updated_at" BEFORE UPDATE ON "public"."artists" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_project_files_updated_at" BEFORE UPDATE ON "public"."project_files" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_projects_updated_at" BEFORE UPDATE ON "public"."projects" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_royalty_calculations_updated_at" BEFORE UPDATE ON "public"."royalty_calculations" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



ALTER TABLE ONLY "public"."artists"
    ADD CONSTRAINT "artists_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."audio_files"
    ADD CONSTRAINT "audio_files_folder_id_fkey" FOREIGN KEY ("folder_id") REFERENCES "public"."audio_folders"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."audio_folders"
    ADD CONSTRAINT "audio_folders_artist_id_fkey" FOREIGN KEY ("artist_id") REFERENCES "public"."artists"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."audio_folders"
    ADD CONSTRAINT "audio_folders_parent_id_fkey" FOREIGN KEY ("parent_id") REFERENCES "public"."audio_folders"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."board_columns"
    ADD CONSTRAINT "board_columns_artist_id_fkey" FOREIGN KEY ("artist_id") REFERENCES "public"."artists"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."board_columns"
    ADD CONSTRAINT "board_columns_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."board_task_artists"
    ADD CONSTRAINT "board_task_artists_artist_id_fkey" FOREIGN KEY ("artist_id") REFERENCES "public"."artists"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."board_task_artists"
    ADD CONSTRAINT "board_task_artists_task_id_fkey" FOREIGN KEY ("task_id") REFERENCES "public"."board_tasks"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."board_task_comments"
    ADD CONSTRAINT "board_task_comments_task_id_fkey" FOREIGN KEY ("task_id") REFERENCES "public"."board_tasks"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."board_task_comments"
    ADD CONSTRAINT "board_task_comments_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."board_task_contracts"
    ADD CONSTRAINT "board_task_contracts_project_file_id_fkey" FOREIGN KEY ("project_file_id") REFERENCES "public"."project_files"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."board_task_contracts"
    ADD CONSTRAINT "board_task_contracts_task_id_fkey" FOREIGN KEY ("task_id") REFERENCES "public"."board_tasks"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."board_task_projects"
    ADD CONSTRAINT "board_task_projects_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "public"."projects"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."board_task_projects"
    ADD CONSTRAINT "board_task_projects_task_id_fkey" FOREIGN KEY ("task_id") REFERENCES "public"."board_tasks"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."board_tasks"
    ADD CONSTRAINT "board_tasks_artist_id_fkey" FOREIGN KEY ("artist_id") REFERENCES "public"."artists"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."board_tasks"
    ADD CONSTRAINT "board_tasks_column_id_fkey" FOREIGN KEY ("column_id") REFERENCES "public"."board_columns"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."board_tasks"
    ADD CONSTRAINT "board_tasks_parent_task_id_fkey" FOREIGN KEY ("parent_task_id") REFERENCES "public"."board_tasks"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."board_tasks"
    ADD CONSTRAINT "board_tasks_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "public"."projects"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."board_tasks"
    ADD CONSTRAINT "board_tasks_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."contacts"
    ADD CONSTRAINT "contacts_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."drive_sync_mappings"
    ADD CONSTRAINT "drive_sync_mappings_project_file_id_fkey" FOREIGN KEY ("project_file_id") REFERENCES "public"."project_files"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."drive_sync_mappings"
    ADD CONSTRAINT "drive_sync_mappings_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "public"."projects"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."drive_sync_mappings"
    ADD CONSTRAINT "drive_sync_mappings_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."file_shares"
    ADD CONSTRAINT "file_shares_contact_id_fkey" FOREIGN KEY ("contact_id") REFERENCES "public"."contacts"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."file_shares"
    ADD CONSTRAINT "file_shares_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."integration_connections"
    ADD CONSTRAINT "integration_connections_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."notification_settings"
    ADD CONSTRAINT "notification_settings_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."payments"
    ADD CONSTRAINT "payments_contact_id_fkey" FOREIGN KEY ("contact_id") REFERENCES "public"."contacts"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."payments"
    ADD CONSTRAINT "payments_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."profiles"
    ADD CONSTRAINT "profiles_id_fkey" FOREIGN KEY ("id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."project_audio_links"
    ADD CONSTRAINT "project_audio_links_audio_file_id_fkey" FOREIGN KEY ("audio_file_id") REFERENCES "public"."audio_files"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."project_audio_links"
    ADD CONSTRAINT "project_audio_links_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "public"."projects"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."project_files"
    ADD CONSTRAINT "project_files_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "public"."projects"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."projects"
    ADD CONSTRAINT "projects_artist_id_fkey" FOREIGN KEY ("artist_id") REFERENCES "public"."artists"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."royalty_calculation_contracts"
    ADD CONSTRAINT "royalty_calculation_contracts_calculation_id_fkey" FOREIGN KEY ("calculation_id") REFERENCES "public"."royalty_calculations"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."royalty_calculation_contracts"
    ADD CONSTRAINT "royalty_calculation_contracts_contract_id_fkey" FOREIGN KEY ("contract_id") REFERENCES "public"."project_files"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."royalty_calculations"
    ADD CONSTRAINT "royalty_calculations_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "public"."projects"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."royalty_calculations"
    ADD CONSTRAINT "royalty_calculations_royalty_statement_id_fkey" FOREIGN KEY ("royalty_statement_id") REFERENCES "public"."project_files"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."sync_log"
    ADD CONSTRAINT "sync_log_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."user_onboarding"
    ADD CONSTRAINT "user_onboarding_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."workspace_settings"
    ADD CONSTRAINT "workspace_settings_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



CREATE POLICY "Allow anonymous inserts" ON "public"."mailing_list" FOR INSERT WITH CHECK (true);



CREATE POLICY "Public profiles are viewable by everyone." ON "public"."profiles" FOR SELECT USING (true);



CREATE POLICY "Users can create audio files for their artists" ON "public"."audio_files" FOR INSERT WITH CHECK ((EXISTS ( SELECT 1
   FROM ("public"."audio_folders" "af"
     JOIN "public"."artists" "a" ON (("a"."id" = "af"."artist_id")))
  WHERE (("af"."id" = "audio_files"."folder_id") AND ("a"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can create audio folders for their artists" ON "public"."audio_folders" FOR INSERT WITH CHECK ((EXISTS ( SELECT 1
   FROM "public"."artists" "a"
  WHERE (("a"."id" = "audio_folders"."artist_id") AND ("a"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can create own project files" ON "public"."project_files" FOR INSERT WITH CHECK ((EXISTS ( SELECT 1
   FROM ("public"."projects"
     JOIN "public"."artists" ON (("artists"."id" = "projects"."artist_id")))
  WHERE (("projects"."id" = "project_files"."project_id") AND ("artists"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can create own projects" ON "public"."projects" FOR INSERT WITH CHECK ((EXISTS ( SELECT 1
   FROM "public"."artists"
  WHERE (("artists"."id" = "projects"."artist_id") AND ("artists"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can create project audio links" ON "public"."project_audio_links" FOR INSERT WITH CHECK ((EXISTS ( SELECT 1
   FROM ("public"."projects" "p"
     JOIN "public"."artists" "a" ON (("a"."id" = "p"."artist_id")))
  WHERE (("p"."id" = "project_audio_links"."project_id") AND ("a"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can create their own artists" ON "public"."artists" FOR INSERT WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can create their own calculations" ON "public"."royalty_calculations" FOR INSERT WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can delete audio files for their artists" ON "public"."audio_files" FOR DELETE USING ((EXISTS ( SELECT 1
   FROM ("public"."audio_folders" "af"
     JOIN "public"."artists" "a" ON (("a"."id" = "af"."artist_id")))
  WHERE (("af"."id" = "audio_files"."folder_id") AND ("a"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can delete audio folders for their artists" ON "public"."audio_folders" FOR DELETE USING ((EXISTS ( SELECT 1
   FROM "public"."artists" "a"
  WHERE (("a"."id" = "audio_folders"."artist_id") AND ("a"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can delete own contacts" ON "public"."contacts" FOR DELETE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can delete own project files" ON "public"."project_files" FOR DELETE USING ((EXISTS ( SELECT 1
   FROM ("public"."projects"
     JOIN "public"."artists" ON (("artists"."id" = "projects"."artist_id")))
  WHERE (("projects"."id" = "project_files"."project_id") AND ("artists"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can delete own projects" ON "public"."projects" FOR DELETE USING ((EXISTS ( SELECT 1
   FROM "public"."artists"
  WHERE (("artists"."id" = "projects"."artist_id") AND ("artists"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can delete their own artists" ON "public"."artists" FOR DELETE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can delete their own calculation contracts" ON "public"."royalty_calculation_contracts" FOR DELETE USING ((EXISTS ( SELECT 1
   FROM "public"."royalty_calculations" "rc"
  WHERE (("rc"."id" = "royalty_calculation_contracts"."calculation_id") AND ("rc"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can delete their own calculations" ON "public"."royalty_calculations" FOR DELETE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can delete their project audio links" ON "public"."project_audio_links" FOR DELETE USING ((EXISTS ( SELECT 1
   FROM ("public"."projects" "p"
     JOIN "public"."artists" "a" ON (("a"."id" = "p"."artist_id")))
  WHERE (("p"."id" = "project_audio_links"."project_id") AND ("a"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can insert own contacts" ON "public"."contacts" FOR INSERT WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can insert own onboarding status" ON "public"."user_onboarding" FOR INSERT WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can insert own payments" ON "public"."payments" FOR INSERT WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can insert their own calculation contracts" ON "public"."royalty_calculation_contracts" FOR INSERT WITH CHECK ((EXISTS ( SELECT 1
   FROM "public"."royalty_calculations" "rc"
  WHERE (("rc"."id" = "royalty_calculation_contracts"."calculation_id") AND ("rc"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can insert their own file shares" ON "public"."file_shares" FOR INSERT WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can insert their own profile." ON "public"."profiles" FOR INSERT WITH CHECK (("auth"."uid"() = "id"));



CREATE POLICY "Users can read own onboarding status" ON "public"."user_onboarding" FOR SELECT USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can update own contacts" ON "public"."contacts" FOR UPDATE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can update own onboarding status" ON "public"."user_onboarding" FOR UPDATE USING (("auth"."uid"() = "user_id")) WITH CHECK (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can update own payments" ON "public"."payments" FOR UPDATE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can update own profile." ON "public"."profiles" FOR UPDATE USING (("auth"."uid"() = "id"));



CREATE POLICY "Users can update own project files" ON "public"."project_files" FOR UPDATE USING ((EXISTS ( SELECT 1
   FROM ("public"."projects"
     JOIN "public"."artists" ON (("artists"."id" = "projects"."artist_id")))
  WHERE (("projects"."id" = "project_files"."project_id") AND ("artists"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can update own projects" ON "public"."projects" FOR UPDATE USING ((EXISTS ( SELECT 1
   FROM "public"."artists"
  WHERE (("artists"."id" = "projects"."artist_id") AND ("artists"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can update their own artists" ON "public"."artists" FOR UPDATE USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view audio files for their artists" ON "public"."audio_files" FOR SELECT USING ((EXISTS ( SELECT 1
   FROM ("public"."audio_folders" "af"
     JOIN "public"."artists" "a" ON (("a"."id" = "af"."artist_id")))
  WHERE (("af"."id" = "audio_files"."folder_id") AND ("a"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can view audio folders for their artists" ON "public"."audio_folders" FOR SELECT USING ((EXISTS ( SELECT 1
   FROM "public"."artists" "a"
  WHERE (("a"."id" = "audio_folders"."artist_id") AND ("a"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can view own contacts" ON "public"."contacts" FOR SELECT USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view own payments" ON "public"."payments" FOR SELECT USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view own project files" ON "public"."project_files" FOR SELECT USING ((EXISTS ( SELECT 1
   FROM ("public"."projects"
     JOIN "public"."artists" ON (("artists"."id" = "projects"."artist_id")))
  WHERE (("projects"."id" = "project_files"."project_id") AND ("artists"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can view own projects" ON "public"."projects" FOR SELECT USING ((EXISTS ( SELECT 1
   FROM "public"."artists"
  WHERE (("artists"."id" = "projects"."artist_id") AND ("artists"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can view their own artists" ON "public"."artists" FOR SELECT USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view their own calculation contracts" ON "public"."royalty_calculation_contracts" FOR SELECT USING ((EXISTS ( SELECT 1
   FROM "public"."royalty_calculations" "rc"
  WHERE (("rc"."id" = "royalty_calculation_contracts"."calculation_id") AND ("rc"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users can view their own calculations" ON "public"."royalty_calculations" FOR SELECT USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view their own file shares" ON "public"."file_shares" FOR SELECT USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users can view their project audio links" ON "public"."project_audio_links" FOR SELECT USING ((EXISTS ( SELECT 1
   FROM ("public"."projects" "p"
     JOIN "public"."artists" "a" ON (("a"."id" = "p"."artist_id")))
  WHERE (("p"."id" = "project_audio_links"."project_id") AND ("a"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users manage own columns" ON "public"."board_columns" USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users manage own comments" ON "public"."board_task_comments" USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users manage own connections" ON "public"."integration_connections" USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users manage own drive mappings" ON "public"."drive_sync_mappings" USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users manage own notification settings" ON "public"."notification_settings" USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users manage own task artists" ON "public"."board_task_artists" USING ((EXISTS ( SELECT 1
   FROM "public"."board_tasks"
  WHERE (("board_tasks"."id" = "board_task_artists"."task_id") AND ("board_tasks"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users manage own task contracts" ON "public"."board_task_contracts" USING ((EXISTS ( SELECT 1
   FROM "public"."board_tasks"
  WHERE (("board_tasks"."id" = "board_task_contracts"."task_id") AND ("board_tasks"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users manage own task projects" ON "public"."board_task_projects" USING ((EXISTS ( SELECT 1
   FROM "public"."board_tasks"
  WHERE (("board_tasks"."id" = "board_task_projects"."task_id") AND ("board_tasks"."user_id" = "auth"."uid"())))));



CREATE POLICY "Users manage own tasks" ON "public"."board_tasks" USING (("auth"."uid"() = "user_id"));



CREATE POLICY "Users view own sync logs" ON "public"."sync_log" USING (("auth"."uid"() = "user_id"));



ALTER TABLE "public"."artists" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."audio_files" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."audio_folders" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."board_columns" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."board_task_artists" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."board_task_comments" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."board_task_contracts" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."board_task_projects" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."board_tasks" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."contacts" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."drive_sync_mappings" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."file_shares" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."integration_connections" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."mailing_list" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."notification_settings" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."payments" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."profiles" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."project_audio_links" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."project_files" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."projects" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."royalty_calculation_contracts" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."royalty_calculations" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."sync_log" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."user_onboarding" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."workspace_settings" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "ws_settings_user" ON "public"."workspace_settings" USING (("auth"."uid"() = "user_id"));





ALTER PUBLICATION "supabase_realtime" OWNER TO "postgres";


GRANT USAGE ON SCHEMA "public" TO "postgres";
GRANT USAGE ON SCHEMA "public" TO "anon";
GRANT USAGE ON SCHEMA "public" TO "authenticated";
GRANT USAGE ON SCHEMA "public" TO "service_role";

























































































































































GRANT ALL ON FUNCTION "public"."delete_orphan_calculation"() TO "anon";
GRANT ALL ON FUNCTION "public"."delete_orphan_calculation"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."delete_orphan_calculation"() TO "service_role";



GRANT ALL ON FUNCTION "public"."execute_readonly_query"("query_text" "text") TO "anon";
GRANT ALL ON FUNCTION "public"."execute_readonly_query"("query_text" "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."execute_readonly_query"("query_text" "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."handle_new_user"() TO "anon";
GRANT ALL ON FUNCTION "public"."handle_new_user"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."handle_new_user"() TO "service_role";



GRANT ALL ON FUNCTION "public"."handle_new_user_onboarding"() TO "anon";
GRANT ALL ON FUNCTION "public"."handle_new_user_onboarding"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."handle_new_user_onboarding"() TO "service_role";



GRANT ALL ON FUNCTION "public"."update_updated_at_column"() TO "anon";
GRANT ALL ON FUNCTION "public"."update_updated_at_column"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."update_updated_at_column"() TO "service_role";


















GRANT ALL ON TABLE "public"."artists" TO "anon";
GRANT ALL ON TABLE "public"."artists" TO "authenticated";
GRANT ALL ON TABLE "public"."artists" TO "service_role";



GRANT ALL ON TABLE "public"."audio_files" TO "anon";
GRANT ALL ON TABLE "public"."audio_files" TO "authenticated";
GRANT ALL ON TABLE "public"."audio_files" TO "service_role";



GRANT ALL ON TABLE "public"."audio_folders" TO "anon";
GRANT ALL ON TABLE "public"."audio_folders" TO "authenticated";
GRANT ALL ON TABLE "public"."audio_folders" TO "service_role";



GRANT ALL ON TABLE "public"."board_columns" TO "anon";
GRANT ALL ON TABLE "public"."board_columns" TO "authenticated";
GRANT ALL ON TABLE "public"."board_columns" TO "service_role";



GRANT ALL ON TABLE "public"."board_task_artists" TO "anon";
GRANT ALL ON TABLE "public"."board_task_artists" TO "authenticated";
GRANT ALL ON TABLE "public"."board_task_artists" TO "service_role";



GRANT ALL ON TABLE "public"."board_task_comments" TO "anon";
GRANT ALL ON TABLE "public"."board_task_comments" TO "authenticated";
GRANT ALL ON TABLE "public"."board_task_comments" TO "service_role";



GRANT ALL ON TABLE "public"."board_task_contracts" TO "anon";
GRANT ALL ON TABLE "public"."board_task_contracts" TO "authenticated";
GRANT ALL ON TABLE "public"."board_task_contracts" TO "service_role";



GRANT ALL ON TABLE "public"."board_task_projects" TO "anon";
GRANT ALL ON TABLE "public"."board_task_projects" TO "authenticated";
GRANT ALL ON TABLE "public"."board_task_projects" TO "service_role";



GRANT ALL ON TABLE "public"."board_tasks" TO "anon";
GRANT ALL ON TABLE "public"."board_tasks" TO "authenticated";
GRANT ALL ON TABLE "public"."board_tasks" TO "service_role";



GRANT ALL ON TABLE "public"."contacts" TO "anon";
GRANT ALL ON TABLE "public"."contacts" TO "authenticated";
GRANT ALL ON TABLE "public"."contacts" TO "service_role";



GRANT ALL ON TABLE "public"."drive_sync_mappings" TO "anon";
GRANT ALL ON TABLE "public"."drive_sync_mappings" TO "authenticated";
GRANT ALL ON TABLE "public"."drive_sync_mappings" TO "service_role";



GRANT ALL ON TABLE "public"."file_shares" TO "anon";
GRANT ALL ON TABLE "public"."file_shares" TO "authenticated";
GRANT ALL ON TABLE "public"."file_shares" TO "service_role";



GRANT ALL ON TABLE "public"."integration_connections" TO "anon";
GRANT ALL ON TABLE "public"."integration_connections" TO "authenticated";
GRANT ALL ON TABLE "public"."integration_connections" TO "service_role";



GRANT ALL ON TABLE "public"."mailing_list" TO "anon";
GRANT ALL ON TABLE "public"."mailing_list" TO "authenticated";
GRANT ALL ON TABLE "public"."mailing_list" TO "service_role";



GRANT ALL ON TABLE "public"."notification_settings" TO "anon";
GRANT ALL ON TABLE "public"."notification_settings" TO "authenticated";
GRANT ALL ON TABLE "public"."notification_settings" TO "service_role";



GRANT ALL ON TABLE "public"."payments" TO "anon";
GRANT ALL ON TABLE "public"."payments" TO "authenticated";
GRANT ALL ON TABLE "public"."payments" TO "service_role";



GRANT ALL ON TABLE "public"."profiles" TO "anon";
GRANT ALL ON TABLE "public"."profiles" TO "authenticated";
GRANT ALL ON TABLE "public"."profiles" TO "service_role";



GRANT ALL ON TABLE "public"."project_audio_links" TO "anon";
GRANT ALL ON TABLE "public"."project_audio_links" TO "authenticated";
GRANT ALL ON TABLE "public"."project_audio_links" TO "service_role";



GRANT ALL ON TABLE "public"."project_files" TO "anon";
GRANT ALL ON TABLE "public"."project_files" TO "authenticated";
GRANT ALL ON TABLE "public"."project_files" TO "service_role";



GRANT ALL ON TABLE "public"."projects" TO "anon";
GRANT ALL ON TABLE "public"."projects" TO "authenticated";
GRANT ALL ON TABLE "public"."projects" TO "service_role";



GRANT ALL ON TABLE "public"."royalty_calculation_contracts" TO "anon";
GRANT ALL ON TABLE "public"."royalty_calculation_contracts" TO "authenticated";
GRANT ALL ON TABLE "public"."royalty_calculation_contracts" TO "service_role";



GRANT ALL ON TABLE "public"."royalty_calculations" TO "anon";
GRANT ALL ON TABLE "public"."royalty_calculations" TO "authenticated";
GRANT ALL ON TABLE "public"."royalty_calculations" TO "service_role";



GRANT ALL ON TABLE "public"."sync_log" TO "anon";
GRANT ALL ON TABLE "public"."sync_log" TO "authenticated";
GRANT ALL ON TABLE "public"."sync_log" TO "service_role";



GRANT ALL ON TABLE "public"."user_onboarding" TO "anon";
GRANT ALL ON TABLE "public"."user_onboarding" TO "authenticated";
GRANT ALL ON TABLE "public"."user_onboarding" TO "service_role";



GRANT ALL ON TABLE "public"."workspace_settings" TO "anon";
GRANT ALL ON TABLE "public"."workspace_settings" TO "authenticated";
GRANT ALL ON TABLE "public"."workspace_settings" TO "service_role";









ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "service_role";































