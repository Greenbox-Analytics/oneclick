-- ============================================================================
-- Subscription foundation (Sub-project 1 of Pricing Tiers initiative)
-- Spec: docs/superpowers/specs/2026-05-09-pricing-tiers-foundation-design.md
--
-- Storage is OWNER-SCOPED: usage_counters.total_storage_bytes for user X =
-- bytes in projects X owns (via artists.user_id). Triggers fire on the
-- application tables (project_files, audio_files) which already store
-- file_size, using incremental +=/-= updates. recalc_user_storage(uuid)
-- is retained as a self-healing repair function.
--
-- tier_entitlements has PUBLIC RLS so the unauthenticated Pricing page
-- (Sub-project 2) can render tier comparisons without auth.
--
-- Convention: -1 in any "max_*" column = unlimited.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- subscriptions
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS subscriptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
  tier TEXT NOT NULL CHECK (tier IN ('free', 'pro')) DEFAULT 'free',
  status TEXT NOT NULL CHECK (status IN ('active', 'canceled', 'past_due', 'trialing')) DEFAULT 'active',
  stripe_customer_id TEXT UNIQUE,
  stripe_subscription_id TEXT UNIQUE,
  stripe_price_id TEXT,
  current_period_start TIMESTAMPTZ,
  current_period_end TIMESTAMPTZ,
  cancel_at_period_end BOOLEAN DEFAULT false,
  canceled_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id ON subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_stripe_customer_id ON subscriptions(stripe_customer_id);

CREATE OR REPLACE FUNCTION create_default_subscription()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO subscriptions (user_id, tier, status)
  VALUES (NEW.id, 'free', 'active')
  ON CONFLICT (user_id) DO NOTHING;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created_subscription ON auth.users;
CREATE TRIGGER on_auth_user_created_subscription
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION create_default_subscription();

ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users read own subscription" ON subscriptions;
CREATE POLICY "Users read own subscription" ON subscriptions
  FOR SELECT USING (auth.uid() = user_id);

-- ----------------------------------------------------------------------------
-- tier_entitlements (PUBLIC RLS for the Pricing page)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tier_entitlements (
  tier TEXT PRIMARY KEY CHECK (tier IN ('free', 'pro')),
  max_artists INTEGER NOT NULL,
  max_projects INTEGER NOT NULL,
  max_boards INTEGER NOT NULL,
  max_tasks INTEGER NOT NULL,
  max_storage_bytes BIGINT NOT NULL,
  max_split_sheets_per_month INTEGER NOT NULL,
  zoe_enabled BOOLEAN NOT NULL DEFAULT false,
  oneclick_enabled BOOLEAN NOT NULL DEFAULT false,
  registry_enabled BOOLEAN NOT NULL DEFAULT false,
  integrations_allowed JSONB NOT NULL DEFAULT '[]'::jsonb,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO tier_entitlements (tier, max_artists, max_projects, max_boards, max_tasks,
                               max_storage_bytes, max_split_sheets_per_month,
                               zoe_enabled, oneclick_enabled, registry_enabled,
                               integrations_allowed)
VALUES
  ('free', 3, 3, 3, 50, 1073741824, 5, false, false, false, '["google_drive"]'::jsonb),
  ('pro', -1, -1, -1, -1, -1, -1, true, true, true,
   '["google_drive", "slack", "notion", "monday"]'::jsonb)
ON CONFLICT (tier) DO NOTHING;

ALTER TABLE tier_entitlements ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Public reads tier_entitlements" ON tier_entitlements;
CREATE POLICY "Public reads tier_entitlements" ON tier_entitlements
  FOR SELECT USING (true);

-- ----------------------------------------------------------------------------
-- usage_counters (owner-scoped storage)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS usage_counters (
  user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  total_storage_bytes BIGINT NOT NULL DEFAULT 0,
  split_sheets_this_period INTEGER NOT NULL DEFAULT 0,
  period_start TIMESTAMPTZ NOT NULL DEFAULT now(),
  period_end TIMESTAMPTZ NOT NULL DEFAULT (now() + INTERVAL '1 month'),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION create_default_usage_counter()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO usage_counters (user_id) VALUES (NEW.id) ON CONFLICT DO NOTHING;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created_usage_counter ON auth.users;
CREATE TRIGGER on_auth_user_created_usage_counter
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION create_default_usage_counter();

-- Repair function: recompute total_storage_bytes from scratch for one user.
-- Sums file_size from project_files (via projects -> artists -> user_id)
-- and audio_files (via audio_folders -> artists -> user_id).
CREATE OR REPLACE FUNCTION recalc_user_storage(p_user_id UUID)
RETURNS VOID AS $$
DECLARE
  total BIGINT;
BEGIN
  IF p_user_id IS NULL THEN RETURN; END IF;
  SELECT
    COALESCE((SELECT SUM(pf.file_size)
              FROM project_files pf
              JOIN projects p ON p.id = pf.project_id
              JOIN artists a ON a.id = p.artist_id
              WHERE a.user_id = p_user_id), 0)
    +
    COALESCE((SELECT SUM(af.file_size)
              FROM audio_files af
              JOIN audio_folders afo ON afo.id = af.folder_id
              JOIN artists a ON a.id = afo.artist_id
              WHERE a.user_id = p_user_id), 0)
  INTO total;

  UPDATE usage_counters
  SET total_storage_bytes = total, updated_at = now()
  WHERE user_id = p_user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Helper: resolve owner_id from a project_files row.
CREATE OR REPLACE FUNCTION _project_owner_for_pf(pf_project_id UUID)
RETURNS UUID AS $$
  SELECT a.user_id
  FROM projects p
  JOIN artists a ON a.id = p.artist_id
  WHERE p.id = pf_project_id;
$$ LANGUAGE sql STABLE SECURITY DEFINER;

-- Helper: resolve owner_id from an audio_files row (via audio_folders).
CREATE OR REPLACE FUNCTION _audio_owner_for_af(af_folder_id UUID)
RETURNS UUID AS $$
  SELECT a.user_id
  FROM audio_folders afo
  JOIN artists a ON a.id = afo.artist_id
  WHERE afo.id = af_folder_id;
$$ LANGUAGE sql STABLE SECURITY DEFINER;

-- Incremental storage trigger on project_files.
-- Cross-owner UPDATE handling: if NEW.project_id <> OLD.project_id and the two projects
-- have different owners, decrement OLD owner's counter AND increment NEW owner's counter.
-- Same-project UPDATE: simple delta on the single owner.
CREATE OR REPLACE FUNCTION trigger_storage_pf_change()
RETURNS TRIGGER AS $$
DECLARE
  owner_id UUID;
  old_owner UUID;
  new_owner UUID;
BEGIN
  IF TG_OP = 'INSERT' THEN
    owner_id := _project_owner_for_pf(NEW.project_id);
    IF owner_id IS NOT NULL AND NEW.file_size IS NOT NULL THEN
      UPDATE usage_counters
      SET total_storage_bytes = total_storage_bytes + NEW.file_size,
          updated_at = now()
      WHERE user_id = owner_id;
    END IF;
  ELSIF TG_OP = 'DELETE' THEN
    owner_id := _project_owner_for_pf(OLD.project_id);
    IF owner_id IS NOT NULL AND OLD.file_size IS NOT NULL THEN
      UPDATE usage_counters
      SET total_storage_bytes = GREATEST(total_storage_bytes - OLD.file_size, 0),
          updated_at = now()
      WHERE user_id = owner_id;
    END IF;
  ELSIF TG_OP = 'UPDATE' THEN
    -- Cross-project move (potentially cross-owner): handle both sides explicitly.
    IF NEW.project_id IS DISTINCT FROM OLD.project_id THEN
      old_owner := _project_owner_for_pf(OLD.project_id);
      new_owner := _project_owner_for_pf(NEW.project_id);
      IF old_owner IS NOT NULL AND OLD.file_size IS NOT NULL THEN
        UPDATE usage_counters
        SET total_storage_bytes = GREATEST(total_storage_bytes - OLD.file_size, 0),
            updated_at = now()
        WHERE user_id = old_owner;
      END IF;
      IF new_owner IS NOT NULL AND NEW.file_size IS NOT NULL THEN
        UPDATE usage_counters
        SET total_storage_bytes = total_storage_bytes + NEW.file_size,
            updated_at = now()
        WHERE user_id = new_owner;
      END IF;
    ELSE
      -- Same project, single owner: net delta on file_size.
      owner_id := _project_owner_for_pf(NEW.project_id);
      IF owner_id IS NOT NULL THEN
        UPDATE usage_counters
        SET total_storage_bytes = GREATEST(
              total_storage_bytes + COALESCE(NEW.file_size, 0) - COALESCE(OLD.file_size, 0), 0),
            updated_at = now()
        WHERE user_id = owner_id;
      END IF;
    END IF;
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_project_files_change_storage ON project_files;
CREATE TRIGGER on_project_files_change_storage
  AFTER INSERT OR UPDATE OR DELETE ON project_files
  FOR EACH ROW EXECUTE FUNCTION trigger_storage_pf_change();

-- Incremental storage trigger on audio_files.
-- Cross-owner UPDATE: if folder_id changes to a folder under a different artist (different owner),
-- decrement old owner + increment new owner. Same-folder UPDATE: simple delta.
CREATE OR REPLACE FUNCTION trigger_storage_af_change()
RETURNS TRIGGER AS $$
DECLARE
  owner_id UUID;
  old_owner UUID;
  new_owner UUID;
BEGIN
  IF TG_OP = 'INSERT' THEN
    owner_id := _audio_owner_for_af(NEW.folder_id);
    IF owner_id IS NOT NULL AND NEW.file_size IS NOT NULL THEN
      UPDATE usage_counters
      SET total_storage_bytes = total_storage_bytes + NEW.file_size,
          updated_at = now()
      WHERE user_id = owner_id;
    END IF;
  ELSIF TG_OP = 'DELETE' THEN
    owner_id := _audio_owner_for_af(OLD.folder_id);
    IF owner_id IS NOT NULL AND OLD.file_size IS NOT NULL THEN
      UPDATE usage_counters
      SET total_storage_bytes = GREATEST(total_storage_bytes - OLD.file_size, 0),
          updated_at = now()
      WHERE user_id = owner_id;
    END IF;
  ELSIF TG_OP = 'UPDATE' THEN
    -- Cross-folder move (potentially cross-owner): handle both sides.
    IF NEW.folder_id IS DISTINCT FROM OLD.folder_id THEN
      old_owner := _audio_owner_for_af(OLD.folder_id);
      new_owner := _audio_owner_for_af(NEW.folder_id);
      IF old_owner IS NOT NULL AND OLD.file_size IS NOT NULL THEN
        UPDATE usage_counters
        SET total_storage_bytes = GREATEST(total_storage_bytes - OLD.file_size, 0),
            updated_at = now()
        WHERE user_id = old_owner;
      END IF;
      IF new_owner IS NOT NULL AND NEW.file_size IS NOT NULL THEN
        UPDATE usage_counters
        SET total_storage_bytes = total_storage_bytes + NEW.file_size,
            updated_at = now()
        WHERE user_id = new_owner;
      END IF;
    ELSE
      -- Same folder, single owner: net delta on file_size.
      owner_id := _audio_owner_for_af(NEW.folder_id);
      IF owner_id IS NOT NULL THEN
        UPDATE usage_counters
        SET total_storage_bytes = GREATEST(
              total_storage_bytes + COALESCE(NEW.file_size, 0) - COALESCE(OLD.file_size, 0), 0),
            updated_at = now()
        WHERE user_id = owner_id;
      END IF;
    END IF;
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_audio_files_change_storage ON audio_files;
CREATE TRIGGER on_audio_files_change_storage
  AFTER INSERT OR UPDATE OR DELETE ON audio_files
  FOR EACH ROW EXECUTE FUNCTION trigger_storage_af_change();

ALTER TABLE usage_counters ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users read own usage" ON usage_counters;
CREATE POLICY "Users read own usage" ON usage_counters
  FOR SELECT USING (auth.uid() = user_id);

-- ----------------------------------------------------------------------------
-- tier_overrides (NO granted_by column — comes back with admin UI in Sub-project 3)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tier_overrides (
  user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  max_artists INTEGER,
  max_projects INTEGER,
  max_boards INTEGER,
  max_tasks INTEGER,
  max_storage_bytes BIGINT,
  max_split_sheets_per_month INTEGER,
  zoe_enabled BOOLEAN,
  oneclick_enabled BOOLEAN,
  registry_enabled BOOLEAN,
  integrations_allowed JSONB,
  reason TEXT,
  granted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ
);

ALTER TABLE tier_overrides ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users read own override" ON tier_overrides;
CREATE POLICY "Users read own override" ON tier_overrides
  FOR SELECT USING (auth.uid() = user_id);
