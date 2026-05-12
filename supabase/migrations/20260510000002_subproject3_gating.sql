-- ============================================================================
-- Sub-project 3: storage cap RLS triggers + max_boards cleanup
-- Spec: docs/superpowers/specs/2026-05-10-pricing-tiers-sp3-design.md
-- ============================================================================

-- Drop the max_boards cap (no meaningful unit; see SP3 brainstorming)
UPDATE tier_entitlements SET max_boards = -1;

-- ----------------------------------------------------------------------------
-- effective_storage_cap: tier default + per-user override (with expiry filter)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION effective_storage_cap(p_user_id UUID)
RETURNS BIGINT AS $$
DECLARE
  user_tier TEXT;
  tier_cap BIGINT;
  override_cap BIGINT;
  override_expires TIMESTAMPTZ;
BEGIN
  IF p_user_id IS NULL THEN RETURN -1; END IF;

  SELECT tier INTO user_tier FROM subscriptions WHERE user_id = p_user_id;
  IF user_tier IS NULL THEN user_tier := 'free'; END IF;

  SELECT max_storage_bytes INTO tier_cap FROM tier_entitlements WHERE tier = user_tier;
  IF tier_cap IS NULL THEN tier_cap := 1073741824; END IF;  -- defensive Free default

  SELECT max_storage_bytes, expires_at INTO override_cap, override_expires
    FROM tier_overrides WHERE user_id = p_user_id;

  IF override_cap IS NOT NULL
     AND (override_expires IS NULL OR override_expires >= now()) THEN
    RETURN override_cap;
  END IF;

  RETURN tier_cap;
END;
$$ LANGUAGE plpgsql STABLE SECURITY DEFINER;

-- ----------------------------------------------------------------------------
-- BEFORE INSERT trigger on project_files: reject if owner over cap
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION enforce_storage_cap_pf()
RETURNS TRIGGER AS $$
DECLARE
  owner_id UUID;
  cap BIGINT;
  current_usage BIGINT;
BEGIN
  IF NEW.file_size IS NULL OR NEW.file_size <= 0 THEN
    RETURN NEW;
  END IF;

  SELECT a.user_id INTO owner_id
    FROM projects p JOIN artists a ON a.id = p.artist_id
    WHERE p.id = NEW.project_id;
  IF owner_id IS NULL THEN
    RETURN NEW;
  END IF;

  cap := effective_storage_cap(owner_id);
  IF cap = -1 THEN
    RETURN NEW;
  END IF;

  SELECT total_storage_bytes INTO current_usage
    FROM usage_counters WHERE user_id = owner_id;
  IF current_usage IS NULL THEN current_usage := 0; END IF;

  IF current_usage + NEW.file_size > cap THEN
    RAISE EXCEPTION 'Storage cap exceeded for user %: % + % > % bytes',
      owner_id, current_usage, NEW.file_size, cap
      USING ERRCODE = '23514';
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS enforce_storage_cap_before_pf ON project_files;
CREATE TRIGGER enforce_storage_cap_before_pf
  BEFORE INSERT ON project_files
  FOR EACH ROW EXECUTE FUNCTION enforce_storage_cap_pf();

-- ----------------------------------------------------------------------------
-- BEFORE INSERT trigger on audio_files: reject if owner over cap
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION enforce_storage_cap_af()
RETURNS TRIGGER AS $$
DECLARE
  owner_id UUID;
  cap BIGINT;
  current_usage BIGINT;
BEGIN
  IF NEW.file_size IS NULL OR NEW.file_size <= 0 THEN
    RETURN NEW;
  END IF;

  SELECT a.user_id INTO owner_id
    FROM audio_folders afo JOIN artists a ON a.id = afo.artist_id
    WHERE afo.id = NEW.folder_id;
  IF owner_id IS NULL THEN
    RETURN NEW;
  END IF;

  cap := effective_storage_cap(owner_id);
  IF cap = -1 THEN
    RETURN NEW;
  END IF;

  SELECT total_storage_bytes INTO current_usage
    FROM usage_counters WHERE user_id = owner_id;
  IF current_usage IS NULL THEN current_usage := 0; END IF;

  IF current_usage + NEW.file_size > cap THEN
    RAISE EXCEPTION 'Storage cap exceeded for user %: % + % > % bytes',
      owner_id, current_usage, NEW.file_size, cap
      USING ERRCODE = '23514';
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS enforce_storage_cap_before_af ON audio_files;
CREATE TRIGGER enforce_storage_cap_before_af
  BEFORE INSERT ON audio_files
  FOR EACH ROW EXECUTE FUNCTION enforce_storage_cap_af();
