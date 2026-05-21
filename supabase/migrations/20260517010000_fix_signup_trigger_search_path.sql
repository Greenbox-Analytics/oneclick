-- ============================================================================
-- Fix: signups failing with "Database error saving new user".
--
-- create_default_subscription() and create_default_usage_counter() were
-- SECURITY DEFINER triggers on auth.users INSERT that referenced their
-- target tables unqualified and never set search_path.
--
-- Common myth: SECURITY DEFINER auto-switches search_path to the owner's.
-- It does not. The caller's search_path is inherited. GoTrue calls these
-- triggers as supabase_auth_admin, whose default search_path is `auth`.
-- Unqualified `subscriptions` resolved to `auth.subscriptions` (doesn't
-- exist) → trigger raised → whole auth.users INSERT rolled back → GoTrue
-- returned 500 "Database error saving new user".
--
-- Fix: add SET search_path TO 'public' AND fully schema-qualify the table
-- names (defense in depth — either alone would resolve this, both makes
-- the function robust against future role/path changes).
--
-- The bug originated in 20260509000001_subscription_foundation.sql. The
-- two other triggers added later (handle_new_user_onboarding,
-- process_pending_project_invites) correctly set search_path; these
-- two predated that pattern.
-- ============================================================================

CREATE OR REPLACE FUNCTION public.create_default_subscription()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
  INSERT INTO public.subscriptions (user_id, tier, status)
  VALUES (NEW.id, 'free', 'active')
  ON CONFLICT (user_id) DO NOTHING;
  RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.create_default_usage_counter()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
  INSERT INTO public.usage_counters (user_id) VALUES (NEW.id) ON CONFLICT DO NOTHING;
  RETURN NEW;
END;
$$;
