-- Rename `industry` → `role` on both tables.
-- The column captures the user's professional role, not a sector, so the name
-- was misleading. UI strings have already been updated; this migration aligns
-- the schema.

ALTER TABLE public.team_cards RENAME COLUMN industry TO role;

-- profiles.industry was added via the Supabase dashboard in some environments
-- and is absent from the on-disk migration history. Rename only if present.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'profiles'
      AND column_name = 'industry'
  ) THEN
    ALTER TABLE public.profiles RENAME COLUMN industry TO role;
  END IF;
END $$;

-- team_cards.visible_fields is a jsonb array of field keys. Existing rows may
-- contain the literal "industry"; rewrite each array in place so the UI keeps
-- matching after the rename.
UPDATE public.team_cards
SET visible_fields = (
  SELECT jsonb_agg(
    CASE
      WHEN elem = to_jsonb('industry'::text) THEN to_jsonb('role'::text)
      ELSE elem
    END
  )
  FROM jsonb_array_elements(visible_fields::jsonb) AS elem
)
WHERE visible_fields::jsonb @> '["industry"]'::jsonb;
