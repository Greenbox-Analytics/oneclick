-- Credentials vault: per-artist login/password storage for external platforms
-- (DistroKid, Spotify for Artists, ASCAP/BMI, etc.).
-- Passwords are AES-256-GCM encrypted server-side by the backend before insert;
-- this table stores only ciphertext. Decryption requires the backend key and a
-- re-auth with the Msanii password via POST /credentials/{id}/reveal.

CREATE TABLE IF NOT EXISTS artist_credentials (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  artist_id UUID NOT NULL REFERENCES artists(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  platform_name TEXT NOT NULL,
  login_identifier TEXT NOT NULL,
  password_ciphertext TEXT NOT NULL,
  url TEXT,
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_artist_credentials_artist ON artist_credentials(artist_id);
CREATE INDEX IF NOT EXISTS idx_artist_credentials_user ON artist_credentials(user_id);

ALTER TABLE artist_credentials ENABLE ROW LEVEL SECURITY;

CREATE POLICY "artist_credentials_owner_select"
  ON artist_credentials FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "artist_credentials_owner_insert"
  ON artist_credentials FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "artist_credentials_owner_update"
  ON artist_credentials FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "artist_credentials_owner_delete"
  ON artist_credentials FOR DELETE
  USING (auth.uid() = user_id);

-- Keep updated_at fresh on row modification
CREATE OR REPLACE FUNCTION update_artist_credentials_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_artist_credentials_updated_at
  BEFORE UPDATE ON artist_credentials
  FOR EACH ROW
  EXECUTE FUNCTION update_artist_credentials_updated_at();
