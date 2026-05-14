-- Add welcome_email_sent_at to profiles to gate the one-time welcome email
-- sent immediately after Google OAuth sign-up.
ALTER TABLE public.profiles
  ADD COLUMN welcome_email_sent_at TIMESTAMPTZ NULL;

-- Backfill existing users so they don't receive a retroactive welcome on next sign-in.
UPDATE public.profiles
  SET welcome_email_sent_at = now()
  WHERE welcome_email_sent_at IS NULL;
