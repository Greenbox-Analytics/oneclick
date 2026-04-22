-- Track the outcome of each Resend send attempt so the UI can surface
-- delivery failures and offer a retry path without recreating the invite.

ALTER TABLE public.pending_project_invites
ADD COLUMN IF NOT EXISTS last_email_error TEXT,
ADD COLUMN IF NOT EXISTS last_email_attempt_at TIMESTAMPTZ;
