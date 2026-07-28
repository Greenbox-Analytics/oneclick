-- ============================================================================
-- contact_submissions: public contact form (support tickets + general messages)
-- Spec: docs/superpowers/specs/2026-07-27-contact-page-design.md
--
-- Written only by POST /contact-submissions (src/backend/contact/router.py),
-- which uses the service-role client. The row is the durable source of truth;
-- the Resend notification to ops is best-effort on top of it.
-- ============================================================================

CREATE TABLE IF NOT EXISTS contact_submissions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  -- Shown to the submitter on the success screen; format MSN-XXXX-NNNN.
  -- UNIQUE so a generation collision surfaces as a constraint error the
  -- endpoint can retry, rather than two people sharing one reference.
  reference_id TEXT NOT NULL UNIQUE,
  mode TEXT NOT NULL CHECK (mode IN ('ticket', 'message')),

  name TEXT NOT NULL,
  email TEXT NOT NULL,
  subject TEXT NOT NULL,
  message TEXT NOT NULL,

  -- Mode-specific fields. Ticket mode collects product + account_email;
  -- message mode collects company + topic. The unused pair stays NULL.
  product TEXT,
  account_email TEXT,
  company TEXT,
  topic TEXT,

  -- NULL when submitted while logged out.
  user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,

  attachment_count INTEGER NOT NULL DEFAULT 0,
  -- Retained for the per-IP throttle window; not exposed to any client.
  client_ip TEXT,

  status TEXT NOT NULL CHECK (status IN ('new', 'in_progress', 'resolved', 'closed')) DEFAULT 'new',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_contact_submissions_status ON contact_submissions(status);
CREATE INDEX IF NOT EXISTS idx_contact_submissions_email ON contact_submissions(email);
-- Supports the throttle lookup: recent rows for one IP / email.
CREATE INDEX IF NOT EXISTS idx_contact_submissions_created_at ON contact_submissions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_contact_submissions_client_ip_created
  ON contact_submissions(client_ip, created_at DESC);

ALTER TABLE contact_submissions ENABLE ROW LEVEL SECURITY;

-- Deny-all read. Submissions contain third-party contact details and free-text
-- that may include account information; only the service role should see them.
DROP POLICY IF EXISTS "No public read on contact_submissions" ON contact_submissions;
CREATE POLICY "No public read on contact_submissions" ON contact_submissions FOR SELECT USING (false);

-- INSERT is intentionally NOT exposed via RLS. Anonymous browser INSERTs are
-- blocked; the backend endpoint inserts with the service role after running
-- the honeypot check, rate limit, and attachment validation.
