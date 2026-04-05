-- Fix: collaborators_select_by_email_or_id policy was querying auth.users table
-- directly, which the authenticated role doesn't have SELECT permission on.
-- Replace with auth.jwt() ->> 'email' which reads from the JWT token.

DROP POLICY IF EXISTS "collaborators_select_by_email_or_id" ON registry_collaborators;

CREATE POLICY "collaborators_select_by_email_or_id" ON registry_collaborators
  FOR SELECT USING (
    LOWER(email) = LOWER(auth.jwt() ->> 'email')
    OR collaborator_user_id = auth.uid()
    OR invited_by = auth.uid()
  );

-- Fix: team_cards had no self-read policy — users couldn't read their own card
CREATE POLICY "Users can read own team card" ON team_cards
  FOR SELECT USING (user_id = auth.uid());
