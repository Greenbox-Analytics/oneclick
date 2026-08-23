-- ============================================================================
-- Workspace scoping, follow-up 1/2: the six registry sub-tables that hang off a
-- work but never got a team policy.
--
-- docs/licensing.md states the rule plainly: "If you add a table that hangs off
-- an artist, add one FOR ALL policy that calls can_access_artist -- that is the
-- whole integration." 20260803000002 did that for the ten artist-scoped tables
-- and 20260805000002 caught notes/note_folders. These six were missed, and they
-- are precisely the tables a work's detail page reads.
--
-- The effect before this migration: `works_team_artist` (20260803000002) is a
-- FOR ALL policy over can_access_artist, so an org member could SELECT the work
-- row -- and then get nothing for its splits, licences, agreements or files,
-- because every sub-table still gated on work_role()/project_members alone. A
-- visible work that opens empty.
--
-- Reach is defense-in-depth today (the API path runs service-role and is gated
-- by registry/access.py's get_work_access, which gained the same artist branch
-- in this change). It matters for any DIRECT PostgREST read, and it keeps the
-- SQL layer and the Python mirror answering the same question -- which is the
-- entire reason artist_access.py exists.
--
-- These are ADDITIVE. Permissive policies OR, so nothing that could be read
-- before becomes unreadable; the existing work_role()/project_members policies
-- from 20260614000100_registry_permissions_rls.sql are untouched.
--
-- QA: supabase/qa/gates_team_artists.sql exercises the artist-ownership layer
-- these policies extend; dedicated gates for the six sub-tables are a noted
-- follow-up in that file's header.
-- ============================================================================
BEGIN;

-- Every one of these hangs off works_registry, which carries artist_id
-- directly, so each policy is the same single hop. A helper is deliberately NOT
-- introduced: can_access_artist is already the one predicate, and wrapping it
-- would add a second name for the same question.

-- ------------------------------------------------------------ ownership_stakes
CREATE POLICY "stakes_team_artist" ON ownership_stakes
  FOR ALL USING (
    EXISTS (SELECT 1 FROM works_registry w
             WHERE w.id = ownership_stakes.work_id
               AND can_access_artist(w.artist_id, auth.uid()))
  ) WITH CHECK (
    EXISTS (SELECT 1 FROM works_registry w
             WHERE w.id = ownership_stakes.work_id
               AND can_access_artist(w.artist_id, auth.uid()))
  );

-- ----------------------------------------------------------- licensing_rights
CREATE POLICY "licenses_team_artist" ON licensing_rights
  FOR ALL USING (
    EXISTS (SELECT 1 FROM works_registry w
             WHERE w.id = licensing_rights.work_id
               AND can_access_artist(w.artist_id, auth.uid()))
  ) WITH CHECK (
    EXISTS (SELECT 1 FROM works_registry w
             WHERE w.id = licensing_rights.work_id
               AND can_access_artist(w.artist_id, auth.uid()))
  );

-- -------------------------------------------------------- registry_agreements
-- Agreements are append-only elsewhere (agreements_select/_insert, no update or
-- delete policy exists), so FOR ALL here grants read + insert and nothing more:
-- there is no UPDATE/DELETE path on this table to widen.
CREATE POLICY "agreements_team_artist" ON registry_agreements
  FOR ALL USING (
    EXISTS (SELECT 1 FROM works_registry w
             WHERE w.id = registry_agreements.work_id
               AND can_access_artist(w.artist_id, auth.uid()))
  ) WITH CHECK (
    EXISTS (SELECT 1 FROM works_registry w
             WHERE w.id = registry_agreements.work_id
               AND can_access_artist(w.artist_id, auth.uid()))
  );

-- ----------------------------------------------------- registry_access_grants
-- Grants describe what a work-only collaborator may see. An org member managing
-- the work needs to read and revoke them, the same way a project admin does.
CREATE POLICY "grants_team_artist" ON registry_access_grants
  FOR ALL USING (
    EXISTS (SELECT 1 FROM registry_collaborators c
              JOIN works_registry w ON w.id = c.work_id
             WHERE c.id = registry_access_grants.collaborator_id
               AND can_access_artist(w.artist_id, auth.uid()))
  ) WITH CHECK (
    EXISTS (SELECT 1 FROM registry_collaborators c
              JOIN works_registry w ON w.id = c.work_id
             WHERE c.id = registry_access_grants.collaborator_id
               AND can_access_artist(w.artist_id, auth.uid()))
  );

-- ------------------------------------------------------------------ work_files
CREATE POLICY "work_files_team_artist" ON work_files
  FOR ALL USING (
    EXISTS (SELECT 1 FROM works_registry w
             WHERE w.id = work_files.work_id
               AND can_access_artist(w.artist_id, auth.uid()))
  ) WITH CHECK (
    EXISTS (SELECT 1 FROM works_registry w
             WHERE w.id = work_files.work_id
               AND can_access_artist(w.artist_id, auth.uid()))
  );

-- ------------------------------------------------------------ work_audio_links
CREATE POLICY "work_audio_links_team_artist" ON work_audio_links
  FOR ALL USING (
    EXISTS (SELECT 1 FROM works_registry w
             WHERE w.id = work_audio_links.work_id
               AND can_access_artist(w.artist_id, auth.uid()))
  ) WITH CHECK (
    EXISTS (SELECT 1 FROM works_registry w
             WHERE w.id = work_audio_links.work_id
               AND can_access_artist(w.artist_id, auth.uid()))
  );

COMMIT;
