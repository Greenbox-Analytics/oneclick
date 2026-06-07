-- Backfill owner membership for projects created before the POST /projects
-- owner-insert fix. Projects created via the backend service-role path
-- (Zoe / OneClick) have zero project_members rows, because the
-- auto_create_project_owner trigger only fires when auth.uid() is non-NULL.
-- Without an owner row, the creator can't upload files, manage members, or
-- delete the project (the UI gates those on project_members role).
--
-- ORDER: apply this ONLY AFTER the POST /projects owner-insert fix is deployed
-- to this environment. If applied earlier, any Zoe/OneClick project created in
-- the interim is still made owner-less; re-running this backfill later is safe
-- (idempotent) and will pick them up.
--
-- The rightful owner is the artist's owning user: artists are private to their
-- creator (artists.user_id), and every project belongs to exactly one artist.
--
-- Idempotent: only inserts for projects that have no owner yet; ON CONFLICT
-- DO NOTHING guards the unique indexes (one_owner_per_project and the
-- UNIQUE(project_id, user_id) constraint).
--
-- Residual: a project whose artist's user is already a NON-owner member (and has
-- no owner) is skipped via ON CONFLICT and stays owner-less — expected and rare
-- (owner-less projects normally have zero members). Likewise, projects whose
-- artist has user_id IS NULL are skipped (no rightful owner to assign). Confirm
-- the remainder after applying:
--   SELECT count(*) FROM projects p
--   WHERE NOT EXISTS (SELECT 1 FROM project_members pm
--                     WHERE pm.project_id = p.id AND pm.role = 'owner');

INSERT INTO project_members (project_id, user_id, role)
SELECT p.id, a.user_id, 'owner'
FROM projects p
JOIN artists a ON a.id = p.artist_id
WHERE a.user_id IS NOT NULL
  AND NOT EXISTS (
    SELECT 1 FROM project_members pm
    WHERE pm.project_id = p.id AND pm.role = 'owner'
  )
ON CONFLICT DO NOTHING;
