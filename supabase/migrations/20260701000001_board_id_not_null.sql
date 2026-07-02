-- Team Boards Phase 3a — lock board_id NOT NULL + de-dup personal boards. Apply ONLY after
-- Task 1 code (which lands board_id on all inserts) is deployed.
BEGIN;

-- Personal-board uniqueness for ARTIST-scoped boards only. Excludes the NULL-artist Personal
-- board (so this never collides with boards.artist_id's ON DELETE SET NULL when an artist is
-- deleted). ensure_personal_board tolerates the residual NULL-board race via re-read.
CREATE UNIQUE INDEX IF NOT EXISTS uq_boards_personal_artist
  ON boards (owner_id, artist_id) WHERE team_id IS NULL AND artist_id IS NOT NULL;

-- Re-backfill any rows created NULL between the Phase-1 apply and Task-1 deploy.
INSERT INTO boards (team_id, owner_id, name, artist_id)
SELECT NULL, x.user_id, COALESCE(a.name, 'Personal'), x.artist_id
FROM (
  SELECT DISTINCT user_id, artist_id FROM board_columns WHERE board_id IS NULL
  UNION
  SELECT DISTINCT user_id, artist_id FROM board_tasks WHERE board_id IS NULL AND column_id IS NULL
) x
LEFT JOIN artists a ON a.id = x.artist_id
WHERE NOT EXISTS (
  SELECT 1 FROM boards b WHERE b.owner_id = x.user_id AND b.team_id IS NULL
    AND b.artist_id IS NOT DISTINCT FROM x.artist_id
);

UPDATE board_columns c SET board_id = b.id
FROM boards b
WHERE c.board_id IS NULL AND b.owner_id = c.user_id AND b.team_id IS NULL
  AND b.artist_id IS NOT DISTINCT FROM c.artist_id;

UPDATE board_tasks t SET board_id = col.board_id
FROM board_columns col WHERE t.board_id IS NULL AND t.column_id = col.id;

UPDATE board_tasks t SET board_id = b.id
FROM boards b
WHERE t.board_id IS NULL AND t.column_id IS NULL AND b.owner_id = t.user_id AND b.team_id IS NULL
  AND b.artist_id IS NOT DISTINCT FROM t.artist_id;

DO $$
DECLARE n_cols INT; n_tasks INT;
BEGIN
  SELECT count(*) INTO n_cols  FROM board_columns WHERE board_id IS NULL;
  SELECT count(*) INTO n_tasks FROM board_tasks   WHERE board_id IS NULL;
  IF n_cols > 0 OR n_tasks > 0 THEN
    RAISE EXCEPTION 'board_id still NULL: % cols, % tasks — is Task 1 deployed?', n_cols, n_tasks;
  END IF;
END $$;

ALTER TABLE board_columns ALTER COLUMN board_id SET NOT NULL;
ALTER TABLE board_tasks   ALTER COLUMN board_id SET NOT NULL;

COMMIT;
