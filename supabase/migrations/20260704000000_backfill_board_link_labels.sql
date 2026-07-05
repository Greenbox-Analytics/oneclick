-- Backfill denormalized labels so pre-existing links show names for all team members.
-- The `label` columns already exist (20260629000003); this only fills them for rows created
-- before the link-write path started snapshotting labels. Idempotent via `label IS NULL`.
-- No schema change. No deploy-order constraint — it only fills display data.
BEGIN;

UPDATE board_task_artists   j SET label = a.name      FROM artists a        WHERE a.id = j.artist_id        AND j.label IS NULL;
UPDATE board_task_projects  j SET label = p.name      FROM projects p       WHERE p.id = j.project_id       AND j.label IS NULL;
UPDATE board_task_contracts j SET label = f.file_name FROM project_files f  WHERE f.id = j.project_file_id  AND j.label IS NULL;
UPDATE board_task_works     j SET label = w.title     FROM works_registry w WHERE w.id = j.work_id           AND j.label IS NULL;

COMMIT;
