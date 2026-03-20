ALTER TABLE board_tasks ADD COLUMN completed_at TIMESTAMPTZ DEFAULT NULL;

-- Backfill existing Done tasks
UPDATE board_tasks SET completed_at = updated_at
WHERE column_id IN (SELECT id FROM board_columns WHERE LOWER(title) = 'done')
AND completed_at IS NULL;
