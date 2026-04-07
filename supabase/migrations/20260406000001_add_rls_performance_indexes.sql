-- =============================================================
-- RLS Performance Indexes
-- =============================================================
-- These composite indexes optimize the most frequently executed
-- RLS policy subqueries. Without them, RLS evaluation triggers
-- full table scans on every authenticated request.
-- =============================================================

-- HIGH PRIORITY: registry_collaborators
-- Used in work access RLS policies: "is this user a confirmed collaborator?"
-- Policies: works_registry, work_files, work_audio_links SELECT/UPDATE/DELETE
CREATE INDEX IF NOT EXISTS idx_registry_collaborators_user_status
  ON registry_collaborators (collaborator_user_id, status);

-- Used in work membership checks: "does this work have an active collaborator?"
CREATE INDEX IF NOT EXISTS idx_registry_collaborators_work_status
  ON registry_collaborators (work_id, status);

-- Used in team card visibility: "did this user invite any confirmed collaborators?"
CREATE INDEX IF NOT EXISTS idx_registry_collaborators_invited_status
  ON registry_collaborators (invited_by, status);

-- HIGH PRIORITY: project_members
-- Used in role-based access checks: "is this user an editor+ on this project?"
-- Policies: works_insert_editors, works_update_editors, project_files_insert
CREATE INDEX IF NOT EXISTS idx_project_members_project_role
  ON project_members (project_id, role);

-- Used in is_project_member() helper function: "is user a member of project?"
-- Existing indexes are (project_id) and (user_id) separately — composite is faster
CREATE INDEX IF NOT EXISTS idx_project_members_user_project
  ON project_members (user_id, project_id);

-- HIGH PRIORITY: works_registry
-- Used in work access through project membership: "does user own this work via project?"
CREATE INDEX IF NOT EXISTS idx_works_registry_project_user
  ON works_registry (project_id, user_id);

-- MEDIUM PRIORITY: Covering index for full work access check
-- Covers the complete RLS predicate: "is user a confirmed collaborator on work X?"
-- Avoids table lookup after index scan
CREATE INDEX IF NOT EXISTS idx_registry_collaborators_work_user_status
  ON registry_collaborators (work_id, collaborator_user_id, status);
