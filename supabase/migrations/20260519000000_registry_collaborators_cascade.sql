-- Follow-up to 20260518000000_fix_user_delete_cascades.sql, which patched
-- project_members.invited_by, pending_project_invites.invited_by, and
-- royalty_calculations.user_id, but missed the two registry_collaborators
-- FKs to auth.users. Without these, account deletion fails for any user
-- who has ever invited or been invited as a registry collaborator.
--
-- Source: 20260329000000_create_rights_registry.sql:110-111 — both columns
-- were created with bare `references auth.users(id)` (defaults to NO ACTION).

-- 1. registry_collaborators.invited_by  (was NO ACTION → CASCADE)
-- NOT NULL, so SET NULL is not an option. The invitation row is meaningless
-- once the inviter is gone, so cascading the delete is the right call.
ALTER TABLE public.registry_collaborators
  DROP CONSTRAINT IF EXISTS registry_collaborators_invited_by_fkey;
ALTER TABLE public.registry_collaborators
  ADD CONSTRAINT registry_collaborators_invited_by_fkey
    FOREIGN KEY (invited_by)
    REFERENCES auth.users(id)
    ON DELETE CASCADE;

-- 2. registry_collaborators.collaborator_user_id  (was NO ACTION → SET NULL)
-- Nullable. We preserve the row (so the inviter still sees historical
-- "X was invited" data) but null out the link to the deleted user.
ALTER TABLE public.registry_collaborators
  DROP CONSTRAINT IF EXISTS registry_collaborators_collaborator_user_id_fkey;
ALTER TABLE public.registry_collaborators
  ADD CONSTRAINT registry_collaborators_collaborator_user_id_fkey
    FOREIGN KEY (collaborator_user_id)
    REFERENCES auth.users(id)
    ON DELETE SET NULL;
