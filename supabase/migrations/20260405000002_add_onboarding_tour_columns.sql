-- Add onboarding tour tracking columns for Registry, Project Detail, and Profile pages
ALTER TABLE user_onboarding
  ADD COLUMN registry_completed boolean NOT NULL DEFAULT false,
  ADD COLUMN project_detail_completed boolean NOT NULL DEFAULT false,
  ADD COLUMN profile_completed boolean NOT NULL DEFAULT false;
