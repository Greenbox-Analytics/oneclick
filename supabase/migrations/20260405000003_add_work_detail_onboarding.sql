-- Add onboarding tour tracking column for Work Detail page
ALTER TABLE user_onboarding
  ADD COLUMN work_detail_completed boolean NOT NULL DEFAULT false;
