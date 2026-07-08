#!/usr/bin/env bash
# Seed all GSM secrets the backend workflows reference.
#
# Usage:
#   GCP_PROJECT_ID=msanii-prod ./scripts/seed-gsm-secrets.sh
#
# This script is idempotent (safe to re-run). It:
#   1. Creates each secret if it doesn't exist (errors-on-create are ignored).
#   2. Adds a new version when you provide a value (otherwise leaves untouched).
#   3. Grants the Cloud Run runtime service account read access to each.
#
# After running, populate the secret values manually:
#   echo -n "real-value" | gcloud secrets versions add SECRET_NAME --data-file=-
# or use the GCP Console (Security → Secret Manager).

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID=your-gcp-project}"
# The default Cloud Run runtime SA. Override RUNTIME_SA if you use a custom one.
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="${RUNTIME_SA:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"

echo "Project:     $PROJECT_ID"
echo "Runtime SA:  $RUNTIME_SA"
echo

# Secrets shared across dev + prod (same value).
SHARED_SECRETS=(
  VITE_SUPABASE_URL
  VITE_SUPABASE_ANON_KEY
  VITE_SUPABASE_SECRET_KEY
  INTEGRATION_ENCRYPTION_KEY
  INTEGRATION_OAUTH_STATE_SECRET
  CREDENTIALS_AES_KEY
  GOOGLE_DRIVE_CLIENT_ID
  GOOGLE_DRIVE_CLIENT_SECRET
  SLACK_CLIENT_ID
  SLACK_CLIENT_SECRET
  OPENAI_API_KEY
  OPENAI_LLM_MODEL
  OPENAI_LLM_MODEL_LARGE
  RESEND_API_KEY
  RESEND_FROM_EMAIL
  ADMIN_EMAILS
  # Stripe price IDs: shared because dev + prod currently run against the same
  # Stripe mode (test). Move to ENV_SCOPED_SECRETS once prod flips to live mode.
  STRIPE_PRICE_MONTHLY
  STRIPE_PRICE_ANNUAL
  # PostHog: shared because dev + prod report to the same PostHog project.
  # If you split into separate projects per env, move these three to
  # ENV_SCOPED_SECRETS. Tip: add an `env` person property at identify time so
  # you can still distinguish dev vs prod traffic in the shared project.
  POSTHOG_PROJECT_TOKEN
  POSTHOG_PERSONAL_API_KEY
  POSTHOG_PROJECT_ID
)

# Secrets that differ per env. With shared PostHog + shared Stripe price IDs,
# only the Stripe API keys/webhook secrets remain env-divergent (test mode in
# dev, live mode in prod once you cut over).
# Each name listed here will be created with both _DEV and _PROD suffixes.
ENV_SCOPED_SECRETS=(
  STRIPE_SECRET_KEY
  STRIPE_WEBHOOK_SECRET
)

create_and_grant() {
  local name="$1"
  if gcloud secrets describe "$name" --project="$PROJECT_ID" >/dev/null 2>&1; then
    echo "  [exists] $name"
  else
    gcloud secrets create "$name" --replication-policy=automatic --project="$PROJECT_ID" >/dev/null
    echo "  [created] $name"
  fi
  gcloud secrets add-iam-policy-binding "$name" \
    --member="serviceAccount:$RUNTIME_SA" \
    --role="roles/secretmanager.secretAccessor" \
    --project="$PROJECT_ID" >/dev/null 2>&1 || true
}

echo "Shared secrets:"
for s in "${SHARED_SECRETS[@]}"; do
  create_and_grant "$s"
done

echo
echo "Env-scoped secrets (creating both _DEV and _PROD variants):"
for s in "${ENV_SCOPED_SECRETS[@]}"; do
  create_and_grant "${s}_DEV"
  create_and_grant "${s}_PROD"
done

echo
echo "Done. Populate secret values via:"
echo "  echo -n 'VALUE' | gcloud secrets versions add SECRET_NAME --data-file=- --project=$PROJECT_ID"
echo
echo "List secrets with no values yet (run once you've populated some):"
echo "  for s in \$(gcloud secrets list --project=$PROJECT_ID --format='value(name)'); do"
echo "    versions=\$(gcloud secrets versions list \"\$s\" --project=$PROJECT_ID --format='value(name)' | wc -l)"
echo "    [[ \$versions -eq 0 ]] && echo \"  empty: \$s\""
echo "  done"
