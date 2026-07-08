#!/usr/bin/env bash
# Populate GSM secret values by reading from a local .env file.
#
# Usage:
#   GCP_PROJECT_ID=msanii-prod ./scripts/populate-gsm-from-env.sh dev .env
#   GCP_PROJECT_ID=msanii-prod ./scripts/populate-gsm-from-env.sh prod .env.prod
#
# Args:
#   $1  env name (dev|prod) — controls which env-suffixed secrets get the values
#   $2  path to .env file
#
# Behavior:
#   - For SHARED secrets, pushes the value to the unsuffixed GSM name.
#   - For ENV-SCOPED secrets (Stripe, PostHog), pushes to NAME_DEV or NAME_PROD
#     depending on the env arg.
#   - Skips empty values in the .env file.
#   - Prints a diff-style report at the end (added / skipped / unchanged).
#
# Run scripts/seed-gsm-secrets.sh first to create the secret containers.

set -euo pipefail

ENV_NAME="${1:-}"
ENV_FILE="${2:-}"
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID=your-gcp-project}"

if [[ -z "$ENV_NAME" || -z "$ENV_FILE" ]]; then
  echo "Usage: GCP_PROJECT_ID=<project> $0 <dev|prod> <path-to-env-file>"
  exit 1
fi
if [[ "$ENV_NAME" != "dev" && "$ENV_NAME" != "prod" ]]; then
  echo "ERROR: env must be 'dev' or 'prod'"
  exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE not found"
  exit 1
fi

ENV_SUFFIX="$(echo "$ENV_NAME" | tr '[:lower:]' '[:upper:]')"  # DEV or PROD

# Secrets shared across dev + prod (unsuffixed GSM name = env-var name).
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
  STRIPE_PRICE_MONTHLY
  STRIPE_PRICE_ANNUAL
  POSTHOG_PROJECT_TOKEN
  POSTHOG_PERSONAL_API_KEY
  POSTHOG_PROJECT_ID
)

# Env-scoped secrets: env-var name in code is unsuffixed, GSM name is suffixed.
ENV_SCOPED_SECRETS=(
  STRIPE_SECRET_KEY
  STRIPE_WEBHOOK_SECRET
)

# Parse .env. Strips quotes, ignores comments and blank lines. Returns "" if not found.
get_env_value() {
  local var="$1"
  # shellcheck disable=SC2002
  cat "$ENV_FILE" | awk -F= -v key="$var" '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    {
      sub(/^[[:space:]]+/, "", $1)
      if ($1 == key) {
        $1=""
        sub(/^=/, "")
        # strip surrounding quotes
        gsub(/^["'"'"']|["'"'"']$/, "")
        print
        exit
      }
    }
  '
}

added=()
skipped_empty=()
unchanged=()

push() {
  local env_var="$1"
  local gsm_name="$2"
  local value
  value="$(get_env_value "$env_var")"

  if [[ -z "$value" ]]; then
    skipped_empty+=("$env_var -> $gsm_name (empty in $ENV_FILE)")
    return
  fi

  # Check current latest value; skip if identical.
  local current
  current="$(gcloud secrets versions access latest --secret="$gsm_name" --project="$PROJECT_ID" 2>/dev/null || true)"
  if [[ "$current" == "$value" ]]; then
    unchanged+=("$gsm_name (already latest)")
    return
  fi

  printf '%s' "$value" | gcloud secrets versions add "$gsm_name" --data-file=- --project="$PROJECT_ID" >/dev/null
  added+=("$env_var -> $gsm_name")
}

echo "Reading from: $ENV_FILE"
echo "Project:      $PROJECT_ID"
echo "Env:          $ENV_NAME (suffix: _$ENV_SUFFIX for Stripe/PostHog)"
echo

for s in "${SHARED_SECRETS[@]}"; do
  push "$s" "$s"
done

for s in "${ENV_SCOPED_SECRETS[@]}"; do
  push "$s" "${s}_${ENV_SUFFIX}"
done

echo "Added ${#added[@]} new secret versions:"
printf '  %s\n' "${added[@]}"
echo
echo "Skipped (empty in $ENV_FILE):"
printf '  %s\n' "${skipped_empty[@]}"
echo
echo "Unchanged (latest already matches):"
printf '  %s\n' "${unchanged[@]}"
