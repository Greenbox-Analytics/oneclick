#!/usr/bin/env bash
# Make one Cloud Run service publicly invokable under domain restricted sharing.
#
# Usage:
#   GCP_PROJECT_ID=msanii-484501 SERVICE=msanii-api-dev ./scripts/allow-public-invoker.sh
#
# Optional:
#   REGION=northamerica-northeast2
#
# Idempotent — safe to re-run; every step tolerates "already exists".
#
# WHY THIS EXISTS: constraints/iam.allowedPolicyMemberDomains is a LEGACY
# MANAGED constraint that accepts Cloud Identity customer IDs only. There is no
# "public" value to add to it — `principalSet://goog/public:all` is rejected by
# the API — so `--allow-unauthenticated` fails with FAILED_PRECONDITION no
# matter who runs it. The supported escape is a TAG-CONDITIONAL org policy, and
# it is the narrowest one available: the exception applies to individually
# tagged resources, never to a whole project.
#
# The project policy is a one-time setup and is ALREADY APPLIED for
# msanii-484501 (2026-09-08). To repeat it elsewhere, set a project-level
# policy with inheritFromParent: true and one rule:
#
#   - allowAll: true
#     condition:
#       expression: resource.matchTag("PROJECT_ID/allUsersIngress", "True")
#
# This script does the per-service half: tag the service, then grant allUsers.
#
# NOTE: this is the network-facing grant. The app is not unprotected by it —
# every partner route authenticates a mk_live_ bearer key and rate limits per
# key. Cloud Run has to accept anonymous callers because partners authenticate
# to US, not to Google.

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID=your-gcp-project}"
SERVICE="${SERVICE:?Set SERVICE=msanii-api-dev}"
REGION="${REGION:-northamerica-northeast2}"

TAG_KEY="${PROJECT_ID}/allUsersIngress"
TAG_VALUE="${TAG_KEY}/True"
PARENT="//run.googleapis.com/projects/${PROJECT_ID}/locations/${REGION}/services/${SERVICE}"

gcloud config set project "$PROJECT_ID" >/dev/null
echo "Project: $PROJECT_ID"
echo "Service: $SERVICE ($REGION)"
echo

# ALREADY_EXISTS is the expected outcome on a re-run, so it is not a failure.
# Any other error still surfaces in the output.
run_ok_if_exists() {
  local out
  if out="$("$@" 2>&1)"; then
    echo "    done"
  elif grep -qiE "already exists|ALREADY_EXISTS" <<<"$out"; then
    echo "    already present"
  else
    echo "$out" >&2
    return 1
  fi
}

echo "==> 1/4  Tag key ($TAG_KEY)"
run_ok_if_exists gcloud resource-manager tags keys create allUsersIngress \
  --parent="projects/${PROJECT_ID}" \
  --description="Marks a resource that may be granted to allUsers despite domain restricted sharing"

echo "==> 2/4  Tag value ($TAG_VALUE)"
run_ok_if_exists gcloud resource-manager tags values create True \
  --parent="$TAG_KEY" \
  --description="Public invoker binding permitted on this resource"

echo "==> 3/4  Binding the tag to the service"
run_ok_if_exists gcloud resource-manager tags bindings create \
  --tag-value="$TAG_VALUE" \
  --parent="$PARENT" \
  --location="$REGION"

echo "==> 4/4  Granting allUsers roles/run.invoker"
gcloud run services add-iam-policy-binding "$SERVICE" \
  --region="$REGION" \
  --member=allUsers \
  --role=roles/run.invoker >/dev/null
echo "    done"

echo
URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo "Verifying ${URL}/health"
# The IAM change can take a few seconds to propagate before it stops 403ing.
for attempt in 1 2 3 4 5; do
  CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "${URL}/health" || echo 000)"
  if [ "$CODE" = "200" ]; then
    echo "  200 — $SERVICE is reachable. Re-run the deploy workflow; it will go green."
    exit 0
  fi
  echo "  attempt $attempt: HTTP $CODE, retrying in 5s"
  sleep 5
done

echo "  Still HTTP $CODE. 403 = the IAM binding has not taken effect (check the tag" >&2
echo "  binding landed); 404 = wrong hostname, or the partner host lockdown, which" >&2
echo "  answers only /health and the four tool routers' own routes." >&2
exit 1
