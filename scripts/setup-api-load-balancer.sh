#!/usr/bin/env bash
# Put a global external Application Load Balancer in front of a partner API
# Cloud Run service, so it can serve a custom domain and be locked down to
# LB-only ingress.
#
# Usage:
#   GCP_PROJECT_ID=msanii-484501 \
#   SERVICE=msanii-api-dev \
#   DOMAIN=api-dev.msanii-beta.com \
#     ./scripts/setup-api-load-balancer.sh
#
# DOMAIN takes a comma-separated LIST, and re-running with an extra name adds
# it without disturbing the ones already served:
#
#   DOMAIN=api-dev.msanii-beta.com,api-dev.remediio.com.ai ...
#
# That is the rebrand path. One LB, one IP, one backend; each hostname gets its
# own DNS authorization, certificate and certificate-map entry, so both answer
# at once and partners migrate on their own schedule instead of on a flag day.
# Retire the old name later by dropping it from DOMAIN and deleting its map
# entry — the cert map is the only thing that decides which hosts are served.
#
# Optional:
#   REGION=northamerica-northeast2   # must match the Cloud Run service
#   PREFIX=<name prefix>             # defaults to $SERVICE
#   CLOUD_ARMOR=true                 # also create + attach a rate-limit policy
#
# Idempotent: every create is guarded, so re-running fills in whatever is
# missing. Run it once per environment (dev, then prod).
#
# WHY AN LB AND NOT A DOMAIN MAPPING: Cloud Run domain mappings aren't offered
# in every region (northamerica-northeast2 among them), Google doesn't
# recommend them for production, and they give you nothing but a hostname. The
# LB works with any region and is what makes Cloud Armor and LB-only ingress
# possible.
#
# WHY CERTIFICATE MANAGER AND NOT A CLASSIC MANAGED CERT: a classic managed
# cert validates by resolving the domain to the LB's IP, so it cannot be issued
# — or renewed — while something else (Cloudflare's proxy, say) holds the A
# record. Certificate Manager validates through a CNAME instead, which works
# either way and keeps renewals from silently failing months later.
#
# PREREQUISITE — the service must already allow allUsers on run.invoker. A
# serverless NEG does not authenticate as a principal, so without that binding
# the load balancer gets the same 403 a browser does.
#
# Under domain restricted sharing (constraints/iam.allowedPolicyMemberDomains)
# that binding is refused. The constraint is a LEGACY MANAGED one that accepts
# Cloud Identity customer IDs only — there is no "public" value to add to it,
# and `principalSet://goog/public:all` is rejected outright. The supported
# escape is a TAG-CONDITIONAL policy, which is also the narrowest: the
# exception attaches to individually tagged resources, never to a whole
# project. Done once per project, then once per service:
#
#   gcloud resource-manager tags keys create allUsersIngress \
#     --parent=projects/$PROJECT_ID
#   gcloud resource-manager tags values create True \
#     --parent=$PROJECT_ID/allUsersIngress
#   # project-level policy, inheritFromParent: true, one conditional rule:
#   #   - allowAll: true
#   #     condition:
#   #       expression: resource.matchTag("$PROJECT_ID/allUsersIngress", "True")
#   gcloud org-policies set-policy drs-tagged.yaml --project=$PROJECT_ID
#
#   # per service:
#   gcloud resource-manager tags bindings create \
#     --tag-value=$PROJECT_ID/allUsersIngress/True \
#     --parent=//run.googleapis.com/projects/$PROJECT_ID/locations/$REGION/services/$SERVICE \
#     --location=$REGION
#   gcloud run services add-iam-policy-binding $SERVICE --region=$REGION \
#     --member=allUsers --role=roles/run.invoker
#
# Applied for msanii-484501 + msanii-api-dev on 2026-09-08; msanii-api (prod)
# needs the two per-service commands once its first tagged deploy exists.
#
# ORDER MATTERS. Do not switch the service to LB-only ingress until this script
# has finished AND the LB actually serves /health. Step 9 prints that check.

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID=your-gcp-project}"
SERVICE="${SERVICE:?Set SERVICE=msanii-api-dev (the Cloud Run service)}"
DOMAIN="${DOMAIN:?Set DOMAIN=api-dev.example.com (comma-separated for more than one)}"
IFS=',' read -r -a DOMAINS <<<"${DOMAIN// /}"
REGION="${REGION:-northamerica-northeast2}"
PREFIX="${PREFIX:-$SERVICE}"
CLOUD_ARMOR="${CLOUD_ARMOR:-false}"

NEG="${PREFIX}-neg"
BACKEND="${PREFIX}-backend"
URLMAP="${PREFIX}-urlmap"
HTTPS_PROXY="${PREFIX}-https-proxy"
HTTP_PROXY="${PREFIX}-http-proxy"
REDIRECT_MAP="${PREFIX}-redirect"
IP_NAME="${PREFIX}-ip"
CERT_MAP="${PREFIX}-certmap"
ARMOR="${PREFIX}-armor"

gcloud config set project "$PROJECT_ID" >/dev/null

echo "Project:  $PROJECT_ID"
echo "Service:  $SERVICE ($REGION)"
echo "Domains:  ${DOMAINS[*]}"
echo

# `gcloud ... describe` is the existence test throughout; a miss is expected,
# so its stderr is dropped rather than treated as a failure.
have() { "$@" >/dev/null 2>&1; }

echo "==> 1/9  Enabling APIs"
gcloud services enable compute.googleapis.com certificatemanager.googleapis.com

echo "==> 2/9  Reserving a global static IP ($IP_NAME)"
have gcloud compute addresses describe "$IP_NAME" --global ||
  gcloud compute addresses create "$IP_NAME" --global --ip-version IPV4
LB_IP="$(gcloud compute addresses describe "$IP_NAME" --global --format='value(address)')"

echo "==> 3/9  Serverless NEG for the Cloud Run service ($NEG)"
have gcloud compute network-endpoint-groups describe "$NEG" --region "$REGION" ||
  gcloud compute network-endpoint-groups create "$NEG" \
    --region "$REGION" \
    --network-endpoint-type serverless \
    --cloud-run-service "$SERVICE"

echo "==> 4/9  Backend service ($BACKEND)"
# NO --protocol. It looks harmless, but the API derives portName from it, and
# a serverless NEG refuses to attach to a backend service that has one at all:
#   Invalid value for field 'resource.portName': 'https'. Port name is not
#   supported for a backend service with Serverless network endpoint groups.
# The default (HTTP/portName http) is accepted; HTTPS is not, and the field
# cannot be cleared afterwards by update or import — it is re-derived every
# time — so getting this wrong means deleting and recreating the resource.
# --global-health-checks is absent for a different reason: serverless NEGs are
# not health checked, Cloud Run reports its own readiness.
have gcloud compute backend-services describe "$BACKEND" --global ||
  gcloud compute backend-services create "$BACKEND" \
    --global \
    --load-balancing-scheme EXTERNAL_MANAGED
# stderr is NOT dropped here. An earlier version sent it to /dev/null and
# treated any failure as "already attached", which turned the portName error
# above into a silent no-op: a complete-looking LB with an empty backend.
if attach="$(gcloud compute backend-services add-backend "$BACKEND" \
      --global \
      --network-endpoint-group "$NEG" \
      --network-endpoint-group-region "$REGION" 2>&1)"; then
  echo "    NEG attached"
elif grep -qiE "already|duplicate" <<<"$attach"; then
  echo "    NEG already attached"
else
  echo "$attach" >&2
  exit 1
fi

echo "==> 5/9  Certificates via Certificate Manager (DNS authorization)"
# One cert per hostname rather than one SAN cert covering all of them: a SAN
# cert fails to issue or renew as a WHOLE if any single domain's authorization
# is not satisfied, so adding a not-yet-delegated rebrand domain would take the
# live one down with it. Separate certs fail independently.
have gcloud certificate-manager maps describe "$CERT_MAP" ||
  gcloud certificate-manager maps create "$CERT_MAP"
for d in "${DOMAINS[@]}"; do
  # Resource names cannot contain dots.
  slug="${d//./-}"
  echo "    $d"
  have gcloud certificate-manager dns-authorizations describe "${PREFIX}-dnsauth-${slug}" ||
    gcloud certificate-manager dns-authorizations create "${PREFIX}-dnsauth-${slug}" --domain="$d"
  have gcloud certificate-manager certificates describe "${PREFIX}-cert-${slug}" ||
    gcloud certificate-manager certificates create "${PREFIX}-cert-${slug}" \
      --domains="$d" \
      --dns-authorizations="${PREFIX}-dnsauth-${slug}"
  have gcloud certificate-manager maps entries describe "${PREFIX}-entry-${slug}" --map="$CERT_MAP" ||
    gcloud certificate-manager maps entries create "${PREFIX}-entry-${slug}" \
      --map="$CERT_MAP" \
      --certificates="${PREFIX}-cert-${slug}" \
      --hostname="$d"
done

echo "==> 6/9  URL map + HTTPS proxy"
# Retried: a backend service reports "not ready" for a few seconds after a NEG
# is attached, and a single attempt here fails the whole run.
if ! have gcloud compute url-maps describe "$URLMAP" --global; then
  for attempt in 1 2 3 4 5 6; do
    gcloud compute url-maps create "$URLMAP" --default-service "$BACKEND" --global && break
    echo "    attempt $attempt: backend not ready, waiting 15s"
    sleep 15
  done
fi
have gcloud compute target-https-proxies describe "$HTTPS_PROXY" --global ||
  gcloud compute target-https-proxies create "$HTTPS_PROXY" \
    --url-map "$URLMAP" \
    --certificate-map "$CERT_MAP" \
    --global

echo "==> 7/9  Forwarding rule on :443"
have gcloud compute forwarding-rules describe "${PREFIX}-https" --global ||
  gcloud compute forwarding-rules create "${PREFIX}-https" \
    --global \
    --load-balancing-scheme EXTERNAL_MANAGED \
    --address "$IP_NAME" \
    --target-https-proxy "$HTTPS_PROXY" \
    --ports 443

echo "==> 8/9  Plain HTTP redirected to HTTPS on :80"
# A partner that types http:// gets a 301, never a silently unencrypted key.
if ! have gcloud compute url-maps describe "$REDIRECT_MAP" --global; then
  TMP="$(mktemp -t redirect.XXXXXX.yaml)"
  # No `kind:` — the import schema sets additionalProperties: false and
  # rejects it, with a 60-line schema dump instead of a readable message.
  cat >"$TMP" <<YAML
name: ${REDIRECT_MAP}
defaultUrlRedirect:
  redirectResponseCode: MOVED_PERMANENTLY_DEFAULT
  httpsRedirect: true
YAML
  gcloud compute url-maps import "$REDIRECT_MAP" --source "$TMP" --global --quiet
  rm -f "$TMP"
fi
have gcloud compute target-http-proxies describe "$HTTP_PROXY" --global ||
  gcloud compute target-http-proxies create "$HTTP_PROXY" --url-map "$REDIRECT_MAP" --global
have gcloud compute forwarding-rules describe "${PREFIX}-http" --global ||
  gcloud compute forwarding-rules create "${PREFIX}-http" \
    --global \
    --load-balancing-scheme EXTERNAL_MANAGED \
    --address "$IP_NAME" \
    --target-http-proxy "$HTTP_PROXY" \
    --ports 80

if [ "$CLOUD_ARMOR" = "true" ]; then
  echo "==> 8b     Cloud Armor rate limit"
  # Per-IP, at the edge, BEFORE a request costs a Cloud Run instance. It does
  # not replace the per-key limiter in partner_api (that one knows about keys
  # and credits and is the one that bounds a leaked key's spend) — this bounds
  # volumetric abuse from a single source. Billed per policy + per rule.
  have gcloud compute security-policies describe "$ARMOR" ||
    gcloud compute security-policies create "$ARMOR" \
      --description "Edge rate limit for the partner API"
  gcloud compute security-policies rules create 1000 \
    --security-policy "$ARMOR" \
    --expression "true" \
    --action rate-based-ban \
    --rate-limit-threshold-count 600 \
    --rate-limit-threshold-interval-sec 60 \
    --ban-duration-sec 300 \
    --conform-action allow \
    --exceed-action deny-429 \
    --enforce-on-key IP 2>/dev/null || echo "    (rule 1000 already exists)"
  gcloud compute backend-services update "$BACKEND" --global --security-policy "$ARMOR"
fi

echo
echo "==> 9/9  What you still have to do by hand"
PRIMARY="${DOMAINS[0]}"
cat <<EOF

  1. DNS — for EACH hostname, two records:
EOF
for d in "${DOMAINS[@]}"; do
  slug="${d//./-}"
  rec="$(gcloud certificate-manager dns-authorizations describe "${PREFIX}-dnsauth-${slug}" \
    --format='value(dnsResourceRecord.name,dnsResourceRecord.data)')"
  cat <<EOF

     $d
       challenge  CNAME  $rec
       hostname   A      $d -> $LB_IP
EOF
done
cat <<EOF

     Leave the CNAMEs in place forever — renewals re-validate against them.
     Behind Cloudflare the proxy (orange cloud) may be ON from the start: the
     cert validates through the CNAME, not the A record. SSL mode "Full
     (strict)".

  2. Wait for each certificate. Minutes to ~an hour:
       gcloud certificate-manager certificates list --format='table(name, managed.state)'
     Proceed at ACTIVE. Certs are per hostname and fail independently, so a
     domain you have not delegated yet cannot hold up one that is live.

  3. Prove the LB actually serves the app:
       curl -fsS https://${PRIMARY}/health

  4. ONLY THEN lock the service down. Set the GitHub secret so deploys keep it
     that way (DEV_API_HOSTNAME / PROD_API_HOSTNAME = $PRIMARY), and flip the
     live service now rather than waiting for the next deploy:
       gcloud run services update $SERVICE --region $REGION \\
         --ingress internal-and-cloud-load-balancing

     After this the .run.app URL stops answering and only the LB can reach the
     service. --allow-unauthenticated stays REQUIRED: ingress is the network
     gate, IAM is a separate one, and the LB does not authenticate as a
     principal. That is also why the tag-conditional DRS exception at the top
     of this script is needed either way.

  5. AFTER DNS is live, verify what the request log records as the client IP.
     partner_api.service.client_ip takes the LAST X-Forwarded-For hop; an ALB
     adds one, so the real client may now be second-to-last. Curl from a known
     address and read partner_api_requests.client_ip. If it shows the load
     balancer, that is a one-line fix — and until it is right, the leaked-key
     forensics record the LB for every request.

EOF
