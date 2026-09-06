# Partner API

Exposes OneClick royalty calculation to an external partner as a machine API. A partner's backend authenticates with an opaque bearer key, uploads a royalty statement plus contracts (or structured terms), and streams back the calculated payments. Work is billed to the partner organization's shared credit pool.

**Partner-facing reference:** [`partner-api-reference.md`](partner-api-reference.md) is the document to hand a partner — endpoints, field-by-field request and response shapes, error codes, billing rules. The drop-in Python client + smoke test is `examples/partner-api/msanii_partner.py` (its wire format is pinned by `tests/test_partner_example_client.py`). This page is the internal one: how it is built and why.

**Specs:** `docs/superpowers/specs/2026-08-10-oneclick-partner-api-design.md` (v1) and `2026-09-04-partner-portal-api-keys-design.md` (the org-admin key console). Org-pool overdraft is a separate, still-provisional spec (`2026-09-03-partner-org-overdraft-design.md`) and nothing here depends on it.

## The shape of it

Three surfaces, deliberately on different hosts:

| Surface | Mounted at | Auth | Who |
|---|---|---|---|
| Machine API | `/oneclick/v1/*`, `/registry/v1/*`, `/splitsheet/v1/*`, `/zoe/v1/*` | `Authorization: Bearer mk_live_…` | The partner's servers |
| Org key console | `/orgs/{org_id}/partner-keys` | User JWT, org admin | The partner's own admin, in `/teams` |
| Provisioning | `/admin/orgs/{org_id}/…` | User JWT, Msanii admin | Msanii staff |

The machine API runs as its **own Cloud Run services** — `msanii-api` (prod, `deploy-api.yml`, on `v*` tags) and `msanii-api-dev` (`deploy-api-dev.yml`, on push to `main`) — built from the same image as the product backend, with `PARTNER_API_ENABLED=true`. That flag also arms `_partner_host_lockdown` in `main.py`, which 404s every path except `/health` and the partner routers' OWN routes (`_PARTNER_API_PATHS`, matched by route regex). The allowlist is by route, not prefix, on purpose: the API shares the `/oneclick` and `/zoe` prefixes with product routes (`/oneclick/calculate-royalties`, `/zoe/ask-stream`), so a prefix allowlist would have let the product surface through on the API host. Without the lockdown that host would answer for the admin console, Stripe webhooks and `/docs` — including on the raw `*.run.app` URL that bypasses the edge. It is a property of the build, not of a WAF rule someone has to remember. Product services never set the flag, so it is a no-op there.

**Paths are tool-first** (owner decision 2026-09-05, replacing `/partner/v1`): each tool's routes live under its own prefix and version — `oneclick_router = APIRouter(prefix="/oneclick/v1")` in `partner_api/router.py`, `registry_router` (`/registry/v1`) in `partner_api/registry.py`, `splitsheet_router` (`/splitsheet/v1`) in `partner_api/splitsheet.py`, `zoe_router = APIRouter(prefix="/zoe/v1")` in `partner_api/zoe.py` — and there are no account-level routes (the old `/me` and `/test` were removed 2026-09-06). `main.py` mounts them bare and the lockdown reads its allowlist off the same tuple, so adding a tool is one router plus one entry in `PARTNER_API_ROUTERS`. Zoe's prefix ends in `/v1` so that OpenAI's own `/chat/completions` and `/models` paths land under it and `base_url=<host>/zoe/v1` works with a stock SDK.

## Keys

**One key type** (owner decision 2026-09-04). A key is the org's credential: it resolves to its org, spends that org's pool, and can do everything the API offers — calculate, read the pool balance, run the free test. There is no hierarchy, no per-customer key, no scoped variant. The original spec's backend/license split solved a B2B2C problem (a partner's server minting keys for *its* customers) that nobody in the console has, and every extra concept on a credential is one more thing to leak. `20260904000002` drops `parent_key_id` and `user_ref`; per-key attribution stays on `credit_ledger.metadata.partner_key_id`.

Keys are minted by **humans only** — the org's own admins in the console, or a Msanii admin. Nothing on the bearer-key surface mints, lists or revokes.

Only the SHA-256 hash is stored. The plaintext leaves the system exactly once, in the mint response. The table is **deny-all RLS** — RLS enabled with zero policies, so only the service-role client touches it and an end-user JWT can never read a key hash. Treat a key like a password: it belongs on the partner's servers, never in anything they ship.

## Endpoints

### Machine (bearer key)

| Method | Path | Notes |
|---|---|---|
| POST | `/oneclick/v1/royalties` | Multipart in, SSE out. See "Calculating" |
| POST | `/registry/v1/splits` | Multipart in, SSE out: `contract_terms` (the royalties input shape) + the Registry `splits` pivot. See "Splits" |
| POST | `/splitsheet/v1/documents` | JSON in, the PDF/DOCX out. No LLM, always the base. See "Split sheets" |
| POST | `/zoe/v1/chat/completions` | OpenAI-compatible, stateless. See "Zoe" |
| GET | `/zoe/v1/models` | `[{"id": "zoe"}]`, so SDKs and gateways that probe it don't 404. Free — and therefore THE key check |

`/me` and `/test` were removed on 2026-09-06 (owner decision): there are no account-level routes, the surface is the four tools. A partner checks a key with `GET /zoe/v1/models`; the balance lives in the team console and in a `402`'s body.

### Org console (`/orgs/{org_id}/partner-keys`, org admin)

`GET` lists every key of the org with a resolved `created_by_label`; `POST` mints (`{label, expires_at?}`); `DELETE` revokes. Gate order is load-bearing and lives in `_gate`: `authz.require_admin` (403) → org exists (404) → `partner_api_enabled` (403 `access_disabled`) → `_require_live_org` on writes only (409). Authz is first so a non-member never learns whether the org exists, is partner-enabled, or is archived. Reads still work on an archived org so an admin can see what existed.

Per-key spend is **not** a fourth endpoint. `GET /orgs/{org_id}/usage` already walks the pool's debit rows for the period, so it returns a `byKey` list alongside `seats`.

The partner-facing docs are the **API** section of the public docs page (`/docs?section=api` — `ApiContent` in `Documentation.tsx`, mirroring `docs/partner-api-reference.md`; keep the two in step), not the org page: the org page keeps the key table plus a link, and a non-enabled team's **API access** card links there too. The section is tabbed (Overview / Authentication / Royalty calculation / Splits / Split sheet / Zoe / Errors / Billing & limits, deep-linkable as `?section=api&tab=…`) and the docs sidebar has two folds, **Platform** and **API** — the API fold is its own reference nav (endpoints, schemas, errors) whose rows open the right tab and scroll to the heading. Beside the article sits the **Trial a request** console (`components/docs/PartnerApiConsole.tsx`; samples and presets in `partnerApiSamples.ts`), one per kind of tab: the free key check (`GET /zoe/v1/models`), a royalties one (real multipart POST built from an editable statement + terms/PDFs, the stream read in the browser), a splits one (PDF picker + `main_artist_name`), a split-sheet one (editable JSON body + format, the file offered as a blob download) and a Zoe one (a real completion). Billed runs are REAL and spend the team's credits; the console never computes a charge — a billed 200 is labelled `billed (base N)` and the true number lives in the team's ledger (the API has no balance read). All consoles stay mounted so a run survives a tab switch. Every call goes straight to the partner host — not through our backend — so what it proves is the exact path the partner's server will take, CORS and all (the partner service already receives `ALLOWED_ORIGINS`). The key lives in page state for the visit and is never persisted. With `VITE_PARTNER_API_URL` unset the console is replaced by a note rather than pointed at a dead host.

### Provisioning (`/admin/orgs/{org_id}/…`, Msanii admin)

`PUT /partner-api` toggles `organizations.partner_api_enabled`. `POST|GET|DELETE /partner-keys` manage keys. Msanii-admin only for the same reason dispersal is: any signed-in user can create an org and is auto-made its admin, so a customer-writable dial would hand the partner surface, and its pool spend path, to anyone.

**Enterprise orgs are born with the bit on** (2026-09-04). An enterprise org only exists because a Msanii admin created it (`POST /admin/orgs`) or promoted it (`PUT /admin/orgs/{id}/kind` → `enterprise`), which is exactly the vetting the bit was guarding — so both writes set `partner_api_enabled = true`, and `20260904000001` backfills the rows that predate that. Demoting to self-serve leaves the bit alone: yanking a partner's live keys is an explicit revoke, never a side effect of a plan change. Self-serve teams stay off unless flipped by hand.

In the UI: the admin console's Organizations drawer (License tab) has the switch, and a **Partner API** tag on the header when it's on. A team admin whose org is *not* enabled sees an **API access** card in `/teams` with "Talk to us about Enterprise" — discoverable, not self-enableable. Turning the bit off 403s every one of the org's keys on the next request (`resolve_key`), so the switch copy says so.

## Calculating

`POST /oneclick/v1/royalties` is multipart: a `statement` file, then **exactly one** of `contracts` (the field repeated once per file — PDF only in v1; `contracts[]` is NOT accepted, FastAPI matches the exact name) or `contract_terms` (structured JSON, no LLM touched). Optional `expenses` JSON and an `Idempotency-Key` header.

Caps are enforced before any work: 10 contract files, 20 MB of contracts, 10 MB statement. The statement is then read *before* any contract is parsed: every statement failure is a `CalculationError` that returns without a debit, so parsing first would hand out unbilled LLM runs. These must stay under Cloud Run's 32 MB request limit or the app-level 413s are unreachable.

Errors split by phase, which matters for a client:

- **Before the stream** they are real HTTP statuses — 401 `invalid_key`, 403 `access_disabled`, 402 `insufficient_credits` (with `price` and `balance`), 413 `file_too_large` / `too_many_contracts`, 422 for a bad request shape.
- **After the stream opens** HTTP is already 200, so a failure arrives as a terminal SSE event `{"type": "error", code, message, suggestion, details}` — `details` is the `CalculationError.details` dict (`available_columns`, `statement_songs`…), the structured context a partner needs to fix the input without a person in the loop. An unexpected exception becomes `internal_error` with a request id and no stack trace.

`: ping` heartbeat comments keep the connection alive during the LLM parse, which routinely outlives a proxy's ~100s idle timeout.

`run_partner_calc` owns its temp directory and deletes it in its own `finally`, and the worker task is started **before** the `StreamingResponse` is returned — started inside the generator it would never run for a client that disconnects first, orphaning the directory on a tmpfs `/tmp`. `asyncio.to_thread` cannot be cancelled, so cleanup must live where the files are read — a disconnecting client must not delete files out from under a running thread.

## Zoe (`partner_api/zoe.py`)

`POST /zoe/v1/chat/completions` speaks OpenAI's chat-completions protocol — the OpenAI SDK with `base_url=<host>/zoe/v1`, `api_key=mk_live_…`, `model="zoe"` works unchanged, streaming included (`chat.completion.chunk` frames, then `data: [DONE]`). The DTO honours `messages` (roles system/user/assistant, string or text-part content), `stream`, `temperature`, `max_tokens`; every other OpenAI field is **ignored, not rejected** (`extra="ignore"`) so a stock SDK call never 422s on `n` or `user`. `model` must be `zoe` (404 `model_not_found` otherwise — a caller asking for `gpt-4o` must not silently get Zoe). Input is bounded because it is a money path with a metered tail: 100 messages, 100k characters total, `max_tokens` ≤ 4,000 (also the default).

**Stateless by design.** No stored contracts, no conversation memory, no RAG: the partner supplies the whole context in `messages`. A key spends credits; it must never widen into an org's stored documents, which belong to members and their artists (`test_prompt_is_stateless_no_supabase_reads` pins that the only Supabase traffic is billing). The persona is `ZOE_SYSTEM_PROMPT`, the in-app general-knowledge prompt with its prompt-only topical guard; a partner's own `system` message follows it and may add context but cannot precede it.

**Billing** is the calculation's pattern on a second price row — `credit_prices.partner_zoe_message` (`20260905000001`, seeded 5, dialled independently of the product's `zoe_message`) via `get_price(sb, ZOE_ACTION)` → `check_pool` (402 before any model call) → `compute_charge` → `debit_run(..., action=ZOE_ACTION)`. Charge-on-delivery in both modes: the non-streaming body is yielded from a `StreamingResponse` generator and the debit sits after that yield; the stream's debit sits after the last content chunk and before the terminal `stop` + `[DONE]` frames, so a client that drops mid-answer closes the generator at a yield and is never charged. The stream is a SYNC generator wrapped in `iter_with_llm_context` (Starlette steps it in the threadpool; the tracked client snapshots the accumulator at `create()`), and the two pricing inputs are read inside the last step, after the tracked stream has recorded its usage. No idempotency: a chat completion has no natural request identity (OpenAI has none either), so every delivered answer pays. **Every completion pays at least the base** — the product's free "conversational" fast path is a UI nicety the API does not have.

## Splits (`partner_api/registry.py`)

`POST /registry/v1/splits` is the Registry's contract parse for partners: the same pipeline as file-mode calculating (`pdf_to_markdown` → `get_or_parse` on the shared, cross-tenant, text-keyed parse cache → `merge_contracts`), minus the statement, run by `service.run_partner_parse` in a worker thread that owns tmpdir cleanup. The result event carries TWO views of the merged contract: `contract_terms` via `models.from_contract_data` — the exact `PartnerContractTerms` shape the royalties endpoint accepts, so a partner parses once and calculates at the base forever after — and `splits` via `registry.contract_splits.parse_royalty_splits(contract_data=…, main_artist_name=…)`, the pivot the Add Work wizard uses. `main_artist_name` is folded into the idempotency fingerprint (a different artist is a different pivot). Priced by `partner_registry_parse` (seeded 30 by `20260906000001`), `compute_charge`'d and debited after the result frame like the calc; a `ValueError` from the PDF/parse layer becomes an unbilled `CONTRACT_UNREADABLE` error event, anything else `internal_error` + `request_id`.

## Split sheets (`partner_api/splitsheet.py`)

`POST /splitsheet/v1/documents` takes the product's `SplitSheetRequest` minus the save-to-artist fields (`models.PartnerSplitSheetRequest`, frozen: `split_type` and `format` are Literals, shares bounded 0–100, 1–50 contributors) and renders through the SAME `generate_split_sheet_pdf` / `_docx`. No LLM, so `compute_charge(action, base, None, None)` is always the base — and it is charged per DOCUMENT, matching the product (pdf + docx of one sheet = two charges). The file is one frame: `StreamingResponse` yields the bytes, then debits (charge-on-delivery), with `Content-Disposition` + `Content-Length` set. Idempotent under `Idempotency-Key` on a SHA-256 of the validated body (format included). Priced by `partner_split_sheet` (seeded 20). A generator failure is a 500 `internal_error` with `request_id`, unbilled.

## Billing

Charge-on-success against the org's **pool** wallet, never a personal one, and never a member cap — a machine call has no member.

**Success means DELIVERED** (owner decision 2026-09-04). The debit sits *after* the `result` event is yielded, so a client whose connection drops closes the generator at that yield and nothing below it runs: no output, no charge. The LLM spend on an undelivered run is our loss, not the partner's — the onus for a dropped connection is ours. The same reordering applies to the product's OneClick stream, whose deliverable also rides in one frame (`main.py`, both the cache-hit and fresh branches). Zoe is deliberately untouched: its answer streams incrementally, so by the time the terminal `done` event is reached the user already has the text, and a disconnect mid-answer stops the generator before the debit anyway.

The amount comes from `ai_pricing.compute_charge`, the same three-term formula the product uses: `max(base, metered, base + size_tail)`. The base is `credit_prices.partner_oneclick_run` (seeded at 30), and it is a floor. There must never be a second implementation of that formula.

`check_pool` is the only billing authority — the `debit_credits` RPC deliberately tolerates overdraft, so nothing downstream re-decides it. It compares `bundle + reserve >= price`.

Idempotency: `derive_request_id` builds `uuid5(key_id, "{Idempotency-Key}:{payload_sha256}:{period_end}")`. All four terms are load-bearing. The key id stops one partner's header colliding with another's. The payload fingerprint stops one header ridden across different payloads buying free runs. **The period end** stops a pinned header buying a year of runs for one charge — `idx_credit_ledger_request_id` is a global, never-expiring unique index, so without it the first ledger row would keep matching forever. Same key, same deliverable, same period is charged once; a new period pays again. No header means a fresh `uuid4`, so every retry pays.

Ledger metadata carries `source: "partner_api"` and `partner_key_id`, which is what `byKey` groups on. Partner rows carry no `org_member_id`, so they can never land in a member's seat total.

## Offboarding

Keys belong to the **org**, not the person who created them. Removing a member never revokes a key — auto-revoking would take "Production backend" offline because someone left. Instead `notify_admins_of_removed_members_keys` tells every remaining active admin there is something to rotate. It fires only on a real transition to `removed`, never on suspend (reversible) and never on a retried removal. It swallows its own failures, so a notification problem can never fail a removal.

## Environment

| Var | Where | Purpose |
|---|---|---|
| `PARTNER_API_ENABLED` | API services only (`msanii-api`, `msanii-api-dev`) | Arms the API routers **and** the host lockdown. Must never be set on a product service |
| `CREDITS_ENABLED` | Both | Required — keys spend credits |
| `LICENSING_ENABLED` | Both | Required — keys belong to orgs |
| `VITE_PARTNER_API_URL` | Frontend | Base URL of the partner service, shown in the console and used by its Try-it box. Blank hides the live test |

All three must be true for the machine surface to answer; otherwise it 404s, which is a true rollback. The org console requires only the latter two, since it runs on the product host.

## Running it locally

`task dev` starts three servers: the frontend on :8080, the product backend on :8000, and the API on :8001 (`task dev:partner` alone) — `http://localhost:8001/zoe/v1/models`, `/oneclick/v1/royalties`, `/registry/v1/splits`, `/splitsheet/v1/documents`, `/zoe/v1/chat/completions`. The partner one is the same app with `PARTNER_API_ENABLED=true` set on that process — the flag must never go in `.env`, where it would 404 every non-partner route on :8000 too. Set `VITE_PARTNER_API_URL=http://localhost:8001` so the docs page's console points at it, and restart Vite (it reads `.env` only at startup).

The key console at `/teams` runs on the product backend and needs no flag; it appears once the org has the capability bit — automatic for an enterprise org, or flipped for a self-serve team from Admin → Organizations → (org) → License → Partner API. The team's id is shown under its name on that page.

## Testing

```bash
cd src/backend && poetry run pytest tests/test_partner_keys.py tests/test_partner_models.py \
  tests/test_partner_billing.py tests/test_partner_router.py tests/test_partner_calculate.py \
  tests/test_partner_org_router.py tests/test_partner_portal_service.py \
  tests/test_partner_registry.py tests/test_partner_splitsheet.py tests/test_oneclick_billing_delivery.py \
  tests/test_orgs_offboard_partner_keys.py tests/test_compute_charge.py tests/test_fetch_all.py \
  tests/test_partner_example_client.py tests/test_partner_zoe.py -v

npx vitest run src/components/orgs/__tests__/partner-keys-helpers.test.ts \
  src/components/orgs/__tests__/api-keys-panel.test.tsx \
  src/components/docs/__tests__/partner-api-console.test.tsx src/pages/__tests__/documentation-api.test.tsx
```

pytest mocks Supabase and never reaches Postgres, so `supabase/qa/gates_partner_api_keys.sql` is the only executable coverage of the RLS layer. Run it in the SQL editor after applying the migration. It ends by RAISING `PASS (2/2): …` — that error IS the success signal, thrown to roll the whole gate back; any other message is a real failure.

The two delivery-billing tests cannot use TestClient or httpx's ASGI transport (both drain the response, so neither can drop a connection). `test_partner_calculate.py` drives the generator directly and `aclose()`s it at the result frame; `test_oneclick_billing_delivery.py` wraps `main.StreamingResponse` to cut the stream there. Both are mutation-verified: reorder the debit and they go red.

Tests must **never** set `PARTNER_API_ENABLED` except where they are proving the lockdown, and note that `main.py` runs `load_dotenv()` at import time inside the `client` fixture — a test needing a flag off should `monkeypatch.delenv` it explicitly rather than trusting conftest alone.

## Deploy

The API has its own pair of workflows, identical copies of the product backend's minus the OneClick cache-clear steps: `.github/workflows/deploy-api-dev.yml` deploys `msanii-api-dev` on every push to `main` that touches `src/backend/**`, and `deploy-api.yml` deploys `msanii-api` on a `v*` tag. Each builds its own image (`gcr.io/…/msanii-api[-dev]`) from the same Dockerfile and deploys it with `--timeout 600`, `--max-instances 5`, `--concurrency 10` and `PARTNER_API_ENABLED=true` appended to the env string; every other env var and GSM secret is the product service's, same names. Concurrency is capped because each in-flight calculation holds a worker thread for the whole LLM parse; Cloud Run's default of 80 would queue dozens of runs behind the thread pool, and `--max-instances` alone bounds nothing. 10 × 5 is 50 concurrent runs platform-wide — a calibration dial, not a law. The product workflows no longer deploy any partner service; the earlier `msanii-backend-partner` service is orphaned and can be deleted from Cloud Run.

Both API workflows curl `/health` after deploying. `/health` is the one path the lockdown leaves open, so that doubles as a lockdown smoke test.

Wrapping the service in its own domain is a Cloud Run domain mapping (or a load balancer) on `msanii-api`; nothing in the app assumes a host. Two things must then point at that domain: the frontend's `VITE_PARTNER_API_URL` (per environment — it feeds the docs page's base URL, samples and trial console) and the partners' own configuration. The API service's `ALLOWED_ORIGINS` already covers the app origin, which is what the trial console needs.

## Not built (deliberate)

Per-key rate limiting (on Zoe too, whose base is 5 — a runaway loop is bounded per call by the pool balance and visible in `ai_usage_log` under `zoe_partner`; the only free route, `GET /zoe/v1/models`, touches no model and no pool), key rotation as one endpoint, CSV export, cross-period usage history, partner-facing webhooks, self-serve enablement of `partner_api_enabled`, and a load balancer or API gateway. Cloud Run already spreads load across instances of one service, and a gateway's request meter would be a second number that disagrees with the ledger, since the charge is computed after the work. Revisit rate limiting in-app if abuse appears.
