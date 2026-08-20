# Licensing & Teams

Everything behind `LICENSING_ENABLED`: organizations ("Teams" in the UI), the shared credit pool, per-member caps, and **team-owned artist profiles**.

> **The flag is a true rollback — until real orgs exist.** With `LICENSING_ENABLED` unset, every `/orgs/*` route 404s at the router level, `resolve_billing_org_for_project` short-circuits before any query, and `useActiveTeam()` returns null so no team affordance renders. Nobody can be in a team, so every artist has `team_id IS NULL` and every code path is the one that shipped before licensing existed. **Caveat:** the SQL layer (RLS policies, triggers, `can_access_artist`) is live regardless of the flag. Once orgs and team artists exist, flipping the flag off freezes the feature (no new orgs, no transfers, no org billing) but existing members keep team-artist access via RLS and can still create team artists straight through PostgREST. Rollback-after-adoption is "frozen but live", not a no-op.

---

## The two edges

Licensing has exactly two ownership questions, and each is answered by one column.

| Question | Edge | Where |
|---|---|---|
| Who is in this team? | `org_members.user_id` + `status='active'` | `is_org_member` / `is_org_admin` SQL helpers, mirrored by `orgs/authz.py` |
| Who owns this artist — and therefore its projects, works, files, audio and credentials? | `artists.team_id` | `can_access_artist()` |

`artists.team_id IS NULL` means personal (the behaviour that predates licensing). `NOT NULL` means the org owns it, and everything below the artist follows by foreign key — ten tables cascade off one row, so no per-project or per-file ownership record exists or is needed.

**Historical note:** a per-project `org_project_links` table used to answer the second question one project at a time. It was retired in `20260804000001` — two sources of truth for "who pays" is exactly what the artist edge exists to eliminate. If you find a reference to linking or unlinking a project, it is stale.

---

## `can_access_artist` — the one predicate

```sql
can_access_artist(p_artist_id UUID, p_user_id UUID, p_require_admin BOOLEAN DEFAULT FALSE)
```

`SECURITY DEFINER`, `STABLE`, `search_path` pinned. Defined in `20260803000001_team_owned_artists.sql`.

- **true** for the personal owner of a personal artist (`team_id IS NULL AND user_id = caller`)
- **true** for an ACTIVE member of the owning org, when that org is not archived
- with `p_require_admin => TRUE`, the team branch additionally requires `role = 'admin'`
- **false** otherwise — including a suspended member, a removed member, and an archived org

Org *status* is deliberately not checked, only `archived_at`: a `pending` org (created, not yet paid up) is a team still being set up, and its members should be able to build the roster they are about to pay for. Billing has its own `status='active'` gate in `resolve_billing_org_for_project` — access and payment are different questions.

Every team RLS policy on every artist-scoped table calls this rather than repeating the join, so access cannot drift table to table. **If you add a table that hangs off an artist, add one `FOR ALL` policy that calls this function** — that is the whole integration.

---

## Access control is three layers, not two

| Layer | Grants | Enforced by |
|---|---|---|
| **Artist ownership** | Everything under one artist — projects, works, files, audio, credentials | `can_access_artist` (personal owner, or active org member) |
| **Project members** (`owner`/`admin`/`editor`/`viewer`) | All works in one project | `project_members` policies |
| **Work-only collaborators** | Only the specific work they were invited to | `works_registry` / `work_files` policies |

They are permissive and OR together — a user reaches a row if *any* layer grants it.

### The creator is not the owner

`artists.user_id` keeps holding whoever **created** the artist, even after it is handed to a team. On a team artist it is a creator stamp, nothing more; `team_id` is ownership.

This matters because every policy written before licensing reads `artists.user_id = auth.uid()`. `20260803000002` re-scopes all 21 of them with `AND a.team_id IS NULL` — without that, the creator of a team artist would keep personal-owner rights over the whole subtree **forever, including after being offboarded** (`can_access_artist` would deny; the old policy would grant anyway, and permissive policies OR). `20260805000002` applies the same re-scope to `notes` / `note_folders`, which the first pass missed, and adds their team-layer `can_access_artist` policies.

Practical rules when touching this area:

- Never write a new policy or query as `artists.user_id = auth.uid()` without `AND team_id IS NULL`.
- Never use `artists.user_id` to mean "owner". Use `team_id` when set (`recalc_user_storage`, `_bump_storage`, `list_user_storage_paths` and the `NewArtist` cap count all do).
- `artists.user_id` is `ON DELETE SET NULL` (not CASCADE) so a creator deleting their account cannot take the label's roster with them.

---

## Team-owned artists

### Creating one

Artists are created **client-side, straight against PostgREST** — there is no backend `POST /artists`. Two paths do it:

- `src/pages/NewArtist.tsx`
- `src/components/NewArtistDialog.tsx`

Both default `team_id` to the active billing-context org and show `TeamOwnershipField` with a "Keep this artist private to me" escape hatch. Because there is no backend endpoint, **RLS is the only enforcement**: `artists_insert_team` requires an ACTIVE membership in the target org, a **non-archived** org (`20260805000004` — inserting into an archived org would create an artist nobody can access), *and* pins `user_id` to the caller, so neither the team nor the creator stamp can be forged.

Team artists are excluded from the personal `maxArtists` cap count (`.is("team_id", null)` in `NewArtist.tsx`) — the team pays for them.

### Transferring one

```
POST /orgs/{org_id}/artists/{artist_id}/transfer
```

`src/backend/orgs/artists.py`. One-way in v1: 409 if the artist already belongs to a team. Moving an artist *out* of a team whose credits paid for its files is a support decision with a refund question attached, not a self-serve button.

- 404 if the caller holds no active seat in the destination org (membership is checked first, so probing artist ids leaks nothing)
- 403 if the caller is not the artist's current personal owner
- Stamps `transferred_at` / `transferred_by`, then re-derives both storage totals

Nothing is copied — one column moves the whole subtree.

### `team_id` has exactly one writer

`artists_lock_team_id` (a `BEFORE UPDATE OF team_id` trigger) raises `insufficient_privilege` whenever `auth.uid() IS NOT NULL`. The transfer endpoint runs on the service-role client, where `auth.uid()` is NULL and RLS is bypassed — so it is the only path that can change ownership. A policy `WITH CHECK` cannot see `OLD`, so it could not tell "renamed the artist" from "yanked it out of its team"; hence a trigger.

The same reasoning produces `lock_asset_owner_move` on `project_files` / `audio_files`: a member with legitimate read could otherwise walk a file out with `UPDATE project_files SET project_id = <my personal project>`. It compares `team_id`, so same-team moves stay ordinary work.

`lock_parent_owner_move` (`20260805000001`) extends the same rule one level up, where the attack was still open: `projects.artist_id`, `audio_folders.artist_id`, and `works_registry.artist_id` / `project_id` cannot change the resolved owner (team, or personal owner when `team_id IS NULL`) under an end-user JWT. Without it, one `UPDATE projects SET artist_id = <my personal artist>` walked out an entire project — every file rides along untouched because `project_id` never changes, so the file-level trigger never fires.

### What is NOT protected

A member who can read a team artist can retype its name, email and splits into a personal artist. That is not preventable by RLS and nothing in the app pretends otherwise — there is deliberately no "duplicate artist" affordance, but facts leaving in someone's head is a contract problem. What *is* enforced is the assets: file rows cannot cross an ownership boundary, and DSP credentials are admin-only (read included).

---

## Credits: the pool and the caps

> **Superseded (2026-08-15 self-serve teams) — this section describes ENTERPRISE orgs only.** "Any signed-in user can create an org" and the dispersal/activation-floor mechanics below predate self-serve teams, when `POST /orgs` was the only creation path and every org landed as `kind='enterprise'`. That path is now admin-only (`POST /admin/orgs`). A self-serve org (`kind='self_serve'`, created by `POST /orgs`) has no dispersal, no activation floor, and is born `status='active'` — see [Self-serve teams](#self-serve-teams) below for how its pool gets funded instead.

An org negotiates a monthly credit volume, set **only by a Msanii admin** (`PUT /admin/orgs/{id}/dispersal`) — never by the org's own admin. Any signed-in user can create an org and is auto-made its admin, and dispersed credits count toward the activation floor, so a customer-writable dial would mint free credits and self-activate.

- The org holds **one** pool wallet (`credit_wallets`, `owner_type='org'`). The daily sweep grants `organizations.monthly_dispersal_credits` into the **expiring** bundle bucket, so an unspent month can't be banked; purchased packs land in reserve and never expire. Dispersal flows to `pending` orgs too — it counts toward the activation floor and auto-activates the org the moment the floor is met (`orgs.wallets.maybe_activate_org`, shared with the pack-purchase path). On org wallets the dispersal component of `cumulative_paid_in` is the `monthly_grant` ledger kind — the sweep disperses via `rollover_wallet`, and nothing writes the kind `'dispersal'`.
- Members hold **no wallet**. They spend straight from the pool, bounded by `org_members.monthly_cap` → `organizations.default_member_cap` → uncapped.
- The cap counter (`cap_used` / `cap_period_end`) moves **inside `debit_credits`** under the pool lock. A service-side pre-check cannot stop two concurrent actions both slipping under the ceiling.
- An over-cap debit is recorded and flagged (`cap_exceeded`), never rejected — charge-on-success means the work already happened.

Caps are ceilings, not reservations, so they may deliberately sum to more than the dispersal.

**Two walls, different remedies:**

| Wall | What the member sees | Fix |
|---|---|---|
| `capReached` | "ask an admin to raise your cap" + a request CTA | `credit_requests` is a **cap-raise** request; approving writes `monthly_cap` and moves no credits |
| Dry pool | no actionable CTA | only an admin buying credits helps. There is no pay-as-you-go on a pool |

### Who pays for a piece of work

`EntitlementsService.resolve_billing_org_for_project` (`subscriptions/service.py`) is the single place a project resolves to a paying org:

```
project → projects.artist_id → artists.team_id → org
```

It returns a context only when the org is `active`, not archived, **and** the caller holds an active seat. Anything short of that returns `None` and falls through to personal billing — derivation only ever *upgrades*, never restricts, and a team the caller is not in must never become their payer.

A **suspended** org parks the member's stored billing preference instead of clearing it: usage bills personally while suspended, and org billing resumes automatically on reactivation. Only genuinely dead references (org deleted or archived, member removed) lazy-clear. Because the fallback drains a personal wallet, members are told when it happens — suspend/remove/archive each send an email plus an in-app notification, and the frontend shows a one-time "you're now billing to your personal plan" notice when the context flips without the user choosing it.

---

## Storage

Bytes follow artist ownership, and the accounting is maintained by **triggers**, not by the recalc function.

- `trigger_storage_pf_change` / `trigger_storage_af_change` fire on every `project_files` / `audio_files` write and call `_bump_storage(artist_id, delta)`.
- `_bump_storage` is the one place that decides where a delta lands: `organizations.storage_bytes` when the artist is team-owned, else the creator's `usage_counters.total_storage_bytes`.
- `recalc_user_storage(user_id)` and `recalc_team_storage(org_id)` are **repair** paths — full recomputes for support and for the transfer endpoint. Changing only these would leave every real upload charging the wrong side.

A team's cap is `ENTERPRISE_SEAT_STORAGE_BYTES × active seats` (`EntitlementsService._team_storage`). The env value is per person; charging a ten-person team one seat's worth of space would make the feature unusable. The cap moves when the roster does, so an org that loses members can sit over its cap — existing files are never touched, only further uploads block.

Storage is a **hard cap on every tier**, with no pay-per-use.

---

## Endpoints

### Member / admin (`/orgs`, all gated by `require_licensing`)

> **Superseded (2026-08-15 self-serve teams):** `POST /orgs` is now **self-serve-only** and slot-gated (`NoSlotError` → 402) — it no longer produces `kind='enterprise'` rows. Enterprise orgs are created exclusively via the Msanii-admin `POST /admin/orgs`. See [Self-serve teams](#self-serve-teams) for the full self-serve endpoint set (archive/unarchive, dissolve, coverage claim/release, transfer-credits, top-up, ledger).

| Method | Path | Who |
|---|---|---|
| POST / GET | `/orgs` | any signed-in user (creator becomes admin) — self-serve only, see above |
| GET / PUT | `/orgs/{org_id}` | member / admin (`default_member_cap` lives here) |
| POST | `/orgs/{org_id}/archive` | admin |
| GET | `/orgs/{org_id}/usage` | member |
| GET | `/orgs/{org_id}/members` | member — roster for board pickers: `user_id`/`full_name`/`avatar_url`/`role`, **no emails** |
| POST / GET / DELETE | `/orgs/{org_id}/invites`, `/invites/{invite_id}` | admin |
| POST | `/orgs/invites/{token}/accept` · `/decline` | invitee |
| GET | `/orgs/invites/{token}/preview` | anyone holding the token (unauthenticated) — returns only `orgName` / `kind` for the claim page |
| PUT | `/orgs/{org_id}/members/{member_id}/role` · `/cap` | admin |
| POST | `/orgs/{org_id}/members/{member_id}/suspend` · `/reactivate` | admin |
| DELETE | `/orgs/{org_id}/members/{member_id}` | admin |
| POST / GET | `/orgs/{org_id}/credit-requests` | member creates, admin lists |
| POST | `/orgs/{org_id}/credit-requests/{id}/approve` · `/deny` | admin |
| GET | `/orgs/{org_id}/projects` | admin — the projects this org owns, via artist ownership |
| **POST** | **`/orgs/{org_id}/artists/{artist_id}/transfer`** | the artist's personal owner, who must hold a seat |
| PUT / DELETE | `/orgs/{org_id}/projects/{project_id}/members/{member_id}` | admin — grant/adjust/revoke seat access |
| POST | `/orgs/{org_id}/archive` · `/unarchive` | admin — self-serve only; unarchive is slot- and storage-guard-gated |
| GET | `/orgs/{org_id}/dissolve-preview` | admin — self-serve only, read-only |
| POST | `/orgs/{org_id}/dissolve` | admin — self-serve only, typed-name-confirmed, terminal |
| POST | `/orgs/{org_id}/coverage/claim` · `/release` | admin — self-serve only |
| GET | `/orgs/{org_id}/ledger` | admin — pool ledger |
| POST | `/orgs/{org_id}/transfer-credits` | admin — self-serve only; personal reserve → pool |
| POST | `/orgs/{org_id}/cancel-topup` | admin — self-serve only; exempt from `_require_live_org` (the retry path for a cancel that failed mid-archive/dissolve) |
| POST | `/subscriptions/org-topup-checkout` | admin — starts the recurring org top-up subscription |

### Msanii admin (`/admin`)

| Method | Path | Purpose |
|---|---|---|
| PUT | `/admin/orgs/{org_id}/dispersal` | set the monthly credit volume (enterprise) |
| POST | `/admin/orgs/{org_id}/suspend` · `/reactivate` | lifecycle |
| GET | `/admin/orgs/{org_id}/pool` | pool balance |
| POST | `/admin/orgs/{org_id}/pool/clawback` | reserve-only |
| POST | `/admin/orgs` | create an **enterprise** org for an existing customer account — the only producer of `kind='enterprise'` rows besides the kind flip below |
| PUT | `/admin/orgs/{org_id}/kind` | flip an org between `self_serve` and `enterprise`; flipping to `self_serve` requires `covered_by_user_id` (422 without it) and is slot-gated (402 via `NoSlotError`) |

---

## Frontend

| Hook / component | File | Purpose |
|---|---|---|
| `useOrgs` and friends | `src/hooks/useOrgs.ts` | org CRUD, members, invites, credit requests, owned-project list |
| `useActiveTeam()` | `src/hooks/useArtistTeam.ts` | the org whose billing context is active, or null. **Null whenever the flag is off** — every team affordance keys off this |
| `useTransferArtistToTeam()` | `src/hooks/useArtistTeam.ts` | the transfer mutation |
| `TeamOwnershipField` | `src/components/artists/TeamOwnershipField.tsx` | the shared "shared with {team} / keep private" control on both creation paths |
| `useEntitlements().billingContext` | `src/hooks/useEntitlements.ts` | `{type: "personal"} | {type: "org", orgId, orgName, role}` |

---

## Environment variables

| Var | Default | Effect |
|---|---|---|
| `LICENSING_ENABLED` | unset (off) | Master switch. Off = every `/orgs/*` route 404s and no derivation runs |
| `CREDITS_ENABLED` | unset (off) | The credits model. Off = legacy tier gating |
| `ENTERPRISE_SEAT_STORAGE_BYTES` | `500 GiB` | **Per seat.** A team's cap is this × active seats (enterprise only — self-serve uses `tier_entitlements.team_storage_bytes`, see below) |
| `ORG_GRACE_DAYS` | `14` | Self-serve only. Days an uncovered team sits in grace before `evaluate_standing` flips it to `status='lapsed'` |
| `TEAM_STORAGE_OVERAGE_USD_PER_GB` | `0.025` | Self-serve Pro only. PAYG rate for team-storage pool overage, billed monthly as a Stripe InvoiceItem on the covering owner's personal customer |

`conftest.py` clears the first two, so backend tests must set them explicitly with `monkeypatch.setenv` — a developer's `.env` can never leak into a test run.

---

## Testing

pytest mocks the Supabase client and never reaches Postgres, so **the SQL layer's only executable coverage is the gate scripts.** Run them after any migration in this area.

| What | How |
|---|---|
| Backend logic | `cd src/backend && poetry run pytest -q` |
| Artist ownership, RLS, storage triggers | paste `supabase/qa/gates_team_artists.sql` into the Supabase SQL editor |
| The money RPCs | paste `supabase/qa/launch_gates_credit_rpcs.sql` |
| Full HTTP lifecycle | `poetry run python -m scripts.qa_licensing_loop --port 8000` (needs both flags on) |

`gates_team_artists.sql` builds a throwaway org, members, artists, projects and files, asserts, then **RAISEs on purpose** — the error message *is* the report, and the raise is what rolls the test data back. It needs at least 3 rows in `auth.users`. Expected totals: **6** after `20260803000001`, **14** after `…0002`, **16** after `…0003`, **19** after `20260805000001` (parent-pointer move gates), **21** after `…0002` (notes gates).

Every PL/pgSQL variable in those scripts is `v_`-prefixed. PL/pgSQL's `variable_conflict` defaults to `error`, so a variable named `org_id` in `WHERE org_id = org_id` aborts the whole block with "column reference org_id is ambiguous".

---

## Self-serve teams

Spec: `docs/superpowers/specs/2026-08-15-pricing-tiers-teams-design.md`. Same `organizations` table as everything above — `organizations.kind` is `self_serve` | `enterprise` (existing orgs backfilled `enterprise`; zero behavior change for them) — but a different lifecycle, funded and governed differently. Where a term in this chapter conflicts with earlier chapters (dispersal, activation floor, "any signed-in user can create an org"), the earlier text describes enterprise only; see the superseded notes above.

### Flag matrix, including the half-flag case

`orgs.service.create_org` branches on both flags:

| `LICENSING_ENABLED` | `CREDITS_ENABLED` | Result |
|---|---|---|
| on | on | Self-serve: `kind='self_serve'`, slot-gated (`NoSlotError` → 402 at the router), `covered_by`/`covered_at` set to the creator, born `status='active'` — **no activation floor, the slot IS the activation** |
| on | off | **503.** This half-flag window must NOT fall through to the branch below — that would silently mint a permanent `kind='enterprise'` row nobody governs (no dispersal, no activation, no admin visibility) on every hit |
| off | irrelevant | `/orgs/*` 404s at the router before `create_org` is ever called (see the top-of-file rollback note) |

### Tier matrix

| Tier | Monthly grant | Grandfathered grant (until `grandfathered_until`) | Team slots (`max_teams`) | Team size excl. owner (`max_team_members`) | Team storage pool (`team_storage_bytes`) |
|---|---|---|---|---|---|
| Free | 100 | — | 0 | 0 | 0 |
| Basic | 2,000 | 3,000 | 1 | 3 | 100 GiB |
| Pro | 5,000 | 8,000 | 3 | 10¹ | 250 GiB then PAYG |

¹ Pro's team size is not a flat number: seats unlock in blocks of `SEATS_PER_PRO` (5) per Pro member, the covering owner counting as the first Pro for free — `effective_limit = min(max_team_members, SEATS_PER_PRO * (1 + pro_member_count))`. A lone Pro coverer gets 5; one more Pro member joining unlocks the full 10 (`tier_entitlements.pro.max_team_members`, unchanged). See below.

All three team dials live on `tier_entitlements` (migration `20260816000001`), read through `orgs.standing.team_dials_for_user` — a Msanii admin's own dials always resolve as Pro, regardless of their subscription tier. **Joining a team is free on every tier; only owning (covering) one draws on a slot.** A Pro team's effective ceiling is `orgs.service.SEATS_PER_PRO`'s formula (owner decision 2026-08-16, superseding a same-day flat-5-member cap that was never applied): `min(dials.max_team_members, SEATS_PER_PRO * (1 + pro_member_count))`, where `pro_member_count` is the org's other ACTIVE members (excluding the coverer) whose resolved tier is Pro (`orgs.standing.resolve_tier_for_user`, the same admin-implicit-Pro resolution `team_dials_for_user` uses, factored out so the two can't drift). Basic's 3-member cap sits below `SEATS_PER_PRO` so the formula never raises it — a Pro-tier user being a member on a Basic-covered team changes nothing. The gate is ADDING-only, never holding: `_self_serve_seat_room` re-checks on both `invite_member` and `accept_invite`, and a Pro member leaving or downgrading later lowers the ceiling for future invites but does not evict anyone already in. Hitting the ceiling gets no per-org override and no paid seat add-on: `TeamFullError` raises with `next_step="contact"` once at `max_team_members`, and the invite dialog points the admin at Enterprise instead (`team_seat_wall_hit` analytics event fired at the router).

**Grandfathering:** subscribers paid at merge time keep their pre-rescale grant (Basic 3,000 / Pro 8,000) via `subscriptions.grandfathered_monthly_credits`, honored only while `now() < grandfathered_until` — stamped ONCE at the backfill from the row's `current_period_end` (so a monthly subscriber keeps the old grant until this month's renewal, an annual subscriber until their term ends). The stamp is never extended by an interval switch; only a tier change ends it, and ends it EARLY by nulling both columns immediately in the same webhook write. Precedence (explicit admin override > unexpired grandfather > tier value) lives in exactly one helper, `EntitlementsService._resolve_monthly_grant`, called at all five grant sites — a mismatch there is the one way two users could see two different bundles for the same tier.

### Coverage is claimed, never assigned

**One coverer per team, and admin ≠ payer.** A team may have several admins (all Pro, even) — they all manage it, but exactly one of them covers it: that person's plan supplies the slot, their per-owner pool absorbs the team's storage, and their personal Stripe customer is billed for Pro PAYG overage. The others' slots stay free for teams of their own. Pro-ness matters in two independent ways: the *coverer* provides slot + storage; any Pro *member* (admin or not) unlocks the next `SEATS_PER_PRO` block. **Coverage never moves automatically** (owner decision 2026-08-16 — claiming means "put this on my plan and my card", so it must be a deliberate act): the sweep does not hunt for a co-admin with a free slot; it grace-stamps, `_notify_standing` tells every active admin to claim, and past `ORG_GRACE_DAYS` the team lapses if nobody did.

`covered_by` / `covered_at` on `organizations` name who is spending a team slot on this org. Two endpoints move it, both self-serve-only (`_require_self_serve_org`, 409 otherwise) and admin-only:

- `POST /orgs/{id}/coverage/claim` — free-slot-gated (`standing.require_free_slot`, 402 via `NoSlotError`). Idempotent if the caller already actively covers it. If the org is `lapsed`, claiming also requires `storage_guard.reactivation_allowed` (402 otherwise) and flips it back to `active`.
- `POST /orgs/{id}/coverage/release` — **current coverer only**, and the write is conditioned on `WHERE covered_by = user_id`, not just the preceding read: a rival admin's claim landing in the gap between read and write means zero rows match, and the caller gets a 403 telling them to refresh, rather than silently clobbering the rival's fresh `covered_at`. `covered_by` deliberately stays set after a release (it feeds last-coverer storage attribution and the sweep's ranking) — only `covered_at` clears, which is the sweep's signal to start evaluating this org.

### Standing: a daily predicate, never an event

`orgs.standing.evaluate_standing` (called from the daily sweep) is the ONLY writer of grace/lapsed state. Per run:

1. Rank each coverer's orgs by `covered_at DESC`, keep the top `max_teams` — and only if the coverer still holds an ACTIVE ADMIN seat in that org (`_holds_active_admin_seat`; a demoted/offboarded coverer keeps neither the ranking slot nor the seat that earned it).
2. A covered org whose `grace_started_at` was set gets it cleared (and, if it was `lapsed`, flips back to `active` — but only if `storage_guard.reactivation_allowed` passes; a covered-but-over-pool team stays lapsed until space frees up).
3. An uncovered org with no `grace_started_at` gets one stamped `now()`, and every ACTIVE admin is notified (in-app + best-effort email).
4. An uncovered org already in grace, past `ORG_GRACE_DAYS` (env, default 14) days, flips to `status='lapsed'`.

`can_access_artist`'s team branch gained the one new RLS predicate this whole feature adds — `AND o.status <> 'lapsed'` — denying the entire artist subtree to **everyone, admins included**, the same posture as `archived_at`. A bad ranking input for one coverer (corrupt `covered_at`, a dials read blowing up) skips only that coverer's orgs for the run, never treated as "uncovered" (which would wrongly grace-stamp/lapse over a data problem).

### Archive vs. dissolve

Two very different endings, both admin-only, both self-serve-only:

| | Archive | Dissolve |
|---|---|---|
| Reversible? | Yes — `POST /orgs/{id}/unarchive` | No — terminal |
| What happens to artists | Nothing; `team_id` stays attached, subtree goes dormant via `can_access_artist`'s `archived_at` check | Every team artist reverts to a person: the creator if they still hold an active seat, else the dissolving admin (`_dissolve_recipients`) — a service-role `UPDATE artists SET team_id = NULL`, which also **reassigns `user_id`** to the fallback recipient, since `team_id = NULL` makes `user_id` mean "owner" again |
| What happens to the pool | Untouched — whatever it holds survives | Purchased **reserve** forfeited via the existing clawback RPC (fixed `request_id`, so a retry can't claw back twice); the expiring **bundle** bucket is left inert, not reclaimed |
| Balance precondition | None — members hold no credits, nothing to strand | None |
| Org row / wallet / ledger | Retained | Retained (support still needs to read them) |
| Slot | Freed immediately | Freed (was already freed if archived first; `dissolved_at` stamps `archived_at` too if not already set) |

**Unarchive** requires a free slot again (`NoSlotError` → 402, not atomic with the slot check — accepted as a low-frequency admin-only race) and `storage_guard.reactivation_allowed` (402 if the owner is over-pool). Known limitation: org-granted `project_members` rows and members' billing contexts torn down at archive time are **not** restored by unarchiving — a member who lost project access must be re-added by hand.

**Dissolve** is typed-name-confirmed and idempotent (`{"already": true}` on retry, no double-write). Order matters — money first, then the irreversible work, then cleanup: (1) forfeit the pool's purchased reserve, (2) cancel the top-up Stripe subscription (fires `org_topup_canceled` with `trigger="dissolve"`), (3) revert artists to people — **not** best-effort, a crash here aborts the whole call and leaves the org intact and retryable, because a `team_id` left pointing at a dissolved org locks its creator out of their own subtree, (4) re-derive storage totals, (5) soft-remove seats / expire invites / drop org-granted project access (best-effort — the daily sweep reconciles anything missed), (6) stamp `dissolved_at` (+`archived_at` if not already set). `GET /orgs/{id}/dissolve-preview` runs the same authz (active admin) as the execute — it discloses the roster and pool balance, so a plain member must never see it — and shows `forfeitReserve`, `inertBundle`, and per-artist recipients before the admin confirms.

`_require_live_org` (409 dissolved always; 409 archived unless `allow_archived=True`) guards every mutating lifecycle endpoint (invite/accept, member cap/role, credit-requests, project-member grants, claim-coverage, transfer-credits). Reads (`get_org`, dissolve-preview, usage, ledger) and `cancel-topup` are deliberately exempt — `cancel-topup` is the documented retry path for a Stripe cancel that failed mid-archive/dissolve, so a dead org must still be able to stop its own charge.

### Funding the pool

Self-serve orgs have no dispersal. Every credit lands in the pool's **reserve** bucket (never expires) through one of three admin-only inlets:

1. **Transfer** — `POST /orgs/{id}/transfer-credits`, backed by the `transfer_credits` RPC (migration `20260816000002`). Personal reserve → org pool only (bundle credits are never transferable — moving them would silently end their monthly expiry). Takes a `FOR UPDATE` lock on the source wallet and re-checks the balance under that lock, so two concurrent transfers can't both slip past an insufficient-reserve check; a 409 with the caller's (freshly re-read) `reserveBalance` is returned rather than a silent clamp. Idempotent on `request_id` — a retried transfer reads as `duplicate: true`, not a double-spend.
2. **Packs** — the existing credit-pack purchase flow, targetable at an org (Phase B: `target=org_id` in the Stripe Checkout metadata).
3. **Recurring top-up** — `POST /subscriptions/org-topup-checkout`, riding the **purchasing admin's own personal Stripe customer** (never a separate org customer). The subscription's metadata carries `kind='org_topup'` + `org_id` — the personal subscription webhook handlers (`handle_subscription_updated`, `handle_subscription_deleted`, etc.) early-return on that metadata shape, so an org top-up can never be mistaken for, or overwrite, the purchaser's own plan. `invoice.paid` grants pool reserve, idempotent on the Stripe invoice id. Canceling: `POST /orgs/{id}/cancel-topup` (manual, `trigger="manual"`), automatically on dissolve (`trigger="dissolve"`), and automatically when the purchasing admin is offboarded (`orgs.service._cancel_topup_if_purchaser`, `trigger="offboard"`) — three call sites, one event name, distinguished by the `trigger` property.

### Storage: a separate per-owner pool, plus Pro PAYG

Team storage is **not** the enterprise `ENTERPRISE_SEAT_STORAGE_BYTES × seats` cap, and it is **not** personal storage. `orgs.storage_guard.pool_state(owner_id)` sums `organizations.storage_bytes` across every self-serve org `covered_by` still points at — **active and archived, excluding only dissolved** (a parked archived team's bytes still count, so archive-then-recreate can't dodge the cap).

- **Basic:** 100 GiB hard cap (matches personal storage; `20260817000001`). Uploads that would exceed it are refused (`storage_guard.upload_allowed`, shared message `TEAM_STORAGE_FULL_MSG`).
- **Pro:** 250 GiB, then PAYG. "Pro-like" (`TeamDials.tier == "pro"`, which also covers Msanii admin dials — keyed on tier, not pool size, since Basic's pool is now as big as Pro's used to be) owners always pass the upload gate and the reactivation guard — overage bills instead of blocking.
- **PAYG billing:** a daily sweep step (`storage_guard.bill_team_storage_overage`) turns overage into a Stripe InvoiceItem on the owner's **personal** customer, once per owner-period (stamped on `subscriptions.last_team_storage_invoiced_period`, keyed off the personal wallet's `period_start` — zero overage still stamps, or a fully-covered owner would be re-evaluated forever). Rate is `TEAM_STORAGE_OVERAGE_USD_PER_GB` (env, seeded $0.025/GB/mo). Billing is **per owner**, not per org — `pool_state` already sums across every org that owner covers, so per-org billing would double-count.
- **Reactivation guard:** `reactivation_allowed` — a lapsed or archived team may not wake up (via coverage claim, unarchive, or the standing sweep's recovery branch) while the owner sits over-pool, unless they're pro-like (PAYG absorbs it).
- Two other byte inlets besides upload: `artist_subtree_bytes` sizes an artist transfer before it happens (so a transfer that would blow the pool can be refused up front), and the reactivation guard above.

### Boards

Spec `2026-08-16-boards-on-teams`. The workspace used to have its **own** notion of a team (`teams` / `team_members` / `pending_team_invites`, the backend package `src/backend/teams/`, the Workspace "Teams" tab) sitting beside this one. They are now one thing: **a board belongs to an organization, or to one person.**

- **The edge.** `boards.team_id → organizations(id) ON DELETE RESTRICT` — the same edge as `artists.team_id`, and the column finally means what its name says. `NULL` = personal board. Membership is the org's ACTIVE seats in a LIVE org; **invites happen only in the `/teams` console**, never in the workspace.
- **One predicate, two languages.** SQL `can_access_board(p_board_id, p_user_id)` (RLS on `boards`, `board_members`, `board_task_assignees`, `board_task_works`) is mirrored by `boards/authz._can_access` in Python, whose liveness half is `artist_access.live_org_ids` — the same "live seat" definition team artists use. Keep them in step:

  ```
  personal → owner
  team     → live seat AND (NOT boards.restricted OR owner OR org admin OR listed in board_members)
  ```
- **RLS is not the writer-side gate on its own.** A `FOR UPDATE` policy with no `WITH CHECK` reuses `USING` as the check, so "can see it" would mean "can write it". The `boards` UPDATE policy is therefore **owner-or-admin, not `can_access_board`**, and because `WITH CHECK` cannot see `OLD`, the `boards_lock_team_id` BEFORE UPDATE trigger refuses any `team_id` change made under an end-user JWT (same shape as `artists_lock_team_id`). Without both, a plain member could flip `restricted` — locking colleagues out — or move a board to another org they belong to, straight from the anon-key client the frontend ships. `is_live_org_member` (SQL) backs all three write policies, so a member of an archived/lapsed org cannot INSERT or DELETE a board there either.
- **Restricted narrowing is a visibility list, not a role.** `PUT /boards/boards/{id}` gained `restricted` and `member_user_ids` (a **replace-set**; every id must be an ACTIVE seat of the board's org, else 422), gated to an org admin **or the board's creator**. Rename/description stay open to anyone who can see the board; archive / delete / restore / the archived list stay org-admin. Assigning a task requires the **target** to satisfy `can_access_board` — you can't be assigned to a board you can't open.
- **Roster.** `GET /orgs/{org_id}/members` is the member-visible roster (names/avatars, no emails — those stay on the admin-only `/usage` seats) that feeds the assignee picker, the "Created by" filter and the board-members picker.
- **Free = personal boards only**, and that is not a boards check: a Free user simply can't *own* an org (`standing.require_free_slot` inside `create_org`, which only runs when `standing.self_serve_enabled()` — both flags on; with self-serve off there is no slot check at all). Joining someone else's team is free on every tier, and a Free member of a Basic-covered team may create boards in it.
- **Lifecycle follows the org.** Archived or lapsed → the org's boards are invisible to everyone, admins included, and come back on unarchive/reactivate. **Removing** a member purges their `board_members` rows on that org's boards (`_purge_member_from_org_boards`) so a re-invite can't silently restore narrowed access; **suspending purges nothing** — it is reversible and nothing would restore deleted rows. Task assignments survive both: an inactive seat already fails the predicate, so deleting history buys nothing. **Dissolve** reverts each board to a personal board of its creator, or of the dissolving admin when the creator no longer holds a seat (`_revert_org_boards`, the same recipient rule as `_dissolve_recipients`) — not best-effort: a board still pointing at a dissolved org is reachable by nobody.
- **Migrations, in this order.** `20260818000001_boards_to_orgs.sql` (repoint the FK, add `restricted` + `board_members`, the predicate and the policies; pre-existing board-team boards become their creators' **personal** boards) — apply immediately after the new backend is live, it is forward-breaking for the old one. Then, only once that is verified, `20260818000002_drop_board_teams.sql` drops the old tables, their triggers (including `process_pending_team_invites_on_signup`, which lives on `auth.users` and would otherwise survive), `is_team_member`/`is_team_admin`, and `'team_invite'` from `notifications_type_check`.
- **QA.** `scripts/qa_boards_on_teams.py` — the empty-`member_user_ids` save, the 422, the suspend/remove footprint and the `boards_lock_team_id_trg` refusal are all things `MockQueryBuilder` cannot express, so pytest cannot cover them.

## Not built (deliberate)

- **Per-artist permissions inside a team.** Every member of a team sees every artist the team owns. A real label requirement eventually, and cheap to add because every decision routes through `can_access_artist` — it is a change to one function body. When it happens: the default must stay *open* (an opt-in **restriction**, not an opt-in grant, or every existing team goes dark on deploy), and it wants a join table `org_artist_access(org_id, member_id, artist_id)` rather than an array column, because the admin console needs "who can see artist X".
- **Team → personal transfer.** One-way in v1; support moves it back by hand.
- **Copy detection.** Alerting an admin that some user's *personal* artist matches one of theirs would disclose a private row belonging to someone who may have no relationship with the team, and the false-positive rate (the artist's own profile, two managers sharing a client, common names) would train admins to ignore it.
