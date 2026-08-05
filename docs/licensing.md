# Licensing & Teams

Everything behind `LICENSING_ENABLED`: organizations ("Teams" in the UI), the shared credit pool, per-member caps, and **team-owned artist profiles**.

> **The flag is a true rollback.** With `LICENSING_ENABLED` unset, every `/orgs/*` route 404s at the router level, `resolve_billing_org_for_project` short-circuits before any query, and `useActiveTeam()` returns null so no team affordance renders. Nobody can be in a team, so every artist has `team_id IS NULL` and every code path is the one that shipped before licensing existed.

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

This matters because every policy written before licensing reads `artists.user_id = auth.uid()`. `20260803000002` re-scopes all 21 of them with `AND a.team_id IS NULL` — without that, the creator of a team artist would keep personal-owner rights over the whole subtree **forever, including after being offboarded** (`can_access_artist` would deny; the old policy would grant anyway, and permissive policies OR).

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

Both default `team_id` to the active billing-context org and show `TeamOwnershipField` with a "Keep this artist private to me" escape hatch. Because there is no backend endpoint, **RLS is the only enforcement**: `artists_insert_team` requires an ACTIVE membership in the target org *and* pins `user_id` to the caller, so neither the team nor the creator stamp can be forged.

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

### What is NOT protected

A member who can read a team artist can retype its name, email and splits into a personal artist. That is not preventable by RLS and nothing in the app pretends otherwise — there is deliberately no "duplicate artist" affordance, but facts leaving in someone's head is a contract problem. What *is* enforced is the assets: file rows cannot cross an ownership boundary, and DSP credentials are admin-only (read included).

---

## Credits: the pool and the caps

An org negotiates a monthly credit volume, set **only by a Msanii admin** (`PUT /admin/orgs/{id}/dispersal`) — never by the org's own admin. Any signed-in user can create an org and is auto-made its admin, and dispersed credits count toward the activation floor, so a customer-writable dial would mint free credits and self-activate.

- The org holds **one** pool wallet (`credit_wallets`, `owner_type='org'`). The daily sweep grants `organizations.monthly_dispersal_credits` into the **expiring** bundle bucket, so an unspent month can't be banked; purchased packs land in reserve and never expire.
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

| Method | Path | Who |
|---|---|---|
| POST / GET | `/orgs` | any signed-in user (creator becomes admin) |
| GET / PUT | `/orgs/{org_id}` | member / admin (`default_member_cap` lives here) |
| POST | `/orgs/{org_id}/archive` | admin |
| GET | `/orgs/{org_id}/usage` | member |
| POST / GET / DELETE | `/orgs/{org_id}/invites`, `/invites/{invite_id}` | admin |
| POST | `/orgs/invites/{token}/accept` · `/decline` | invitee |
| PUT | `/orgs/{org_id}/members/{member_id}/role` · `/cap` | admin |
| POST | `/orgs/{org_id}/members/{member_id}/suspend` · `/reactivate` | admin |
| DELETE | `/orgs/{org_id}/members/{member_id}` | admin |
| POST / GET | `/orgs/{org_id}/credit-requests` | member creates, admin lists |
| POST | `/orgs/{org_id}/credit-requests/{id}/approve` · `/deny` | admin |
| GET | `/orgs/{org_id}/projects` | admin — the projects this org owns, via artist ownership |
| **POST** | **`/orgs/{org_id}/artists/{artist_id}/transfer`** | the artist's personal owner, who must hold a seat |
| PUT / DELETE | `/orgs/{org_id}/projects/{project_id}/members/{member_id}` | admin — grant/adjust/revoke seat access |

### Msanii admin (`/admin`)

| Method | Path | Purpose |
|---|---|---|
| PUT | `/admin/orgs/{org_id}/dispersal` | set the monthly credit volume |
| POST | `/admin/orgs/{org_id}/suspend` · `/reactivate` | lifecycle |
| GET | `/admin/orgs/{org_id}/pool` | pool balance |
| POST | `/admin/orgs/{org_id}/pool/clawback` | reserve-only |

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
| `ENTERPRISE_SEAT_STORAGE_BYTES` | `500 GiB` | **Per seat.** A team's cap is this × active seats |

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

`gates_team_artists.sql` builds a throwaway org, members, artists, projects and files, asserts, then **RAISEs on purpose** — the error message *is* the report, and the raise is what rolls the test data back. It needs at least 3 rows in `auth.users`. Expected totals: **6** after `20260803000001`, **14** after `…0002`, **16** after `…0003`.

Every PL/pgSQL variable in those scripts is `v_`-prefixed. PL/pgSQL's `variable_conflict` defaults to `error`, so a variable named `org_id` in `WHERE org_id = org_id` aborts the whole block with "column reference org_id is ambiguous".

---

## Not built (deliberate)

- **Per-artist permissions inside a team.** Every member of a team sees every artist the team owns. A real label requirement eventually, and cheap to add because every decision routes through `can_access_artist` — it is a change to one function body. When it happens: the default must stay *open* (an opt-in **restriction**, not an opt-in grant, or every existing team goes dark on deploy), and it wants a join table `org_artist_access(org_id, member_id, artist_id)` rather than an array column, because the admin console needs "who can see artist X".
- **Team → personal transfer.** One-way in v1; support moves it back by hand.
- **Copy detection.** Alerting an admin that some user's *personal* artist matches one of theirs would disclose a private row belonging to someone who may have no relationship with the team, and the false-positive rate (the artist's own profile, two managers sharing a client, common names) would train admins to ignore it.
