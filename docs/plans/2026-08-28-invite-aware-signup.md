# Invite-aware signup & onboarding

**Date:** 2026-08-28
**Status:** Steps 1–2 implemented; step 3 deferred.

## Problem

A user invited to a team who has no account yet clicks "Sign Up to Join"
in the invite email (`/orgs/invite/{token}`). Two things go wrong:

1. **The invite is lost during signup.** The claim page bounces signed-out
   visitors to `/auth?redirect=/orgs/invite/{token}`, but only sign-in and
   Google honour `redirect`. Email/password signup always routes
   `/auth/confirm-email` → `/onboarding` → `/dashboard`, and nothing persists
   the token. New invitees also get no in-app notification (only existing
   users do), so the emailed link is their only way in.
2. **They are asked to pick Free vs Basic.** Onboarding step 3 (`StepPlan`)
   is unconditional. A team member keeps their own (free) tier and spends
   from the org pool — Basic buys them nothing.

## Design

Frontend-only; no backend or schema changes. One small sessionStorage
"sticky note" (`src/lib/pendingInvite.ts`) carries the invite across the
signup detour, and onboarding reads it to swap the plan step.

### Step 1 — carry the invite through signup

- `Auth.tsx` signup: if `?redirect=` is an org-invite path, stash it and pass
  it to `signUp(email, password, name, redirectPath)`.
- `AuthContext.signUp` sets `emailRedirectTo` to `${origin}${redirectPath}`
  (default `/onboarding`), so the confirmation link — which may open in a new
  tab with empty sessionStorage — still lands on the invite page.
- `ConfirmEmail.tsx` polls as before but navigates to the stashed redirect
  when one exists.
- `OrgInviteClaim.tsx` is the source of truth for the stash: the signed-out
  gate stashes `{token, accepted: false}`; a successful Accept stashes
  `{token, orgName, kind, accepted: true}`; Decline clears it. After Accept
  the page navigates to `/teams`; `ProtectedRoute` sends a not-yet-onboarded
  user to `/onboarding`, which now knows about the team.

### Step 2 — don't sell a plan to an invitee

- `Onboarding.tsx` reads the stash. If `accepted`, step 3 renders
  `StepTeamInvite` ("You're on {org}'s team — no plan to choose") instead of
  `StepPlan`. If not accepted, it checks `GET /orgs/invites/{token}/preview`;
  a live invite renders the same step ("You're joining {org}") and the final
  step's button becomes "Review invitation" → `/orgs/invite/{token}`. A dead
  invite (404) falls back to the normal plan step — the user genuinely needs
  a plan then.
- Free stays the default under the hood (no subscription write), which is
  correct: members are `tier=free` with org-shaped entitlements.

### Step 3 — email-matched fallback (deferred)

`GET /me/pending-invites` matching unexpired `pending_org_invites` by the
signed-in email, consumed by onboarding (and later a dashboard banner) when
the stash is missing (different browser, cleared storage).

## Out of scope

- Auto-joining on signup (explicit Accept stays — the invite is only a
  capability for the emailed address).
- Changing a member's tier — Free + org pool is already correct.
- Existing users on Basic who later get invited.

## Verification

- `npm run build`, `npm test` (new `pendingInvite.test.ts`).
- Manual: invite a fresh email → Sign Up (email/password) → confirm → invite
  page → Accept → onboarding shows team step, no plan → dashboard. Repeat via
  Google. Decline path returns to the normal plan step.
