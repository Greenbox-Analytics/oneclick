/**
 * Carries an org invite across the signup detour.
 *
 * The invite email links to /orgs/invite/{token}. A signed-out visitor is
 * bounced through /auth, and an email/password signup then detours through
 * /auth/confirm-email → /onboarding. Nothing else remembers the token on that
 * path, so this sessionStorage "sticky note" does. OrgInviteClaim is the
 * writer (signed-out gate, Accept, Decline); ConfirmEmail and Onboarding read
 * it. Session-scoped on purpose: it should not outlive the tab.
 */

export const ORG_INVITE_PATH_PREFIX = "/orgs/invite/";

const KEY = "msanii_pending_org_invite";

export interface PendingInvite {
  token: string;
  /** True once the user has accepted on the claim page. */
  accepted: boolean;
  orgName?: string | null;
  kind?: "self_serve" | "enterprise" | null;
}

export function orgInviteTokenFromPath(path: string | null | undefined): string | null {
  if (!path || !path.startsWith(ORG_INVITE_PATH_PREFIX)) return null;
  const token = path.slice(ORG_INVITE_PATH_PREFIX.length).split(/[?#/]/)[0];
  return token || null;
}

export function orgInvitePath(token: string): string {
  return `${ORG_INVITE_PATH_PREFIX}${token}`;
}

export function readPendingInvite(): PendingInvite | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<PendingInvite>;
    if (typeof parsed.token !== "string" || !parsed.token) return null;
    return {
      token: parsed.token,
      accepted: parsed.accepted === true,
      orgName: parsed.orgName ?? null,
      kind: parsed.kind ?? null,
    };
  } catch {
    return null;
  }
}

export function stashPendingInvite(invite: PendingInvite): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(invite));
  } catch {
    /* sessionStorage unavailable — the emailed link still works */
  }
}

export function clearPendingInvite(): void {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}
