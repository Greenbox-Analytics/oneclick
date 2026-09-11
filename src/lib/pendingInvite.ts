/**
 * Carries an org invite across the signup detour.
 *
 * The invite email links to /orgs/invite/{token}. A signed-out visitor is
 * bounced through /auth, and an email/password signup then detours through
 * /auth/confirm-email → /onboarding. Nothing else remembers the token on that
 * path, so this sessionStorage "sticky note" does. OrgInviteClaim is the
 * writer (signed-out gate, Accept, Decline); ConfirmEmail and Onboarding read
 * it. Session-scoped on purpose: it should not outlive the tab.
 *
 * It also carries the invitee's EMAIL (from the emailed link's `?email=`),
 * which /auth uses to prefill the sign-in / sign-up forms. That is a UI hint
 * only — the server matches the signed-in account against the invite row at
 * accept time and never trusts this value.
 */

export const ORG_INVITE_PATH_PREFIX = "/orgs/invite/";

const KEY = "msanii_pending_org_invite";

export type AuthTab = "signin" | "signup";

/** The `location.state` contract between OrgInviteClaim and /auth. The
 * email never rides in the /auth URL (history, logs, analytics). */
export interface AuthInviteState {
  email?: string | null;
  tab?: AuthTab;
}

export interface PendingInvite {
  token: string;
  /** True once the user has accepted on the claim page. */
  accepted: boolean;
  orgName?: string | null;
  kind?: "self_serve" | "enterprise" | null;
  /** Invitee address from the emailed link — a prefill hint for /auth. */
  email?: string | null;
}

const MAX_EMAIL_LENGTH = 254;

/**
 * Loose shape check for a prefill hint that came off a URL: one `@` with
 * something on both sides, no whitespace, sane length. Anything else is
 * dropped rather than rendered into an input. Case is preserved.
 */
export function normalizeInviteEmail(raw: unknown): string | null {
  if (typeof raw !== "string") return null;
  const value = raw.trim();
  if (value.length < 3 || value.length > MAX_EMAIL_LENGTH) return null;
  if (/\s/.test(value)) return null;
  const at = value.indexOf("@");
  if (at < 1 || at !== value.lastIndexOf("@") || at === value.length - 1) return null;
  return value;
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
      email: normalizeInviteEmail(parsed.email),
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
