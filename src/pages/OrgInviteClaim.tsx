// src/pages/OrgInviteClaim.tsx
// Licensing Phase B (spec §7, plan Task 13) — the /orgs/invite/:token claim
// page. Mirrors src/pages/InviteClaim.tsx's (registry collaborator invite)
// shell/gate structure. A best-effort GET preview names the org, but nothing
// gates on it — Accept / Decline call the real POST endpoints directly and
// the resulting success/error shape (200 body `type`, or 403/410/404) drives
// which screen renders. That's also why `useAcceptOrgInvite`/`useDeclineOrgInvite`
// (src/hooks/useOrgs.ts) don't auto-toast on error like this file's other
// hooks — this page owns the distinct expired/wrong-email/not-found copy.
import { useEffect, useState } from "react";
import { useParams, useNavigate, useSearchParams } from "react-router-dom";
import { Loader2, Building2, ShieldAlert } from "lucide-react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { ApiError } from "@/lib/apiFetch";
import { useAcceptOrgInvite, useDeclineOrgInvite, useOrgInvitePreview } from "@/hooks/useOrgs";
import { orgNoun } from "@/lib/tiers";
import {
  clearPendingInvite,
  normalizeInviteEmail,
  readPendingInvite,
  stashPendingInvite,
  type AuthInviteState,
  type AuthTab,
} from "@/lib/pendingInvite";

const authPath = (invitePath: string) => `/auth?redirect=${encodeURIComponent(invitePath)}`;

const Shell = ({ children }: { children: React.ReactNode }) => (
  <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-secondary to-background p-4">
    <div className="w-full max-w-md">{children}</div>
  </div>
);

/**
 * Signed-out gate. The invite route is public so we can show this; both the
 * accept and decline endpoints require auth, so neither fires until the user
 * is signed in (same idiom as InviteClaim.tsx's SignedOut).
 */
const SignedOut = ({
  invitePath,
  token,
  email,
  tab,
}: {
  invitePath: string;
  token: string;
  email: string | null;
  tab: AuthTab;
}) => {
  const navigate = useNavigate();
  // Remember the invite before the auth detour so a new signup's onboarding
  // knows a team is waiting (Google OAuth round-trips keep sessionStorage).
  // The email rides along so /auth can prefill it after a reload too.
  useEffect(() => {
    stashPendingInvite({ token, accepted: false, email });
  }, [token, email]);
  // The prefill hint goes to /auth in router STATE, never the URL.
  const state: AuthInviteState = { email, tab };
  return (
    <Shell>
      <Card>
        <CardHeader className="text-center">
          <div className="mx-auto mb-2 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
            <Building2 className="h-6 w-6 text-primary" />
          </div>
          <CardTitle>You&apos;ve been invited to join a team</CardTitle>
          <CardDescription>
            Sign in or create an account to review and accept the invitation.
          </CardDescription>
        </CardHeader>
        <CardFooter>
          <Button className="w-full" onClick={() => navigate(authPath(invitePath), { state })}>
            Sign in / Create account
          </Button>
        </CardFooter>
      </Card>
    </Shell>
  );
};

const CenteredMessage = ({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: React.ReactNode;
}) => (
  <Shell>
    <Card>
      <CardHeader className="text-center">
        <div className="mx-auto mb-2 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
          <ShieldAlert className="h-6 w-6 text-primary" />
        </div>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      {action ? <CardFooter>{action}</CardFooter> : null}
    </Card>
  </Shell>
);

type ErrorKind = "expired" | "invalid" | "wrong_email" | "not_found";

/**
 * Authenticated body. Only mounted once `user` exists (the route itself
 * handles the signed-out bounce below) — there's no preview fetch to guard.
 */
const OrgInviteClaimAuthed = ({ token, email }: { token: string; email: string | null }) => {
  const navigate = useNavigate();
  const { signOut } = useAuth();
  const acceptInvite = useAcceptOrgInvite();
  const declineInvite = useDeclineOrgInvite();
  const [errorState, setErrorState] = useState<ErrorKind | null>(null);
  const [declined, setDeclined] = useState(false);

  // Best-effort invite preview so the card can NAME the org. It 404s for an
  // unknown/expired token — any failure just leaves the generic
  // "an organization" copy below (also the right neutral default while it
  // loads).
  const { data: invitePreview } = useOrgInvitePreview(token);
  const previewOrgName = invitePreview?.orgName ?? null;
  const previewNoun = orgNoun(invitePreview?.kind);
  const previewArticle = previewNoun === "team" ? "a" : "an";

  const handleAccept = async () => {
    setErrorState(null);
    try {
      await acceptInvite.mutateAsync(token);
      // A brand-new account gets bounced from /teams into onboarding by
      // ProtectedRoute; the accepted stash lets onboarding skip the plan
      // step and name the team instead.
      stashPendingInvite({
        token,
        accepted: true,
        orgName: previewOrgName,
        kind: invitePreview?.kind ?? null,
      });
      // The preview already named the org; fall back to the generic label
      // when it failed to load.
      const orgName = previewOrgName ?? `your ${previewNoun}`;
      toast.success(`You're on ${orgName}'s license — your work now runs on their credits`);
      navigate("/teams");
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 403) {
          setErrorState("wrong_email");
          return;
        }
        if (err.status === 410) {
          setErrorState(err.message.toLowerCase().includes("expired") ? "expired" : "invalid");
          return;
        }
      }
      setErrorState("not_found");
    }
  };

  const handleDecline = async () => {
    try {
      await declineInvite.mutateAsync(token);
      clearPendingInvite();
      setDeclined(true);
    } catch {
      toast.error("Couldn't decline the invitation. Please try again.");
    }
  };

  if (errorState === "wrong_email") {
    return (
      <CenteredMessage
        title="Wrong account"
        description={
          email
            ? `This invite was sent to ${email}. Sign in with that account to accept it.`
            : "This invite was sent to a different email address. Sign in with that account to accept it."
        }
        action={
          <Button
            variant="outline"
            className="w-full"
            onClick={async () => {
              // Stash BEFORE signing out: once the session clears, the
              // signed-out gate re-mounts and re-stashes, so every write
              // must carry the same email or the prefill is lost.
              stashPendingInvite({ token, accepted: false, email });
              await signOut();
              const state: AuthInviteState = { email, tab: "signin" };
              navigate(authPath(`/orgs/invite/${token}`), { state });
            }}
          >
            Sign out
          </Button>
        }
      />
    );
  }

  if (errorState === "expired") {
    return (
      <CenteredMessage
        title="This invitation has expired"
        description={`Ask ${previewOrgName ? `${previewOrgName}'s` : `the ${previewNoun}'s`} admin to send a new invite, then open the new link.`}
      />
    );
  }

  if (errorState === "invalid" || errorState === "not_found") {
    return (
      <CenteredMessage
        title="Invitation not found"
        description="This invitation may have already been used, declined, or the link is invalid."
      />
    );
  }

  if (declined) {
    return (
      <CenteredMessage
        title="Invitation declined"
        description={`You've declined this invitation. If that was a mistake, ask ${previewOrgName ? `${previewOrgName}'s` : `the ${previewNoun}'s`} admin to send a new one.`}
      />
    );
  }

  return (
    <Shell>
      <Card>
        <CardHeader className="text-center">
          <div className="mx-auto mb-2 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
            <Building2 className="h-6 w-6 text-primary" />
          </div>
          <CardTitle>
            {previewOrgName
              ? `You've been invited to join ${previewOrgName}`
              : `You've been invited to join ${previewArticle} ${previewNoun}`}
          </CardTitle>
          <CardDescription>
            Accepting moves your Msanii credits and billing to their shared pool. Your artists, projects,
            and files stay exactly as they are.
          </CardDescription>
        </CardHeader>
        <CardFooter className="flex flex-col gap-2">
          <Button
            className="w-full"
            onClick={handleAccept}
            disabled={acceptInvite.isPending || declineInvite.isPending}
          >
            {acceptInvite.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Accepting…
              </>
            ) : (
              "Accept invitation"
            )}
          </Button>
          <Button
            variant="ghost"
            className="w-full"
            onClick={handleDecline}
            disabled={acceptInvite.isPending || declineInvite.isPending}
          >
            {declineInvite.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Declining…
              </>
            ) : (
              "Decline"
            )}
          </Button>
        </CardFooter>
      </Card>
    </Shell>
  );
};

const OrgInviteClaim = () => {
  const { token } = useParams<{ token: string }>();
  const { user, loading } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();

  // The emailed link carries `?email=<invitee>&signup=1` (orgs/emails.py) so
  // /auth can prefill the address and open the right tab. Latch both on
  // mount, then scrub them from the address bar (`replace`, so Back/refresh
  // don't resurrect them) — done HERE, above the auth gate, so it runs for
  // signed-out and already-signed-in arrivals alike. The Google OAuth
  // landing comes back without the params, so the session stash (written
  // by the signed-out gate) is the fallback. Neither value is trusted by
  // the server: accept still matches the signed-in account to the invite.
  const [hint] = useState(() => ({
    email: normalizeInviteEmail(searchParams.get("email")),
    signup: searchParams.get("signup") === "1",
  }));
  const [stashed] = useState(() => readPendingInvite());
  const hadParams = searchParams.has("email") || searchParams.has("signup");
  useEffect(() => {
    if (!hadParams) return;
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.delete("email");
        next.delete("signup");
        return next;
      },
      { replace: true },
    );
  }, [hadParams, setSearchParams]);
  const inviteEmail = hint.email ?? (stashed && stashed.token === token ? stashed.email ?? null : null);
  const authTab: AuthTab = hint.signup ? "signup" : "signin";

  if (!token) {
    return (
      <CenteredMessage
        title="Invalid invitation link"
        description="This link is missing its invitation token."
      />
    );
  }

  if (loading) {
    return (
      <Shell>
        <Card>
          <CardContent className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </CardContent>
        </Card>
      </Shell>
    );
  }

  // Logged-out gate: the route is public (invited members may not have an
  // account/session yet), but both invite actions require auth — bounce
  // through /auth with a redirect back to this exact invite link, same
  // pattern as the registry collaborator invite claim (InviteClaim.tsx).
  if (!user) {
    return <SignedOut invitePath={`/orgs/invite/${token}`} token={token} email={inviteEmail} tab={authTab} />;
  }

  return <OrgInviteClaimAuthed token={token} email={inviteEmail} />;
};

export default OrgInviteClaim;
