// src/pages/OrgInviteClaim.tsx
// Licensing Phase B (spec §7, plan Task 13) — the /orgs/invite/:token claim
// page. Mirrors src/pages/InviteClaim.tsx's (registry collaborator invite)
// shell/gate structure, with one structural difference: the orgs backend has
// no GET preview endpoint (unlike registry's /invite/{token}/preview), so
// there's nothing to pre-fetch and show before the user commits — Accept /
// Decline call the real POST endpoints directly and the resulting
// success/error shape (200 body `type`, or 403/410/404) drives which screen
// renders. That's also why `useAcceptOrgInvite`/`useDeclineOrgInvite`
// (src/hooks/useOrgs.ts) don't auto-toast on error like this file's other
// hooks — this page owns the distinct expired/wrong-email/not-found copy.
import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Loader2, Building2, ShieldAlert } from "lucide-react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, ApiError, apiFetch } from "@/lib/apiFetch";
import { useAcceptOrgInvite, useDeclineOrgInvite } from "@/hooks/useOrgs";

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
const SignedOut = ({ invitePath }: { invitePath: string }) => {
  const navigate = useNavigate();
  return (
    <Shell>
      <Card>
        <CardHeader className="text-center">
          <div className="mx-auto mb-2 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
            <Building2 className="h-6 w-6 text-primary" />
          </div>
          <CardTitle>You&apos;ve been invited to join an organization</CardTitle>
          <CardDescription>
            Sign in or create an account to review and accept the invitation.
          </CardDescription>
        </CardHeader>
        <CardFooter>
          <Button className="w-full" onClick={() => navigate(`/auth?redirect=${encodeURIComponent(invitePath)}`)}>
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
const OrgInviteClaimAuthed = ({ token }: { token: string }) => {
  const navigate = useNavigate();
  const { signOut } = useAuth();
  const acceptInvite = useAcceptOrgInvite();
  const declineInvite = useDeclineOrgInvite();
  const [errorState, setErrorState] = useState<ErrorKind | null>(null);
  const [declined, setDeclined] = useState(false);

  const handleAccept = async () => {
    setErrorState(null);
    try {
      const result = await acceptInvite.mutateAsync(token);
      // Best-effort: fetch the org's name for the welcome toast — the accept
      // response only carries org_id. A failure here never blocks the
      // (already-successful) accept from landing the user on /organization.
      let orgName = "your organization";
      try {
        const org = await apiFetch<{ name: string }>(`${API_URL}/orgs/${result.org_id}`);
        orgName = org.name;
      } catch {
        // fall back to the generic label above
      }
      toast.success(`You're on ${orgName}'s license — your work now runs on their credits`);
      navigate("/organization");
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
      setDeclined(true);
    } catch {
      toast.error("Couldn't decline the invitation. Please try again.");
    }
  };

  if (errorState === "wrong_email") {
    return (
      <CenteredMessage
        title="Wrong account"
        description="This invite was sent to a different email address. Sign in with that account to accept it."
        action={
          <Button
            variant="outline"
            className="w-full"
            onClick={async () => {
              await signOut();
              navigate(`/auth?redirect=${encodeURIComponent(`/orgs/invite/${token}`)}`);
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
        description="Ask the organization's admin to send a new invite, then open the new link."
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
        description="You've declined this invitation. If that was a mistake, ask the organization's admin to send a new one."
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
          <CardTitle>You&apos;ve been invited to join an organization</CardTitle>
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
    return <SignedOut invitePath={`/orgs/invite/${token}`} />;
  }

  return <OrgInviteClaimAuthed token={token} />;
};

export default OrgInviteClaim;
