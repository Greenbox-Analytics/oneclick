import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Loader2, Shield, Users } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/contexts/AuthContext";
import {
  useInvitePreview,
  useClaimInvitation,
  useConfirmStake,
} from "@/hooks/useRegistry";
import { ApiError } from "@/lib/apiFetch";

interface PreviewStake {
  stake_type: string;
  percentage: number;
  holder_role: string | null;
}

interface PreviewTerm {
  label: string;
  value: string;
}

interface InvitePreview {
  expired?: boolean;
  email_mismatch?: boolean;
  invite_email?: string;
  work_title?: string | null;
  collaborator?: { name: string; role: string; terms: PreviewTerm[] };
  stakes?: PreviewStake[];
  work?: { title: string; project_id: string; artist_id: string } | null;
}

const Shell = ({ children }: { children: React.ReactNode }) => (
  <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-secondary to-background p-4">
    <div className="w-full max-w-md">{children}</div>
  </div>
);

/**
 * Signed-out gate. The invite route is public so we can show this; the preview
 * endpoint itself requires auth, so we never fetch it until the user is in.
 */
const SignedOut = ({ invitePath }: { invitePath: string }) => {
  const navigate = useNavigate();
  return (
    <Shell>
      <Card>
        <CardHeader className="text-center">
          <div className="mx-auto mb-2 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
            <Users className="h-6 w-6 text-primary" />
          </div>
          <CardTitle>You've been invited to collaborate on a work</CardTitle>
          <CardDescription>
            Sign in or create an account to review your split and accept the
            invitation.
          </CardDescription>
        </CardHeader>
        <CardFooter>
          <Button
            className="w-full"
            onClick={() =>
              navigate(`/auth?redirect=${encodeURIComponent(invitePath)}`)
            }
          >
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
          <Shield className="h-6 w-6 text-primary" />
        </div>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      {action ? <CardFooter>{action}</CardFooter> : null}
    </Card>
  </Shell>
);

const StakeLine = ({ stakes, type, label }: { stakes: PreviewStake[]; type: string; label: string }) => {
  const matched = stakes.filter((s) => s.stake_type === type);
  if (matched.length === 0) return null;
  const total = matched.reduce((sum, s) => sum + (s.percentage || 0), 0);
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{total}%</span>
    </div>
  );
};

/**
 * Authenticated body. Only mounted when `user` exists so useInvitePreview's
 * (auth-required) request never fires for a signed-out visitor.
 */
const InviteClaimAuthed = ({ token }: { token: string }) => {
  const navigate = useNavigate();
  const { signOut } = useAuth();
  const { data, isLoading, error } = useInvitePreview(token) as {
    data: InvitePreview | null | undefined;
    isLoading: boolean;
    error: unknown;
  };
  const claimInvitation = useClaimInvitation();
  const confirmStake = useConfirmStake();
  const [accepting, setAccepting] = useState(false);

  if (isLoading) {
    return (
      <Shell>
        <Card>
          <CardContent className="flex flex-col items-center gap-3 py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">
              Loading your invitation…
            </p>
          </CardContent>
        </Card>
      </Shell>
    );
  }

  if (error) {
    const status = error instanceof ApiError ? error.status : undefined;
    const message =
      error instanceof Error ? error.message.toLowerCase() : "";
    if (status === 410 || message.includes("expired")) {
      return (
        <CenteredMessage
          title="This invitation has expired"
          description="Ask the owner to resend it, then open the new link."
        />
      );
    }
    return (
      <CenteredMessage
        title="Invitation not found"
        description="This invitation may have already been handled or the link is invalid."
      />
    );
  }

  if (!data) {
    return (
      <CenteredMessage
        title="Invitation not found"
        description="This invitation may have already been handled or the link is invalid."
      />
    );
  }

  if (data.email_mismatch) {
    return (
      <CenteredMessage
        title="Wrong account"
        description={`This invite was sent to ${
          data.invite_email ?? "another email"
        }. You're signed in as a different account — sign in with that email to accept.`}
        action={
          <Button
            variant="outline"
            className="w-full"
            onClick={async () => {
              await signOut();
              navigate(
                `/auth?redirect=${encodeURIComponent(
                  `/tools/registry/invite/${token}`,
                )}`,
              );
            }}
          >
            Sign out
          </Button>
        }
      />
    );
  }

  const { collaborator, stakes = [], work } = data;
  if (!collaborator) {
    return (
      <CenteredMessage
        title="Invitation not found"
        description="This invitation may have already been handled or the link is invalid."
      />
    );
  }

  const workTitle = work?.title ?? "this work";

  const handleAccept = async () => {
    setAccepting(true);
    try {
      // Claim by token first — links this account to the invite and returns
      // the collaborator record (with work_id).
      const claimed = await claimInvitation.mutateAsync(token);
      // Then confirm by collaborator id — this is what flips status to
      // confirmed and surfaces the work under "Shared with me".
      await confirmStake.mutateAsync(claimed.id);
      // useConfirmStake already fires a success toast ("Stake confirmed") on its
      // own onSuccess — adding another here would double-toast, so we don't.
      navigate(
        claimed.work_id ? `/tools/registry/${claimed.work_id}` : "/tools/registry",
      );
    } catch {
      // The claim/confirm hooks already surface errors via toast — avoid a
      // second toast here. Just re-enable the button.
      setAccepting(false);
    }
  };

  return (
    <Shell>
      <Card>
        <CardHeader>
          <div className="mb-1 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
            <Users className="h-6 w-6 text-primary" />
          </div>
          <CardTitle>You've been invited to collaborate</CardTitle>
          <CardDescription>
            You've been listed as{" "}
            <span className="font-medium text-foreground">
              {collaborator.role}
            </span>{" "}
            on{" "}
            <span className="font-medium text-foreground">{workTitle}</span>.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="rounded-lg border bg-muted/30 p-4 space-y-2">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Your split
            </p>
            {stakes.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No ownership split was attached to this invitation.
              </p>
            ) : (
              <div className="space-y-1">
                <StakeLine stakes={stakes} type="master" label="Master" />
                <StakeLine
                  stakes={stakes}
                  type="publishing"
                  label="Publishing"
                />
              </div>
            )}
          </div>

          {collaborator.terms && collaborator.terms.length > 0 && (
            <>
              <Separator />
              <div className="space-y-2">
                <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Terms
                </p>
                <div className="space-y-1">
                  {collaborator.terms.map((term, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between text-sm"
                    >
                      <span className="text-muted-foreground">
                        {term.label}
                      </span>
                      <span className="font-medium">{term.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </CardContent>
        <CardFooter>
          <Button
            className="w-full"
            onClick={handleAccept}
            disabled={accepting}
          >
            {accepting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Accepting…
              </>
            ) : (
              "Accept invitation"
            )}
          </Button>
        </CardFooter>
      </Card>
    </Shell>
  );
};

const InviteClaim = () => {
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

  if (!user) {
    return <SignedOut invitePath={`/tools/registry/invite/${token}`} />;
  }

  return <InviteClaimAuthed token={token} />;
};

export default InviteClaim;
