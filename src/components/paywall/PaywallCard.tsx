import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Lock } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { cn } from "@/lib/utils";
import type { GatedFeature, CountableResource } from "@/hooks/useEntitlements";
import { useAnalytics, type PaywallFeature } from "@/hooks/useAnalytics";
import { UnlinkProjectHint } from "@/components/paywall/creditWall";

interface PaywallCardProps {
  feature?: GatedFeature;
  resource?: CountableResource;
  reason?: string;
  onUpgrade?: () => void;
  variant?: "page" | "inline" | "modal";
  /** Licensing Phase B (spec §5, rule 8, plan Task 13): true when the denial
   * came from a seat wallet in ORG billing context. There's no upgrade or
   * overage path on a seat — the member asks their org admin instead, so
   * this branch REPLACES the upgrade CTA with a "Request credits" one. */
  managedByOrg?: boolean;
  /** Where the "Request credits" CTA navigates — defaults to /organization
   * (the member view's request form, plan Task 13) when the 402 detail
   * didn't carry one. */
  requestUrl?: string;
  /** Licensing Phase C (spec §6/§11 rule 11c, plan Task 8): true when this
   * wall is a dry ORG seat on a project the CALLER OWNS and can unlink — the
   * backend only sets this after confirming ownership (Task 6, landing
   * separately), so it's safe to render unconditionally once present. Always
   * renders ALONGSIDE the "Request credits" CTA above, never instead of it —
   * an owner who's also an org admin still needs both escape hatches. */
  ownerCanUnlink?: boolean;
  /** The project to unlink, for the hint's link target. Required alongside
   * `ownerCanUnlink` in practice (Task 6 always sets both together); the
   * plain-text fallback below covers the defensive case where it's missing. */
  projectId?: string;
  /** Project display name for the hint's link text — falls back to generic
   * "this project" wording when absent. */
  projectName?: string;
}

const FEATURE_LABELS: Record<GatedFeature, string> = {
  zoe: "Zoe AI",
  oneclick: "OneClick",
  registry: "Metadata Registry",
};

const RESOURCE_LABELS: Record<CountableResource, string> = {
  artist: "artists",
  project: "projects",
  task: "tasks",
};

export const PaywallCard = ({
  feature,
  resource,
  reason,
  onUpgrade,
  variant = "page",
  managedByOrg,
  requestUrl,
  ownerCanUnlink,
  projectId,
  projectName,
}: PaywallCardProps) => {
  const navigate = useNavigate();
  const { capturePaywallShown, capturePaywallUpgradeClicked } = useAnalytics();
  const pwFeature = (feature ?? "create_cap") as PaywallFeature;

  useEffect(() => {
    capturePaywallShown(pwFeature, variant ?? "page", reason);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const upgrade = onUpgrade ?? (() => navigate("/pricing"));
  const handleUpgradeClick = () => {
    capturePaywallUpgradeClicked(pwFeature, variant ?? "page");
    upgrade();
  };
  const handleRequestCredits = () => navigate(requestUrl || "/organization");

  const title = managedByOrg
    ? "Your organization is out of credits"
    : feature
    ? `${FEATURE_LABELS[feature]} is a Pro feature`
    : resource
    ? "You've reached your Free tier limit"
    : "Upgrade to Pro";

  const body = managedByOrg
    ? reason ?? "You've used the credits your organization allocated. Ask your admin for more."
    : reason ??
      (feature
        ? `${FEATURE_LABELS[feature]} is included in Pro. Upgrade to access it.`
        : resource
        ? `You've used all your free ${RESOURCE_LABELS[resource]}. Upgrade to Pro for unlimited.`
        : "Upgrade to Pro to unlock more features.");

  return (
    <Card
      className={cn(
        variant === "page" ? "max-w-md mx-auto my-12 p-8" : "p-6 border-0 shadow-none",
      )}
    >
      <div className="flex flex-col items-center text-center gap-3">
        <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
          <Lock className="w-5 h-5 text-muted-foreground" />
        </div>
        <h2 className="text-xl font-semibold">{title}</h2>
        <p className="text-muted-foreground text-sm max-w-sm">{body}</p>
        {managedByOrg ? (
          <>
            <Button onClick={handleRequestCredits} className="mt-2">
              Request credits
            </Button>
            {ownerCanUnlink && (
              <UnlinkProjectHint projectId={projectId} projectName={projectName} />
            )}
          </>
        ) : (
          <Button onClick={handleUpgradeClick} className="mt-2">
            Upgrade to Pro
          </Button>
        )}
      </div>
    </Card>
  );
};
