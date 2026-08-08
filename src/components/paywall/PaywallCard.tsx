import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Lock } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { cn } from "@/lib/utils";
import { useEntitlements, type GatedFeature, type CountableResource } from "@/hooks/useEntitlements";
import { useAnalytics, type PaywallFeature } from "@/hooks/useAnalytics";

interface PaywallCardProps {
  feature?: GatedFeature;
  resource?: CountableResource;
  reason?: string;
  onUpgrade?: () => void;
  variant?: "page" | "inline" | "modal";
  /** Licensing Phase B (spec §5, rule 8): true when the denial came from an ORG
   * billing context. There's no upgrade or overage path on a pool — the member
   * asks their org admin instead, so this branch REPLACES the upgrade CTA. */
  managedByOrg?: boolean;
  /** True when the MEMBER hit their own monthly limit rather than the pool
   * running dry. Both are org walls; only the first has an action the member can
   * take (ask for a raise), so a dry pool must not show that CTA — it would send
   * them to a form that can't fix anything. */
  capReached?: boolean;
  /** Where the "Ask for a higher limit" CTA navigates — defaults to
   * /organization (the member view's request form) when the 402 didn't carry
   * one. */
  requestUrl?: string;
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
  capReached,
  requestUrl,
}: PaywallCardProps) => {
  const navigate = useNavigate();
  const { capturePaywallShown, capturePaywallUpgradeClicked } = useAnalytics();
  const { data: ent } = useEntitlements();
  const pwFeature = (feature ?? "create_cap") as PaywallFeature;
  // Is the viewer an ADMIN of the org whose pool ran dry? An admin can buy
  // credits themselves — "ask your admin" would be telling them to email
  // themselves.
  const orgRole =
    ent?.billingContext?.type === "org" ? ent.billingContext.role : ent?.credits?.managedByOrg?.role;
  const isOrgAdmin = orgRole === "admin";
  // Credit packs are only sellable when the credits system is on and the
  // wall is personal — never suggest packs on a legacy tier-gate wall.
  const packsAvailable = !managedByOrg && ent?.credits != null;

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

  const title = capReached
    ? "You've reached your monthly limit"
    : managedByOrg
    ? isOrgAdmin
      ? "Your team's credit pool is empty"
      : "Your organization is out of credits"
    : feature
    ? `${FEATURE_LABELS[feature]} is a Pro feature`
    : resource
    ? "You've reached your Free tier limit"
    : "Upgrade to Pro";

  const body = capReached
    ? reason ?? "You've used your monthly credit limit. Ask your organization's admin to raise it."
    : managedByOrg
    ? isOrgAdmin
      ? "Your team's shared credit pool is empty. Buy credits to get everyone working again."
      : reason ?? "Your organization is out of credits. Ask your admin to top up."
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
            {/* A dry pool has no member-side remedy — only an admin buying
                credits fixes it, so the request CTA would be a dead end. */}
            {capReached && (
              <Button onClick={handleRequestCredits} className="mt-2">
                Ask for a higher limit
              </Button>
            )}
            {!capReached && isOrgAdmin && (
              <Button onClick={() => navigate("/organization")} className="mt-2">
                Buy credits
              </Button>
            )}
          </>
        ) : (
          <>
            <Button onClick={handleUpgradeClick} className="mt-2">
              Upgrade to Pro
            </Button>
            {packsAvailable && (
              <p className="text-xs text-muted-foreground">
                Or{" "}
                <button
                  type="button"
                  onClick={() => navigate("/profile")}
                  className="underline underline-offset-2"
                >
                  buy a credit pack
                </button>{" "}
                — packs never expire.
              </p>
            )}
          </>
        )}
      </div>
    </Card>
  );
};
