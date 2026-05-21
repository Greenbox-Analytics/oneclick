import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Lock } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { cn } from "@/lib/utils";
import type { GatedFeature, CountableResource } from "@/hooks/useEntitlements";
import { useAnalytics, type PaywallFeature } from "@/hooks/useAnalytics";

interface PaywallCardProps {
  feature?: GatedFeature;
  resource?: CountableResource;
  reason?: string;
  onUpgrade?: () => void;
  variant?: "page" | "inline" | "modal";
}

const FEATURE_LABELS: Record<GatedFeature, string> = {
  zoe: "Zoe AI",
  oneclick: "OneClick",
  registry: "Rights Registry",
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

  const title = feature
    ? `${FEATURE_LABELS[feature]} is a Pro feature`
    : resource
    ? "You've reached your Free tier limit"
    : "Upgrade to Pro";

  const body =
    reason ??
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
        <Button onClick={handleUpgradeClick} className="mt-2">
          Upgrade to Pro
        </Button>
      </div>
    </Card>
  );
};
