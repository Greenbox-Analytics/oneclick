import { useCanUseFeature, type GatedFeature } from "@/hooks/useEntitlements";
import { PaywallCard } from "./PaywallCard";

interface RequireFeatureProps {
  feature: GatedFeature;
  children: React.ReactNode;
}

export const RequireFeature = ({ feature, children }: RequireFeatureProps) => {
  const { allowed, loading, error } = useCanUseFeature(feature);

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  // Fail OPEN on entitlement fetch error: a network blip should not paywall a
  // legitimate Pro user. Backend gates remain authoritative — if this user
  // isn't actually entitled, the per-action 402 from the server will still
  // surface the paywall. SP1's get_for_user_safe also returns degraded Free
  // defaults on backend error, but that only helps when the request reaches
  // the server; this branch covers the "didn't reach server" case.
  if (error) {
    return <>{children}</>;
  }

  if (!allowed) {
    return <PaywallCard feature={feature} variant="page" />;
  }

  return <>{children}</>;
};
