import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Check, X, Music, ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { useCreateCheckoutSession, type CheckoutPlan } from "@/hooks/useBilling";
import { useAnalytics } from "@/hooks/useAnalytics";
import { tierLabel, usd, annualPerMonth, ENTERPRISE_LABEL, TIER_PRICES, TEAM_STORAGE_OVERAGE_USD_PER_GB } from "@/lib/tiers";

type Feature = { included: boolean; label: string };
type Period = "monthly" | "annual";

// ---------------------------------------------------------------------------
// Credits model: every tool is open on every tier — AI actions draw from a
// monthly credit allowance instead of tier locks. Numbers mirror
// tier_entitlements (100/2,000/5,000 credits; 1/100/250 GB personal storage,
// a hard cap on every tier; teams 0/1/3 owned, 3/10 members, 100/250 GB team
// storage — matching personal, 20260817000001). Existing paid subscribers on the prior 3,000/8,000 grants keep
// them only via grandfather overrides (20260816000001) — this public page
// advertises the new grants, not those.
// ---------------------------------------------------------------------------
const FREE_FEATURES: Feature[] = [
  { included: true, label: "All tools included" },
  { included: true, label: "150 credits per month" },
  { included: true, label: "3 artists, 3 projects, 50 tasks" },
  { included: true, label: "1 GB personal storage" },
  { included: true, label: "5 split sheets per month" },
  { included: true, label: "Google Drive & Dropbox integration" },
  { included: true, label: "Join any team you're invited to" },
  { included: false, label: "Own a team (Basic and up)" },
];

// "basic" DB tier.
const BASIC_FEATURES: Feature[] = [
  { included: true, label: "All tools included" },
  { included: true, label: "2,000 credits per month" },
  { included: true, label: "Unlimited artists, projects, and tasks" },
  { included: true, label: "100 GB personal storage" },
  { included: true, label: "Unlimited split sheets" },
  { included: true, label: "Google Drive & Dropbox integration" },
  { included: true, label: "1 team with up to 3 members" },
  { included: true, label: "100 GB team storage" },
];

// "pro" DB tier.
const PRO_FEATURES: Feature[] = [
  { included: true, label: `Everything in ${tierLabel("basic")}` },
  { included: true, label: "5,000 credits per month" },
  { included: true, label: "250 GB personal storage" },
  { included: true, label: "3 teams with up to 5 members each — scale to 10 per team when another Pro member joins" },
  { included: true, label: `250 GB team storage, then ${usd(TEAM_STORAGE_OVERAGE_USD_PER_GB)}/GB per month` },
  { included: true, label: "Priority support" },
];

// No DB tier — enterprise orgs are set up by Msanii (POST /admin/orgs) after
// a conversation, so this card is a "Talk to us" contact CTA, not a signup.
const ENTERPRISE_FEATURES: Feature[] = [
  { included: true, label: `Everything in ${tierLabel("pro")}` },
  { included: true, label: "Teams larger than 10 people" },
  { included: true, label: "Shared credit pool with per-member limits" },
  { included: true, label: "Centralized billing for your whole organization" },
  { included: true, label: "Priority support and dedicated onboarding" },
  { included: true, label: "Private, custom deployment on request" },
];

const FeatureItem = ({ included, label }: Feature) => (
  <li className="flex items-start gap-3 text-sm">
    {included ? (
      <Check className="w-4 h-4 text-foreground mt-0.5 flex-shrink-0" />
    ) : (
      <X className="w-4 h-4 text-muted-foreground/40 mt-0.5 flex-shrink-0" />
    )}
    <span className={included ? "text-foreground" : "text-muted-foreground/60"}>
      {label}
    </span>
  </li>
);

/** Basic and Pro differ only in tier, blurb, features, and the "Most popular"
 * highlight — each card owns its own billing-period toggle and checkout call. */
const PaidPlanCard = ({
  tier,
  description,
  features,
  highlight,
}: {
  tier: "basic" | "pro";
  description: string;
  features: Feature[];
  highlight?: boolean;
}) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [period, setPeriod] = useState<Period>("monthly");
  const { mutateAsync: createCheckout, isPending } = useCreateCheckoutSession();
  const { captureCheckoutStarted } = useAnalytics();

  const handleClick = async () => {
    const planParam: CheckoutPlan = period === "annual" ? `${tier}_annual` : `${tier}_monthly`;
    if (!user) {
      navigate(`/auth?redirect=/pricing&plan=${planParam}`);
      return;
    }
    try {
      const url = await createCheckout(planParam);
      captureCheckoutStarted(period);
      window.location.href = url;
    } catch {
      toast.error("Couldn't start checkout. Try again or contact support.");
    }
  };

  return (
    <Card className={highlight ? "p-8 flex flex-col border-primary relative" : "p-8 flex flex-col relative"}>
      {highlight && <Badge className="absolute -top-3 left-8">Most popular</Badge>}
      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-1">{tierLabel(tier)}</h2>
        <p className="text-sm text-muted-foreground">
          {description}
        </p>
      </div>

      <Tabs value={period} onValueChange={(v) => setPeriod(v as Period)} className="mb-6">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="monthly">Monthly</TabsTrigger>
          <TabsTrigger value="annual">Annual</TabsTrigger>
        </TabsList>
      </Tabs>

      {period === "monthly" ? (
        <div className="mb-8">
          <span className="text-4xl font-semibold tracking-tight">{usd(TIER_PRICES[tier].monthly)}</span>
          <span className="text-muted-foreground ml-1">/month</span>
        </div>
      ) : (
        <div className="mb-8">
          <span className="text-4xl font-semibold tracking-tight">{usd(TIER_PRICES[tier].annual)}</span>
          <span className="text-muted-foreground ml-1">/year</span>
          <div className="text-sm text-muted-foreground mt-1">≈ {annualPerMonth(tier)}/month — save 2 months</div>
        </div>
      )}

      <ul className="space-y-3 flex-1 mb-8">
        {features.map((f) => (
          <FeatureItem key={f.label} {...f} />
        ))}
      </ul>
      <Button
        size="lg"
        variant={highlight ? "default" : "outline"}
        className="w-full"
        onClick={handleClick}
        disabled={isPending}
      >
        {isPending ? "Starting checkout…" : `Upgrade to ${tierLabel(tier)}`}
      </Button>
    </Card>
  );
};

const Pricing = () => {
  const navigate = useNavigate();
  const { user } = useAuth();

  const handleFreeClick = () => {
    navigate(user ? "/dashboard" : "/auth");
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 sm:gap-4">
              <Button
                variant="ghost"
                size="sm"
                className="gap-1.5 -ml-2 text-muted-foreground"
                onClick={() => navigate("/")}
              >
                <ArrowLeft className="w-4 h-4" />
                Back
              </Button>
              <div
                className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
                onClick={() => navigate("/")}
              >
                <div className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center">
                  <Music className="w-5 h-5 text-primary-foreground" />
                </div>
                <span className="text-lg font-semibold tracking-tight">Msanii</span>
              </div>
            </div>
            <Button
              variant="ghost"
              onClick={() => navigate(user ? "/dashboard" : "/auth")}
            >
              {user ? "Go to dashboard" : "Sign in"}
            </Button>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="container mx-auto px-4 py-16 text-center max-w-2xl">
        <h1 className="text-4xl md:text-5xl font-semibold tracking-tight mb-4">
          Simple pricing
        </h1>
        <p className="text-muted-foreground text-lg">
          Free for indie artists. Basic and Pro for growing catalogs. Enterprise for organizations running it all
          centrally.
        </p>
        {/* Grants were rescaled 2026-08-16 (150/3,000/8,000 -> 100/2,000/5,000
            on Free/Basic/Pro); this page always advertises the new numbers.
            Existing paid subscribers keep their old grant via
            `grandfathered_monthly_credits` until their current billing period
            ends — the "you keep your current credits until renewal" note lives
            on CreditsUsageCard, the surface that shows their actual grant. */}
      </section>

      {/* Pricing cards */}
      <section className="container mx-auto px-4 pb-24">
        <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-6 max-w-7xl mx-auto items-stretch">
          {/* Free */}
          <Card className="p-8 flex flex-col">
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-1">Free</h2>
              <p className="text-sm text-muted-foreground">
                For solo artists getting started
              </p>
            </div>
            <div className="mb-8">
              <span className="text-4xl font-semibold tracking-tight">{usd(TIER_PRICES.free.monthly)}</span>
              <span className="text-muted-foreground ml-1">/month</span>
            </div>
            <ul className="space-y-3 flex-1 mb-8">
              {FREE_FEATURES.map((f) => (
                <FeatureItem key={f.label} {...f} />
              ))}
            </ul>
            <Button
              variant="outline"
              size="lg"
              className="w-full"
              onClick={handleFreeClick}
            >
              {user ? "Continue with Free" : "Get started free"}
            </Button>
          </Card>

          {/* Basic */}
          <PaidPlanCard
            tier="basic"
            description="For independent managers and serious creators"
            features={BASIC_FEATURES}
            highlight
          />

          {/* Pro */}
          <PaidPlanCard
            tier="pro"
            description="For power users and small teams"
            features={PRO_FEATURES}
          />

          {/* Enterprise is never created in-app (owner decision, 2026-08-16) —
              this card is a pure "Talk to us" contact CTA, not a signup flow. */}
          <Card className="p-8 flex flex-col">
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-1">{ENTERPRISE_LABEL}</h2>
              <p className="text-sm text-muted-foreground">
                For organizations managing multiple artists and teams centrally
              </p>
            </div>
            <div className="mb-8">
              <span className="text-2xl font-semibold tracking-tight">Custom</span>
              <div className="text-sm text-muted-foreground mt-1">Shared credit pool — priced for your organization</div>
            </div>
            <ul className="space-y-3 flex-1 mb-8">
              {ENTERPRISE_FEATURES.map((f) => (
                <FeatureItem key={f.label} {...f} />
              ))}
            </ul>
            <p className="text-xs text-muted-foreground mb-3 text-center">
              Custom pricing, a monthly credit plan, and onboarding with our team.
            </p>
            <Button size="lg" variant="outline" className="w-full" onClick={() => navigate("/contact")}>
              Talk to us
            </Button>
          </Card>
        </div>

        <p className="mt-8 text-center text-sm text-muted-foreground">
          A OneClick run or an AI contract parse is 30 credits, a split sheet 20, and a Zoe
          message 5.
        </p>

        <p className="text-center text-xs text-muted-foreground mt-8">
          USD pricing. Cancel anytime. A team is covered by one admin&apos;s plan at a time — that plan supplies the team
          slot and team storage; other admins and members join free on any plan.
        </p>
      </section>
    </div>
  );
};

export default Pricing;
