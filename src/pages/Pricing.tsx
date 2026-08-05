import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Check, X, Music, ArrowLeft } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { useCreateCheckoutSession } from "@/hooks/useBilling";
import { useEntitlements } from "@/hooks/useEntitlements";
import { useAnalytics } from "@/hooks/useAnalytics";
import { tierLabel, usd, annualPerMonth, ENTERPRISE_LABEL, TIER_PRICES } from "@/lib/tiers";

type Feature = { included: boolean; label: string };
type Period = "monthly" | "annual";

// ---------------------------------------------------------------------------
// Legacy (credits off): tools are tier-gated, so Free shows locks.
// ---------------------------------------------------------------------------
const FREE_FEATURES: Feature[] = [
  { included: true, label: "3 artists" },
  { included: true, label: "3 projects" },
  { included: true, label: "50 tasks total" },
  { included: true, label: "1 GB storage" },
  { included: true, label: "5 split sheets per month" },
  { included: true, label: "OneClick royalty calculator (1 run/month)" },
  { included: true, label: "Google Drive integration" },
  { included: false, label: "Zoe AI contract analysis" },
  { included: false, label: "Metadata Registry" },
  { included: false, label: "Slack integration" },
];

// "basic" DB tier.
const BASIC_FEATURES: Feature[] = [
  { included: true, label: "Unlimited artists, projects, tasks" },
  { included: true, label: "Unlimited storage" },
  { included: true, label: "Unlimited split sheets" },
  { included: true, label: "Zoe AI contract analysis" },
  { included: true, label: "Unlimited OneClick royalty calculations" },
  { included: true, label: "Metadata Registry" },
  { included: true, label: "All integrations: Drive, Slack" },
];

// "pro" DB tier.
const PRO_FEATURES: Feature[] = [
  { included: true, label: `Everything in ${tierLabel("basic")}` },
  { included: true, label: "More storage headroom" },
  { included: true, label: "Priority support" },
];

// ---------------------------------------------------------------------------
// Credits model: every tool is open on every tier — AI actions draw from a
// monthly credit allowance instead of tier locks. Credit and storage numbers
// mirror tier_entitlements (150/3,000/8,000 credits; 1/100/250 GB — storage is
// a hard cap on every tier, never pay-per-use).
// ---------------------------------------------------------------------------
const FREE_FEATURES_CREDITS: Feature[] = [
  { included: true, label: "All tools included — Zoe, OneClick, Registry, split sheets" },
  { included: true, label: "150 credits per month for AI-powered actions" },
  { included: true, label: "3 artists, 3 projects, 50 tasks" },
  { included: true, label: "1 GB storage" },
  { included: true, label: "5 split sheets per month" },
  { included: true, label: "Google Drive integration" },
  { included: false, label: "Slack integration" },
];

const BASIC_FEATURES_CREDITS: Feature[] = [
  { included: true, label: "All tools included — usage draws from your monthly credits" },
  { included: true, label: "3,000 credits per month for AI-powered actions" },
  { included: true, label: "Unlimited artists, projects, and tasks" },
  { included: true, label: "100 GB storage" },
  { included: true, label: "Unlimited split sheets" },
  { included: true, label: "All integrations: Drive, Slack" },
];

const PRO_FEATURES_CREDITS: Feature[] = [
  { included: true, label: `Everything in ${tierLabel("basic")}` },
  { included: true, label: "8,000 credits per month for AI-powered actions" },
  { included: true, label: "250 GB storage" },
  { included: true, label: "Priority support" },
];

// No DB tier — org seats resolve to these entitlements (Phase B).
const ENTERPRISE_FEATURES: Feature[] = [
  { included: true, label: `Everything in ${tierLabel("pro")}` },
  { included: true, label: "Centralized billing for your whole team" },
  { included: true, label: "Email-invited member seats" },
  { included: true, label: "Org credit pool with per-seat allocation" },
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

const Pricing = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  // LICENSING_ENABLED is a BACKEND env var — Vite only exposes VITE_-prefixed
  // vars to the browser, so the frontend can't read it and must not try. The
  // signal it ships instead is `availableContexts`, which /me/entitlements
  // includes only when licensing is on. Absent for signed-out visitors too,
  // which is the behaviour we want: they get the contact-us CTA.
  const { data: ent } = useEntitlements();
  const licensingOn = ent?.availableContexts != null;
  // CREDITS_ENABLED is likewise backend-only: `credits` is non-null on
  // /me/entitlements exactly when the flag is on. Signed-out visitors (no
  // entitlements) see the legacy lists — same tradeoff as licensingOn above.
  const creditsOn = ent?.credits != null;
  const freeFeatures = creditsOn ? FREE_FEATURES_CREDITS : FREE_FEATURES;
  const basicFeatures = creditsOn ? BASIC_FEATURES_CREDITS : BASIC_FEATURES;
  const proFeatures = creditsOn ? PRO_FEATURES_CREDITS : PRO_FEATURES;

  const [basicPeriod, setBasicPeriod] = useState<Period>("monthly");
  const [proPeriod, setProPeriod] = useState<Period>("monthly");
  const { mutateAsync: createBasicCheckout, isPending: isBasicPending } = useCreateCheckoutSession();
  const { mutateAsync: createProCheckout, isPending: isProPending } = useCreateCheckoutSession();
  const { captureCheckoutStarted } = useAnalytics();

  const handleFreeClick = () => {
    navigate(user ? "/dashboard" : "/auth");
  };

  const handleBasicClick = async () => {
    if (!user) {
      navigate(`/auth?redirect=/pricing&plan=${basicPeriod}`);
      return;
    }
    try {
      const url = await createBasicCheckout(basicPeriod);
      captureCheckoutStarted(basicPeriod);
      window.location.href = url;
    } catch {
      toast.error("Couldn't start checkout. Try again or contact support.");
    }
  };

  const handleProClick = async () => {
    const planParam = proPeriod === "annual" ? "pro_annual" : "pro_monthly";
    if (!user) {
      navigate(`/auth?redirect=/pricing&plan=${planParam}`);
      return;
    }
    try {
      const url = await createProCheckout(planParam);
      captureCheckoutStarted(proPeriod);
      window.location.href = url;
    } catch {
      toast.error("Couldn't start checkout. Try again or contact support.");
    }
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
              {freeFeatures.map((f) => (
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
          <Card className="p-8 flex flex-col border-primary relative">
            <Badge className="absolute -top-3 left-8">Most popular</Badge>
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-1">{tierLabel("basic")}</h2>
              <p className="text-sm text-muted-foreground">
                For managers, labels, and serious creators
              </p>
            </div>

            <Tabs value={basicPeriod} onValueChange={(v) => setBasicPeriod(v as Period)} className="mb-6">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="monthly">Monthly</TabsTrigger>
                <TabsTrigger value="annual">Annual</TabsTrigger>
              </TabsList>
            </Tabs>

            {basicPeriod === "monthly" ? (
              <div className="mb-8">
                <span className="text-4xl font-semibold tracking-tight">{usd(TIER_PRICES.basic.monthly)}</span>
                <span className="text-muted-foreground ml-1">/month</span>
              </div>
            ) : (
              <div className="mb-8">
                <span className="text-4xl font-semibold tracking-tight">{usd(TIER_PRICES.basic.annual)}</span>
                <span className="text-muted-foreground ml-1">/year</span>
                <div className="text-sm text-muted-foreground mt-1">≈ {annualPerMonth("basic")}/month — save 2 months</div>
              </div>
            )}

            <ul className="space-y-3 flex-1 mb-8">
              {basicFeatures.map((f) => (
                <FeatureItem key={f.label} {...f} />
              ))}
            </ul>
            <Button size="lg" className="w-full" onClick={handleBasicClick} disabled={isBasicPending}>
              {isBasicPending ? "Starting checkout…" : `Upgrade to ${tierLabel("basic")}`}
            </Button>
          </Card>

          {/* Pro */}
          <Card className="p-8 flex flex-col relative">
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-1">{tierLabel("pro")}</h2>
              <p className="text-sm text-muted-foreground">
                For power users who lean hardest on Zoe, OneClick, and Registry
              </p>
            </div>

            <Tabs value={proPeriod} onValueChange={(v) => setProPeriod(v as Period)} className="mb-6">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="monthly">Monthly</TabsTrigger>
                <TabsTrigger value="annual">Annual</TabsTrigger>
              </TabsList>
            </Tabs>

            {proPeriod === "monthly" ? (
              <div className="mb-8">
                <span className="text-4xl font-semibold tracking-tight">{usd(TIER_PRICES.pro.monthly)}</span>
                <span className="text-muted-foreground ml-1">/month</span>
              </div>
            ) : (
              <div className="mb-8">
                <span className="text-4xl font-semibold tracking-tight">{usd(TIER_PRICES.pro.annual)}</span>
                <span className="text-muted-foreground ml-1">/year</span>
                <div className="text-sm text-muted-foreground mt-1">≈ {annualPerMonth("pro")}/month — save 2 months</div>
              </div>
            )}

            <ul className="space-y-3 flex-1 mb-8">
              {proFeatures.map((f) => (
                <FeatureItem key={f.label} {...f} />
              ))}
            </ul>
            <Button
              size="lg"
              variant="outline"
              className="w-full"
              onClick={handleProClick}
              disabled={isProPending}
            >
              {isProPending ? "Starting checkout…" : `Upgrade to ${tierLabel("pro")}`}
            </Button>
          </Card>

          {/* Enterprise — no DB tier; org seats resolve to these entitlements
              in Phase B. Non-functional CTA for now. */}
          <Card className="p-8 flex flex-col">
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-1">{ENTERPRISE_LABEL}</h2>
              <p className="text-sm text-muted-foreground">
                For organizations managing multiple artists and teams centrally
              </p>
            </div>
            <div className="mb-8">
              <span className="text-2xl font-semibold tracking-tight">Custom</span>
              <div className="text-sm text-muted-foreground mt-1">Org credit pool — priced to fit your team</div>
            </div>
            <ul className="space-y-3 flex-1 mb-8">
              {ENTERPRISE_FEATURES.map((f) => (
                <FeatureItem key={f.label} {...f} />
              ))}
            </ul>
            {/* Signed-out visitors can't load entitlements, so licensingOn reads false
                for them — send them to sign in rather than telling them it's unreleased. */}
            {licensingOn || !user ? (
              <>
                <Button
                  size="lg"
                  variant="outline"
                  className="w-full"
                  onClick={() => navigate(user ? "/organization" : "/auth?redirect=/pricing")}
                >
                  {user ? "Create an organization" : "Sign in to create an organization"}
                </Button>
                <p className="text-xs text-muted-foreground mt-3 text-center">
                  Starts free and inactive until it&apos;s funded.{" "}
                  <Link to="/contact" className="underline">
                    Talk to us
                  </Link>{" "}
                  about a monthly plan.
                </p>
              </>
            ) : (
              <>
                <Button size="lg" variant="outline" className="w-full" disabled title="Coming soon">
                  Create an organization
                </Button>
                <p className="text-xs text-muted-foreground mt-3 text-center">
                  Coming soon —{" "}
                  <Link to="/contact" className="underline">
                    reach out
                  </Link>{" "}
                  for early access.
                </p>
              </>
            )}
          </Card>
        </div>

        <p className="text-center text-xs text-muted-foreground mt-8">
          USD pricing. Cancel anytime.
        </p>
      </section>
    </div>
  );
};

export default Pricing;
