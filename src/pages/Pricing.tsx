import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Check, X, Music } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { useCreateCheckoutSession } from "@/hooks/useBilling";
import { useAnalytics } from "@/hooks/useAnalytics";

type Feature = { included: boolean; label: string };

const FREE_FEATURES: Feature[] = [
  { included: true, label: "3 artists" },
  { included: true, label: "3 projects" },
  { included: true, label: "50 tasks total" },
  { included: true, label: "1 GB storage" },
  { included: true, label: "5 split sheets per month" },
  { included: true, label: "OneClick royalty calculator (1 run/month)" },
  { included: true, label: "Google Drive integration" },
  { included: false, label: "Zoe AI contract analysis" },
  { included: false, label: "Rights Registry" },
  { included: false, label: "Slack · Notion · Monday integrations" },
];

const PRO_FEATURES: Feature[] = [
  { included: true, label: "Unlimited artists, projects, tasks" },
  { included: true, label: "Unlimited storage" },
  { included: true, label: "Unlimited split sheets" },
  { included: true, label: "Zoe AI contract analysis" },
  { included: true, label: "Unlimited OneClick royalty calculations" },
  { included: true, label: "Rights Registry" },
  { included: true, label: "All integrations: Drive, Slack, Notion, Monday" },
  { included: true, label: "Priority support" },
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

  const [plan, setPlan] = useState<"monthly" | "annual">("monthly");
  const { mutateAsync: createCheckout, isPending } = useCreateCheckoutSession();
  const { captureCheckoutStarted } = useAnalytics();

  const handleFreeClick = () => {
    navigate(user ? "/dashboard" : "/auth");
  };

  const handleProClick = async () => {
    if (!user) {
      navigate(`/auth?redirect=/pricing&plan=${plan}`);
      return;
    }
    try {
      const url = await createCheckout(plan);
      captureCheckoutStarted(plan);
      window.location.href = url;
    } catch (err) {
      toast.error("Couldn't start checkout. Try again or contact support.");
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div
              className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
              onClick={() => navigate("/")}
            >
              <div className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center">
                <Music className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="text-lg font-semibold tracking-tight">Msanii</span>
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
          Free for indie artists. Pro for everyone scaling beyond a roster of one.
        </p>
      </section>

      {/* Pricing cards */}
      <section className="container mx-auto px-4 pb-24">
        <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
          {/* Free */}
          <Card className="p-8 flex flex-col">
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-1">Free</h2>
              <p className="text-sm text-muted-foreground">
                For solo artists getting started
              </p>
            </div>
            <div className="mb-8">
              <span className="text-4xl font-semibold tracking-tight">$0</span>
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

          {/* Pro */}
          <Card className="p-8 flex flex-col border-primary relative">
            <Badge className="absolute -top-3 left-8">Most popular</Badge>
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-1">Pro</h2>
              <p className="text-sm text-muted-foreground">
                For managers, labels, and serious creators
              </p>
            </div>

            {/* Plan toggle */}
            <Tabs value={plan} onValueChange={(v) => setPlan(v as "monthly" | "annual")} className="mb-6">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="monthly">Monthly</TabsTrigger>
                <TabsTrigger value="annual">Annual</TabsTrigger>
              </TabsList>
            </Tabs>

            {/* Price */}
            {plan === "monthly" ? (
              <div className="mb-8">
                <span className="text-4xl font-semibold tracking-tight">$25</span>
                <span className="text-muted-foreground ml-1">/month</span>
              </div>
            ) : (
              <div className="mb-8">
                <span className="text-4xl font-semibold tracking-tight">$250</span>
                <span className="text-muted-foreground ml-1">/year</span>
                <div className="text-sm text-muted-foreground mt-1">≈ $20.83/month — save 2 months</div>
              </div>
            )}

            <ul className="space-y-3 flex-1 mb-8">
              {PRO_FEATURES.map((f) => (
                <FeatureItem key={f.label} {...f} />
              ))}
            </ul>
            <Button size="lg" className="w-full" onClick={handleProClick} disabled={isPending}>
              {isPending ? "Starting checkout…" : "Upgrade to Pro"}
            </Button>
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
