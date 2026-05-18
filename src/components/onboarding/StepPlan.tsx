import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Check, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

interface StepPlanProps {
  /** Free path — complete onboarding, navigate to dashboard. */
  onChooseFree: () => void | Promise<void>;
  /** Pro path — complete onboarding, redirect to Stripe Checkout. */
  onChoosePro: (plan: "monthly" | "annual") => void | Promise<void>;
  onBack: () => void;
}

const FREE_BULLETS = [
  "3 artists, 3 projects, 50 tasks",
  "5 split sheets per month",
  "OneClick royalty calculator (1/month)",
  "Google Drive integration",
];

const PRO_BULLETS = [
  "Unlimited artists, projects, tasks",
  "Unlimited split sheets",
  "Zoe AI contract analysis",
  "Rights Registry + all integrations",
];

/**
 * Onboarding step 4 of 5 — tier selection.
 *
 * Both paths save the user's onboarding profile to the DB FIRST (handled by
 * the parent's `onChooseFree` / `onChoosePro` callbacks), so a cancelled
 * Stripe Checkout doesn't leave the user stuck redoing onboarding. The Pro
 * path then redirects to Stripe; the Free path lands on /dashboard.
 *
 * Default-selected card is Free (lowest friction). Pro is clearly highlighted
 * but the primary action button on Free is what the user can hit with Enter.
 */
export default function StepPlan({ onChooseFree, onChoosePro, onBack }: StepPlanProps) {
  const [billingCycle, setBillingCycle] = useState<"monthly" | "annual">("monthly");
  const [busy, setBusy] = useState<null | "free" | "pro">(null);

  const handleFree = async () => {
    if (busy) return;
    setBusy("free");
    try {
      await onChooseFree();
    } finally {
      setBusy(null);
    }
  };

  const handlePro = async () => {
    if (busy) return;
    setBusy("pro");
    try {
      await onChoosePro(billingCycle);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="w-full flex flex-col items-center text-center">
      <h2 className="text-2xl font-semibold text-foreground mb-2">Pick your plan</h2>
      <p className="text-sm text-muted-foreground mb-6 max-w-sm">
        Start free. Upgrade anytime — billing is managed in your profile.
      </p>

      <div className="flex items-center gap-2 mb-6 text-xs">
        <button
          type="button"
          className={cn(
            "px-3 py-1 rounded-full border",
            billingCycle === "monthly"
              ? "border-primary bg-primary/10 text-foreground"
              : "border-border text-muted-foreground hover:text-foreground",
          )}
          onClick={() => setBillingCycle("monthly")}
        >
          Monthly
        </button>
        <button
          type="button"
          className={cn(
            "px-3 py-1 rounded-full border",
            billingCycle === "annual"
              ? "border-primary bg-primary/10 text-foreground"
              : "border-border text-muted-foreground hover:text-foreground",
          )}
          onClick={() => setBillingCycle("annual")}
        >
          Annual <span className="text-primary/80 ml-1">save ~20%</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full mb-6">
        <Card className="p-5 text-left border-2 border-border">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-base font-semibold">Free</h3>
            <Badge variant="outline" className="text-xs">Default</Badge>
          </div>
          <p className="text-xs text-muted-foreground mb-4">For trying things out</p>
          <ul className="space-y-1.5 text-sm">
            {FREE_BULLETS.map((b) => (
              <li key={b} className="flex items-start gap-2">
                <Check className="w-3.5 h-3.5 text-foreground/70 mt-0.5 shrink-0" />
                <span className="text-muted-foreground">{b}</span>
              </li>
            ))}
          </ul>
        </Card>

        <Card className="p-5 text-left border-2 border-primary/40 bg-primary/[0.03] relative">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-base font-semibold flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-primary" />
              Pro
            </h3>
            <Badge className="text-xs bg-primary/90">Recommended</Badge>
          </div>
          <p className="text-xs text-muted-foreground mb-4">For active creators</p>
          <ul className="space-y-1.5 text-sm">
            {PRO_BULLETS.map((b) => (
              <li key={b} className="flex items-start gap-2">
                <Check className="w-3.5 h-3.5 text-primary mt-0.5 shrink-0" />
                <span>{b}</span>
              </li>
            ))}
          </ul>
        </Card>
      </div>

      <div className="flex flex-col-reverse sm:flex-row items-center justify-center gap-3 w-full">
        <Button variant="ghost" onClick={onBack} disabled={busy !== null}>
          Back
        </Button>
        <Button
          variant="outline"
          onClick={handleFree}
          disabled={busy !== null}
          className="min-w-[160px]"
        >
          {busy === "free" ? "..." : "Continue with Free"}
        </Button>
        <Button onClick={handlePro} disabled={busy !== null} className="min-w-[160px]">
          {busy === "pro" ? "Redirecting..." : `Upgrade to Pro (${billingCycle})`}
        </Button>
      </div>
    </div>
  );
}
