import { useState, useEffect, useMemo, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { HeaderDocsButton } from "@/components/layout/HeaderDocsButton";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2, Sun, Moon, HelpCircle, Sparkles } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { PageHeader } from "@/components/layout/PageHeader";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { DeleteAccountDialog } from "@/components/profile/DeleteAccountDialog";
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
import { TeamCardPanel } from "@/components/billing/TeamCardPanel";
import { PlanCard } from "@/components/billing/PlanCard";
import { CreditsUsageCard } from "@/components/billing/CreditsUsageCard";
import { ResourceLimitsCard } from "@/components/billing/ResourceLimitsCard";
import { IntegrationsCard } from "@/components/billing/IntegrationsCard";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useEntitlements, type BillingContextOption } from "@/hooks/useEntitlements";
import { useSetBillingContext } from "@/hooks/useBillingContext";
import { useAnalytics, type Plan } from "@/hooks/useAnalytics";
import { peekCachedAnalyticsContext, refreshAnalyticsContext } from "@/hooks/useAnalyticsContext";
import { useArtistsList } from "@/hooks/useArtistsList";
import { useProjectsList } from "@/hooks/useProjectsList";
import { useBoards } from "@/hooks/useBoards";
import { isPaidTier, tierLabel } from "@/lib/tiers";

const formatPeriodEnd = (iso: string): string =>
  new Date(iso).toLocaleDateString(undefined, { month: "long", day: "numeric", year: "numeric" });

const Profile = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [isDark, setIsDark] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("theme") === "dark" || document.documentElement.classList.contains("dark");
    }
    return false;
  });

  const toggleTheme = (dark: boolean) => {
    setIsDark(dark);
    if (dark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  };

  const [formData, setFormData] = useState({
    first_name: "",
    last_name: "",
    given_name: "",
    email: "",
    website: "",
    company: "",
    phone: "",
  });

  useEffect(() => {
    const fetchProfile = async () => {
      if (!user) return;
      try {
        const { data, error } = await supabase.from("profiles").select("*").eq("id", user.id).single();
        if (error) throw error;
        if (data) {
          let firstName = data.first_name || "";
          let lastName = data.last_name || "";
          if (!firstName && !lastName && data.full_name) {
            const parts = data.full_name.trim().split(/\s+/);
            firstName = parts[0] || "";
            lastName = parts.slice(1).join(" ") || "";
          }
          setFormData({
            first_name: firstName,
            last_name: lastName,
            given_name: data.given_name || "",
            email: user.email || "",
            website: data.website || "",
            company: data.company || "",
            phone: data.phone || "",
          });
        }
      } catch (error) {
        console.error("Error fetching profile:", error);
      }
    };
    fetchProfile();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id]);

  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(TOOL_CONFIGS.profile, {
    onComplete: () => markToolCompleted("profile"),
  });

  useEffect(() => {
    if (!onboardingLoading && !statuses.profile && walkthrough.phase === "idle") {
      const timer = setTimeout(() => walkthrough.startModal(), 500);
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onboardingLoading, statuses.profile]);

  const handleSave = async () => {
    if (!user) return;
    setIsLoading(true);
    try {
      const fullName = `${formData.first_name} ${formData.last_name}`.trim();
      const { error } = await supabase.from("profiles").upsert({
        id: user.id,
        first_name: formData.first_name,
        last_name: formData.last_name,
        given_name: formData.given_name,
        full_name: fullName,
        website: formData.website,
        company: formData.company,
        phone: formData.phone,
        updated_at: new Date().toISOString(),
      });
      if (error) throw error;
      toast({ title: "Success", description: "Profile updated successfully." });
    } catch (error: unknown) {
      console.error("Error updating profile:", error);
      const message = error instanceof Error ? error.message : undefined;
      toast({ title: "Error", description: message || "Failed to update profile.", variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  };

  const displayName = formData.given_name || formData.first_name || user?.email?.split("@")[0] || "";

  // Billing context switcher (Licensing Phase B, spec §5) — "Working as:
  // Personal / ⟨Org⟩". Only rendered when the caller has more than one
  // context to choose from (availableContexts is absent/personal-only for
  // everyone else, including licensing-off).
  const { data: ent } = useEntitlements();
  const setBillingContext = useSetBillingContext();
  const availableContexts = ent?.availableContexts ?? [];
  const orgContexts = availableContexts.filter(
    (c): c is Extract<BillingContextOption, { type: "org" }> => c.type === "org",
  );
  const showContextSwitcher = availableContexts.length > 1;
  // Key the switcher's current value off billingContext (present regardless
  // of CREDITS_ENABLED — Licensing follow-ups Task 3), falling back to
  // credits.managedByOrg for safety.
  const billingContextValue =
    (ent?.billingContext?.type === "org" ? ent.billingContext.orgId : undefined) ??
    ent?.credits?.managedByOrg?.orgId ??
    "personal";

  // ---- Ported from the retired /subscription page (merged into Profile) ----

  // Post-Checkout landing: Stripe's success URL returns here with
  // ?welcome=true&stripe_session_id=... — poll entitlements until the webhook
  // lands, fire checkout_completed once, then clean the URL.
  const [searchParams, setSearchParams] = useSearchParams();
  const stripeSessionId = searchParams.get("stripe_session_id");
  const welcome = searchParams.get("welcome") === "true";
  const [isPolling, setIsPolling] = useState(false);
  const queryClient = useQueryClient();
  const { captureCheckoutCompleted } = useAnalytics();
  const checkoutCompletedFiredRef = useRef(false);

  const isPaid = isPaidTier(ent?.tier);
  const analyticsCtx = user?.id ? peekCachedAnalyticsContext(user.id) : null;
  const isTester = analyticsCtx?.is_tester === true;
  const testerExpiresAt = analyticsCtx?.tester_expires_at ?? null;

  // Over-cap detection (Free users above their limits). These list hooks are
  // also mounted by ResourceLimitsCard below — React Query dedupes them.
  const { artists } = useArtistsList();
  const { projects } = useProjectsList();
  const { tasks } = useBoards();
  const isOverCap = useMemo(() => {
    if (!ent) return false;
    return (
      (ent.caps.maxArtists !== -1 && (artists?.length ?? 0) > ent.caps.maxArtists) ||
      (ent.caps.maxProjects !== -1 && (projects?.length ?? 0) > ent.caps.maxProjects) ||
      (ent.caps.maxTasks !== -1 && (tasks?.length ?? 0) > ent.caps.maxTasks) ||
      (ent.caps.maxStorageBytes !== -1 && ent.usage.totalStorageBytes > ent.caps.maxStorageBytes) ||
      (ent.caps.maxSplitSheetsPerMonth !== -1 &&
        ent.usage.splitSheetsThisPeriod > ent.caps.maxSplitSheetsPerMonth)
    );
  }, [ent, artists, projects, tasks]);

  useEffect(() => {
    if (!welcome || !stripeSessionId) return;
    if (isPaid) {
      if (!checkoutCompletedFiredRef.current) {
        checkoutCompletedFiredRef.current = true;
        const completedPlan = (ent?.subscription?.planPeriod as Plan | undefined) ?? "monthly";
        captureCheckoutCompleted(completedPlan);
      }
      // Refresh the analytics-context cache so banners reading the 5-min
      // localStorage cache don't keep showing "you're on Free" post-upgrade.
      if (user?.id) {
        void refreshAnalyticsContext(user.id, user.email);
      }
      setSearchParams({});
      toast({ title: `Welcome to ${tierLabel(ent?.tier)}!`, description: "Your subscription is active." });
      return;
    }
    setIsPolling(true);
    const interval = setInterval(() => {
      queryClient.invalidateQueries({ queryKey: ["entitlements"] });
    }, 1000);
    const timeout = setTimeout(() => {
      clearInterval(interval);
      setIsPolling(false);
      toast({
        title: "Subscription is processing",
        description: "Refresh in a moment if it doesn't show up.",
      });
    }, 10_000);
    return () => {
      clearInterval(interval);
      clearTimeout(timeout);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [welcome, stripeSessionId, isPaid]);

  const handleBillingContextChange = (value: string) => {
    const orgId = value === "personal" ? null : value;
    const target = orgId ? orgContexts.find((o) => o.orgId === orgId) : null;
    setBillingContext.mutate(
      { orgId },
      {
        onSuccess: () => {
          if (!orgId) {
            toast({ title: "Switched to Personal", description: "Your personal plan and credits apply again." });
          } else if (target?.pending) {
            toast({
              title: "Saved",
              description: `${target.orgName} is still activating — billing switches over once it's active.`,
            });
          } else {
            toast({ title: "Switched", description: `Now working as ${target?.orgName ?? "your organization"}.` });
          }
        },
        onError: () => {
          toast({
            title: "Couldn't switch",
            description: "We couldn't update your billing context. Please try again.",
            variant: "destructive",
          });
        },
      },
    );
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Post-Checkout polling overlay (returning from Stripe) */}
      {isPolling && (
        <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur flex items-center justify-center">
          <div className="text-center">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-3" />
            {/* Tier isn't known yet (webhook hasn't landed), so this can't
                name the plan without guessing. */}
            <div className="text-sm">Activating your subscription…</div>
          </div>
        </div>
      )}

      <PageHeader
        backTo="/dashboard"
        actions={
          <>
            <ToolHelpButton onClick={() => walkthrough.replay()} />
            <HeaderDocsButton />
          </>
        }
      />

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        <div className="mb-6">
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Account &amp; Billing</h1>
          <p className="text-muted-foreground mt-1">
            Your profile, plan, and credit usage — all in one place.
          </p>
        </div>

        {/* Over-cap banner (Free users above their limits) */}
        {isOverCap && (
          <div className="mb-6 rounded-lg border border-destructive/20 bg-destructive/5 p-4 flex items-center justify-between gap-4">
            <div className="flex-1">
              <div className="font-medium text-sm">You&apos;re over your Free tier limits</div>
              <div className="text-sm text-muted-foreground mt-1">
                Some create actions are blocked until you reduce your usage or upgrade.
              </div>
            </div>
            <Button onClick={() => navigate("/pricing")}>Upgrade</Button>
          </div>
        )}

        {/* Cancel-scheduled banner */}
        {ent?.subscription?.cancelAtPeriodEnd && ent.subscription.currentPeriodEnd && (
          <div className="mb-6 rounded-lg border border-amber-500/20 bg-amber-500/5 p-4">
            <div className="font-medium text-sm">Subscription scheduled to end</div>
            <div className="text-sm text-muted-foreground mt-1">
              Your {tierLabel(ent?.tier)} access ends on{" "}
              {new Date(ent.subscription.currentPeriodEnd).toLocaleDateString()}. Reactivate via Manage
              subscription if you change your mind.
            </div>
          </div>
        )}

        {/* Beta tester banner */}
        {isTester && (
          <div className="mb-6 rounded-lg border border-primary/30 bg-primary/5 p-4">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-primary" />
              <div className="font-medium text-sm">Beta tester access</div>
            </div>
            <div className="text-sm text-muted-foreground mt-1">
              You have full access to every paid feature as a beta tester
              {testerExpiresAt ? ` until ${formatPeriodEnd(testerExpiresAt)}` : " (no expiration set)"}.
              Thanks for helping us shape Msanii! For questions or feedback,{" "}
              <a href="mailto:tech@greenboxanalytics.ca" className="underline">
                contact us
              </a>
              .
            </div>
          </div>
        )}

        <div className="flex flex-col gap-[22px]">
          {/* Account information */}
          <Card className="p-6" data-walkthrough="profile-info">
            <h2 className="text-lg font-semibold tracking-tight">Account information</h2>
            <div className="text-[13.5px] text-muted-foreground mt-0.5">Update your personal details</div>
            <div className="mt-5 space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="first_name">First name</Label>
                  <Input
                    id="first_name"
                    value={formData.first_name}
                    onChange={(e) => setFormData({ ...formData, first_name: e.target.value })}
                    placeholder="John"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="last_name">Last name</Label>
                  <Input
                    id="last_name"
                    value={formData.last_name}
                    onChange={(e) => setFormData({ ...formData, last_name: e.target.value })}
                    placeholder="Doe"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-1.5">
                  <Label htmlFor="given_name">Preferred name</Label>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <HelpCircle className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>This is the name we&apos;ll use to address you throughout the app</p>
                    </TooltipContent>
                  </Tooltip>
                </div>
                <Input
                  id="given_name"
                  value={formData.given_name}
                  onChange={(e) => setFormData({ ...formData, given_name: e.target.value })}
                  placeholder="Johnny"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input id="email" type="email" value={formData.email} disabled className="bg-muted" />
                <p className="text-xs text-muted-foreground">Email cannot be changed</p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="website">Website</Label>
                  <Input
                    id="website"
                    type="url"
                    value={formData.website}
                    onChange={(e) => setFormData({ ...formData, website: e.target.value })}
                    placeholder="https://example.com"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="company">Company</Label>
                  <Input
                    id="company"
                    value={formData.company}
                    onChange={(e) => setFormData({ ...formData, company: e.target.value })}
                    placeholder="Independent Management"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="phone">Phone</Label>
                <Input
                  id="phone"
                  type="tel"
                  value={formData.phone}
                  onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                  placeholder="+1 (555) 123-4567"
                />
              </div>
              <div className="pt-2">
                <Button onClick={handleSave} disabled={isLoading}>
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    "Save changes"
                  )}
                </Button>
              </div>
            </div>
          </Card>

          {/* TeamCard + Appearance */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-[22px]">
            <div data-walkthrough="profile-teamcard">
              <TeamCardPanel name={displayName} email={formData.email} />
            </div>
            <Card className="p-6" data-walkthrough="profile-theme">
              <h2 className="text-lg font-semibold tracking-tight">Appearance</h2>
              <div className="text-[13.5px] text-muted-foreground mt-0.5">Customize the look and feel</div>
              <div className="flex items-center justify-between mt-4">
                <div className="flex items-center gap-3">
                  {isDark ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
                  <div>
                    <div className="text-sm font-semibold">Dark mode</div>
                    <p className="text-[12.5px] text-muted-foreground">
                      {isDark ? "Dark theme is active" : "Light theme is active"}
                    </p>
                  </div>
                </div>
                <Switch checked={isDark} onCheckedChange={toggleTheme} />
              </div>
            </Card>
          </div>

          {/* Working as — billing context switcher (Licensing Phase B, spec §5) */}
          {showContextSwitcher && (
            <Card className="p-5">
              <div className="flex items-center justify-between gap-4 flex-wrap">
                <div>
                  <div className="text-sm font-semibold">Working as</div>
                  <div className="text-[12.5px] text-muted-foreground mt-0.5">
                    Whose credits and billing apply to your account right now
                  </div>
                </div>
                <Select
                  value={billingContextValue}
                  onValueChange={handleBillingContextChange}
                  disabled={setBillingContext.isPending}
                >
                  <SelectTrigger className="w-[220px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="personal">Personal</SelectItem>
                    {orgContexts.map((o) => (
                      <SelectItem key={o.orgId} value={o.orgId}>
                        {o.orgName}
                        {o.pending ? " (activating soon)" : ""}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </Card>
          )}

          {/* Plan */}
          <PlanCard />

          {/* Credits & usage (renders nothing when the flag is off) */}
          <CreditsUsageCard />

          {/* Resource limits */}
          <ResourceLimitsCard />

          {/* Integrations */}
          <IntegrationsCard />

          {/* Danger zone */}
          <Card className="p-6 border-destructive/50">
            <h2 className="text-lg font-semibold tracking-tight text-destructive">Danger zone</h2>
            <div className="text-[13.5px] text-muted-foreground mt-0.5">
              Permanently delete your account and all associated data. This action cannot be undone —
              your projects, files, audio, and subscription will be removed immediately.
            </div>
            <div className="mt-[18px]">
              <Button variant="destructive" onClick={() => setDeleteDialogOpen(true)}>
                Delete account
              </Button>
            </div>
          </Card>
        </div>

        <DeleteAccountDialog
          open={deleteDialogOpen}
          onOpenChange={setDeleteDialogOpen}
          userEmail={user?.email ?? ""}
        />
      </main>

      <ToolIntroModal
        config={TOOL_CONFIGS.profile}
        isOpen={walkthrough.phase === "modal"}
        onStartTour={walkthrough.startSpotlight}
        onSkip={walkthrough.skip}
      />
      <WalkthroughProvider
        isActive={walkthrough.phase === "spotlight"}
        currentStep={walkthrough.currentStep}
        currentStepIndex={walkthrough.visibleStepIndex}
        totalSteps={walkthrough.totalSteps}
        onNext={walkthrough.next}
        onSkip={walkthrough.skip}
      />
    </div>
  );
};

export default Profile;
