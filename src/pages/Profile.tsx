import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2, Sun, Moon, HelpCircle, BookOpen } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useNavigate } from "react-router-dom";
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

  return (
    <div className="min-h-screen bg-background">
      <PageHeader
        backTo="/dashboard"
        actions={
          <>
            <ToolHelpButton onClick={() => walkthrough.replay()} />
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/docs")}
              title="Documentation"
              className="text-muted-foreground hover:text-foreground"
            >
              <BookOpen className="w-4 h-4" />
            </Button>
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
