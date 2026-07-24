import { useState, useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Music } from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import { useAnalytics } from "@/hooks/useAnalytics";
import { useCreateCheckoutSession } from "@/hooks/useBilling";
import OnboardingProgress from "@/components/onboarding/OnboardingProgress";
import StepWelcome from "@/components/onboarding/StepWelcome";
import StepName from "@/components/onboarding/StepName";
import StepPreferences from "@/components/onboarding/StepPreferences";
import StepPlan from "@/components/onboarding/StepPlan";
import StepReady from "@/components/onboarding/StepReady";
import { markOnboardedCached } from "@/lib/onboardingCache";

const TOTAL_STEPS = 5;

// Map step index to display name for analytics
const STEP_NAMES = {
  0: "welcome",
  1: "name",
  2: "preferences",
  3: "plan",
  4: "ready",
} as const;

const Onboarding = () => {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { user } = useAuth();
  const { captureOnboardingStepCompleted, captureOnboardingFinished } = useAnalytics();
  const { mutateAsync: createCheckout } = useCreateCheckoutSession();
  const [currentStep, setCurrentStep] = useState(0);
  const [checkingStatus, setCheckingStatus] = useState(true);
  const [finishing, setFinishing] = useState(false);

  // Returning from a cancelled Stripe Checkout — show toast, jump back to plan step.
  // Profile was already saved before the redirect, so this is just resuming UI flow.
  useEffect(() => {
    if (searchParams.get("upgrade") === "cancelled") {
      toast.info("Checkout cancelled — you're on the Free plan. Upgrade anytime from Billing.");
      setCurrentStep(3);
      searchParams.delete("upgrade");
      setSearchParams(searchParams, { replace: true });
    }
  }, [searchParams, setSearchParams]);

  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    preferredName: "",
    role: "",
    company: "",
  });

  // Check if onboarding already completed and pre-populate profile data in one query
  useEffect(() => {
    const loadProfile = async () => {
      if (!user) {
        setCheckingStatus(false);
        return;
      }

      const { data } = await supabase
        .from("profiles")
        .select("first_name, last_name, given_name, full_name, company, onboarding_completed")
        .eq("id", user.id)
        .single();

      if (data?.onboarding_completed) {
        // Backfill the durable cache so a fully-onboarded user who lands here
        // (e.g., bookmark to /onboarding) gets their guard bypass restored.
        markOnboardedCached(user.id);
        navigate("/dashboard", { replace: true });
        return;
      }

      if (data) {
        let firstName = data.first_name || "";
        let lastName = data.last_name || "";
        if (!firstName && !lastName && data.full_name) {
          const parts = data.full_name.trim().split(/\s+/);
          firstName = parts[0] || "";
          lastName = parts.slice(1).join(" ") || "";
        }

        setFormData((prev) => ({
          ...prev,
          firstName: firstName || prev.firstName,
          lastName: lastName || prev.lastName,
          preferredName: data.given_name || prev.preferredName,
          company: data.company || prev.company,
        }));
      }

      setCheckingStatus(false);
    };

    loadProfile();
    // Depend on user.id (stable scalar), NOT user (object reference). When the
    // AuthContext hands down a new `user` object on a parent re-render or auth
    // event, depending on the object would re-fire this effect and re-issue
    // navigate("/dashboard"), producing rapid back-and-forth navigations that
    // Chromium throttles ("Throttling navigation to prevent the browser from
    // hanging"). Same fix as useOnboardingStatus.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id, navigate]);

  const handleUpdate = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleStepComplete = (stepIndex: number) => {
    const stepName = STEP_NAMES[stepIndex as keyof typeof STEP_NAMES];

    // Fire onboarding_step_completed for all steps
    captureOnboardingStepCompleted(stepName);

    // Advance to next step
    setCurrentStep(stepIndex + 1);
  };

  /** Persist the onboarding profile. Awaited so callers can sequence Stripe
   * checkout AFTER the row is committed — otherwise a cancelled checkout
   * could strand the user with `onboarding_completed = false`. */
  const persistProfile = async (): Promise<{ error: Error | null }> => {
    if (!user) return { error: new Error("Not authenticated") };
    const fullName = `${formData.firstName} ${formData.lastName}`.trim();
    const { error } = await supabase.from("profiles").upsert({
      id: user.id,
      first_name: formData.firstName,
      last_name: formData.lastName,
      given_name: formData.preferredName || null,
      full_name: fullName,
      role: formData.role || null,
      company: formData.company || null,
      onboarding_completed: true,
      updated_at: new Date().toISOString(),
    });
    return { error: error ? new Error(error.message) : null };
  };

  const handleFinish = async () => {
    if (!user) return;

    // Fire events for the final step (after the plan step — index 4)
    const stepName = STEP_NAMES[4];
    captureOnboardingStepCompleted(stepName);
    captureOnboardingFinished();

    // Persist BEFORE navigating so onboarding_completed=true is committed
    // before any post-onboarding route runs its useOnboardingStatus check.
    // Optimistic navigation here caused a race: the next route could observe
    // onboarding_completed=false, redirect to /onboarding, and ping-pong with
    // Onboarding.loadProfile (which then sees the committed `true` and
    // navigates back), tripping Chromium's navigation throttle.
    setFinishing(true);
    const { error } = await persistProfile();
    if (error) {
      setFinishing(false);
      toast.error("Couldn't save your profile — please retry");
      return;
    }
    // Set the durable bypass BEFORE navigating so the next route's
    // ProtectedRoute can trust localStorage instead of racing the supabase
    // query for onboarding_completed.
    markOnboardedCached(user.id);
    navigate("/dashboard", { replace: true, state: { fromOnboarding: true } });
  };

  /** Free path from the plan step — save profile + skip Stripe. */
  const handleChooseFree = async () => {
    captureOnboardingStepCompleted(STEP_NAMES[3]);
    // Save before advancing so the ready step's optimistic navigation isn't
    // the only chance the profile gets persisted.
    const { error } = await persistProfile();
    if (error) {
      toast.error("Couldn't save your profile — please retry");
      return;
    }
    if (user) markOnboardedCached(user.id);
    setCurrentStep(4);
  };

  /** Pro path — save profile, then redirect to Stripe Checkout.
   * Profile is saved FIRST so a cancelled checkout returns the user to a
   * fully-onboarded state (they just stay on the Free plan). */
  const handleChoosePro = async (plan: "monthly" | "annual") => {
    captureOnboardingStepCompleted(STEP_NAMES[3]);
    const { error } = await persistProfile();
    if (error) {
      toast.error("Couldn't save your profile — please retry");
      return;
    }
    if (user) markOnboardedCached(user.id);
    try {
      const url = await createCheckout({
        plan,
        cancel_path: "/onboarding?upgrade=cancelled",
        // success_path stays default → /subscription?stripe_session_id=...&welcome=true
      });
      window.location.href = url;
    } catch (e) {
      toast.error("Couldn't start checkout — please try again from Billing");
      console.error("createCheckout failed:", e);
    }
  };

  if (checkingStatus) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-background via-background to-secondary/20 p-4">
      {/* Top-left branding */}
      <div className="absolute top-6 left-6 flex items-center gap-2 opacity-60">
        <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
          <Music className="w-4 h-4 text-primary-foreground" />
        </div>
        <span className="text-sm font-semibold text-foreground">Msanii</span>
      </div>

      {/* Progress indicator — hidden on welcome step */}
      {currentStep > 0 && (
        <div className="absolute top-6">
          <OnboardingProgress currentStep={currentStep} totalSteps={TOTAL_STEPS} />
        </div>
      )}

      {/* Step content */}
      <div className="w-full max-w-lg flex items-center justify-center">
        {currentStep === 0 && (
          <StepWelcome onNext={() => handleStepComplete(0)} />
        )}
        {currentStep === 1 && (
          <StepName
            firstName={formData.firstName}
            lastName={formData.lastName}
            onUpdate={handleUpdate}
            onNext={() => handleStepComplete(1)}
            onBack={() => setCurrentStep(0)}
          />
        )}
        {currentStep === 2 && (
          <StepPreferences
            preferredName={formData.preferredName}
            role={formData.role}
            company={formData.company}
            onUpdate={handleUpdate}
            onNext={() => handleStepComplete(2)}
            onBack={() => setCurrentStep(1)}
          />
        )}
        {currentStep === 3 && (
          <StepPlan
            onChooseFree={handleChooseFree}
            onChoosePro={handleChoosePro}
            onBack={() => setCurrentStep(2)}
          />
        )}
        {currentStep === 4 && (
          <StepReady
            preferredName={formData.preferredName}
            firstName={formData.firstName}
            onFinish={handleFinish}
            isLoading={finishing}
          />
        )}
      </div>
    </div>
  );
};

export default Onboarding;
