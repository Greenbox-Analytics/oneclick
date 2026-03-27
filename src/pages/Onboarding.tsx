import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Music } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import OnboardingProgress from "@/components/onboarding/OnboardingProgress";
import StepWelcome from "@/components/onboarding/StepWelcome";
import StepName from "@/components/onboarding/StepName";
import StepPreferences from "@/components/onboarding/StepPreferences";
import StepReady from "@/components/onboarding/StepReady";

const TOTAL_STEPS = 4;

const Onboarding = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { toast } = useToast();
  const [currentStep, setCurrentStep] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    preferredName: "",
    industry: "",
    company: "",
  });

  // Pre-populate from existing profile data
  useEffect(() => {
    const loadExistingProfile = async () => {
      if (!user) return;

      const { data } = await supabase
        .from("profiles")
        .select("first_name, last_name, given_name, full_name, company")
        .eq("id", user.id)
        .single();

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
    };

    loadExistingProfile();
  }, [user]);

  // Redirect if onboarding already completed
  useEffect(() => {
    const checkIfCompleted = async () => {
      if (!user) return;
      const { data } = await supabase
        .from("profiles")
        .select("onboarding_completed")
        .eq("id", user.id)
        .single();

      if (data?.onboarding_completed) {
        navigate("/dashboard", { replace: true });
      }
    };
    checkIfCompleted();
  }, [user, navigate]);

  const handleUpdate = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleFinish = async () => {
    if (!user) return;
    setIsLoading(true);

    try {
      const fullName = `${formData.firstName} ${formData.lastName}`.trim();
      const { error } = await supabase.from("profiles").upsert({
        id: user.id,
        first_name: formData.firstName,
        last_name: formData.lastName,
        given_name: formData.preferredName || null,
        full_name: fullName,
        industry: formData.industry || null,
        company: formData.company || null,
        onboarding_completed: true,
        updated_at: new Date().toISOString(),
      });

      if (error) throw error;

      navigate("/dashboard", { replace: true });
    } catch (error: any) {
      console.error("Error saving onboarding profile:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to save profile. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

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
          <StepWelcome onNext={() => setCurrentStep(1)} />
        )}
        {currentStep === 1 && (
          <StepName
            firstName={formData.firstName}
            lastName={formData.lastName}
            onUpdate={handleUpdate}
            onNext={() => setCurrentStep(2)}
            onBack={() => setCurrentStep(0)}
          />
        )}
        {currentStep === 2 && (
          <StepPreferences
            preferredName={formData.preferredName}
            industry={formData.industry}
            company={formData.company}
            onUpdate={handleUpdate}
            onNext={() => setCurrentStep(3)}
            onBack={() => setCurrentStep(1)}
          />
        )}
        {currentStep === 3 && (
          <StepReady
            preferredName={formData.preferredName}
            firstName={formData.firstName}
            onFinish={handleFinish}
            isLoading={isLoading}
          />
        )}
      </div>
    </div>
  );
};

export default Onboarding;
