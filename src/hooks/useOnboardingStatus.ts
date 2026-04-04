import { useEffect, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";

interface OnboardingStatus {
  onboardingCompleted: boolean;
  walkthroughCompleted: boolean;
  loading: boolean;
}

export const useOnboardingStatus = (): OnboardingStatus => {
  const { user } = useAuth();
  const [onboardingCompleted, setOnboardingCompleted] = useState(false);
  const [walkthroughCompleted, setWalkthroughCompleted] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkStatus = async () => {
      if (!user) {
        setLoading(false);
        return;
      }

      try {
        const { data, error } = await supabase
          .from("profiles")
          .select("onboarding_completed, walkthrough_completed")
          .eq("id", user.id)
          .single();

        if (error) {
          console.error("Error checking onboarding status:", error);
          setOnboardingCompleted(false);
          setWalkthroughCompleted(false);
        } else {
          setOnboardingCompleted(data?.onboarding_completed ?? false);
          setWalkthroughCompleted(data?.walkthrough_completed ?? false);
        }
      } catch (err) {
        console.error("Unexpected error checking onboarding:", err);
      } finally {
        setLoading(false);
      }
    };

    checkStatus();
  }, [user]);

  return { onboardingCompleted, walkthroughCompleted, loading };
};
