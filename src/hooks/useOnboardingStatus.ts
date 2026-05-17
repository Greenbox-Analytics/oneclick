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
    // Depend on user.id (stable scalar), NOT user (object reference) — the
    // AuthContext can re-render with a new user object on every parent render,
    // which would re-fire this effect forever and DoS the supabase REST endpoint
    // with thousands of identical queries (browser then errors with
    // ERR_INSUFFICIENT_RESOURCES). The id is what actually identifies the user.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id]);

  return { onboardingCompleted, walkthroughCompleted, loading };
};
