import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useOnboardingStatus } from "@/hooks/useOnboardingStatus";
import { isOnboardedCached } from "@/lib/onboardingCache";

interface ProtectedRouteProps {
  children: React.ReactNode;
  skipOnboardingCheck?: boolean;
}

export const ProtectedRoute = ({ children, skipOnboardingCheck = false }: ProtectedRouteProps) => {
  const { user, loading: authLoading } = useAuth();
  const { onboardingCompleted, loading: onboardingLoading } = useOnboardingStatus();
  const location = useLocation();

  // Optimistic: if we just came from onboarding, trust the navigation state
  // instead of waiting for a fresh Supabase query (avoids redirect loop).
  const fromOnboarding = (location.state as { fromOnboarding?: boolean } | null)?.fromOnboarding === true;

  // Durable bypass: once a user has finished onboarding on this browser,
  // localStorage remembers it. A transient `false` from a racing Supabase
  // query no longer triggers a redirect to /onboarding (which would then
  // ping-pong with Onboarding.loadProfile's redirect back to /dashboard and
  // hit the Chromium navigation throttle). See lib/onboardingCache.ts.
  const cachedOnboarded = isOnboardedCached(user?.id);

  if (authLoading || (!skipOnboardingCheck && !fromOnboarding && !cachedOnboarded && onboardingLoading)) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/auth" replace />;
  }

  if (
    !skipOnboardingCheck &&
    !fromOnboarding &&
    !cachedOnboarded &&
    !onboardingCompleted &&
    location.pathname !== "/onboarding"
  ) {
    return <Navigate to="/onboarding" replace />;
  }

  return <>{children}</>;
};
