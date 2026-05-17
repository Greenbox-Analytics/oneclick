import { useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useIsAdmin } from "@/hooks/useAdmin";
import { identifyUser } from "@/lib/posthog";

/**
 * Tags PostHog with is_admin:true once /admin/me confirms the user is an
 * admin (env-managed OR DB-managed). Idempotent — PostHog identify is a
 * person-property merge, so calling it more than once is safe.
 *
 * Mount ONCE inside AuthProvider. No-op while loading or for non-admins.
 *
 * NOTE: deps key on user?.id rather than the full user object. Supabase
 * rebuilds the User object reference on every auth-state-change broadcast
 * (e.g., hourly token refresh), and we don't want to re-fire identify
 * unnecessarily — the id is the only field that matters here.
 */
export function useAdminPosthogTag() {
  const { user } = useAuth();
  const { isAdmin, loading } = useIsAdmin();

  useEffect(() => {
    if (loading || !user || !isAdmin) return;
    identifyUser(user.id, {
      email: user.email,
      is_admin: true,
    });
  }, [user?.id, user?.email, isAdmin, loading]);
}
