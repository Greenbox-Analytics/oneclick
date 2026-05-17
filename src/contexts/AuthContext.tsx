import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { User, Session } from '@supabase/supabase-js';
import { supabase } from '@/integrations/supabase/client';
import { apiFetch, API_URL } from '@/lib/apiFetch';
import { identifyUser, resetUser } from '@/lib/posthog';

// Env-driven, never changes at runtime — hoist out of the component so it's not
// rebuilt as a new array reference per render (which would invalidate the
// auth-subscription useEffect deps and trigger an infinite re-subscribe loop).
const ADMIN_EMAILS = (import.meta.env.VITE_ADMIN_EMAILS || "")
  .split(",")
  .map((e: string) => e.trim().toLowerCase())
  .filter(Boolean);

interface AuthContextType {
  user: User | null;
  session: Session | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string, fullName?: string) => Promise<void>;
  signInWithGoogle: () => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);
      setLoading(false);
    });

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      setSession(session);
      setUser(session?.user ?? null);
      setLoading(false);

      if (event === 'SIGNED_IN' && session?.access_token) {
        // Fire-and-forget — backend is idempotent via welcome_email_sent_at.
        apiFetch(`${API_URL}/users/welcome`, { method: 'POST' }).catch((err) => {
          console.warn('Welcome email trigger failed:', err);
        });
      }

      // SP-Analytics: identify/reset PostHog distinct_id
      if (session?.user) {
        const isAdmin = ADMIN_EMAILS.includes((session.user.email || "").toLowerCase());
        identifyUser(session.user.id, {
          email: session.user.email,
          ...(isAdmin && { is_admin: true }),
        });
      } else if (event === "SIGNED_OUT") {
        resetUser();
      }
    });

    return () => subscription.unsubscribe();
    // Subscribe ONCE on mount — re-subscribing on every render causes
    // setUser/setSession to fire repeatedly with fresh object refs from
    // getSession(), which trips every consumer hook that depends on `user`.
  }, []);

  const signIn = async (email: string, password: string) => {
    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    if (error) throw error;
  };

  const signUp = async (email: string, password: string, fullName?: string) => {
    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: {
          full_name: fullName,
        },
        emailRedirectTo: `${window.location.origin}/onboarding`,
      },
    });
    if (error) throw error;
  };

  const signInWithGoogle = async () => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: `${window.location.origin}/dashboard`,
      },
    });
    if (error) throw error;
  };

  const signOut = async () => {
    const { error } = await supabase.auth.signOut();
    if (error) throw error;
  };

  // Memoize so consumers don't re-render when AuthProvider re-renders for
  // unrelated reasons. signIn/signUp/etc are stable closures over supabase.
  const value = useMemo(
    () => ({ user, session, loading, signIn, signUp, signInWithGoogle, signOut }),
    [user, session, loading],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
