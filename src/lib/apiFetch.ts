import { supabase } from "@/integrations/supabase/client";

export const API_URL =
  import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

/**
 * Get Authorization headers from the current Supabase session.
 * Use this for streaming/direct fetch calls that can't use apiFetch.
 */
export async function getAuthHeaders(): Promise<Record<string, string>> {
  const {
    data: { session },
  } = await supabase.auth.getSession();
  if (!session?.access_token) return {};
  return { Authorization: `Bearer ${session.access_token}` };
}

/**
 * Fetch wrapper that automatically includes Supabase auth headers.
 */
export async function apiFetch<T>(
  url: string,
  opts?: RequestInit
): Promise<T> {
  const authHeaders = await getAuthHeaders();
  const res = await fetch(url, {
    ...opts,
    headers: {
      ...authHeaders,
      ...opts?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}
