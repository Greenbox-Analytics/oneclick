import { supabase } from "@/integrations/supabase/client";

export const API_URL =
  import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

/**
 * Error class thrown by apiFetch on non-2xx responses.
 * Extends Error so existing `err.message` usage is unaffected.
 * New callers can branch on `err instanceof ApiError && err.status === 403`.
 */
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

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
 * Auto-sets `Content-Type: application/json` for string bodies (FormData
 * uploads must set their own multipart Content-Type, so are left alone).
 * Throws ApiError (with HTTP status) on non-2xx responses.
 */
export async function apiFetch<T>(
  url: string,
  opts?: RequestInit
): Promise<T> {
  const authHeaders = await getAuthHeaders();
  const isStringBody = typeof opts?.body === "string";
  const jsonHeader: Record<string, string> = isStringBody
    ? { "Content-Type": "application/json" }
    : {};
  const res = await fetch(url, {
    ...opts,
    headers: {
      ...authHeaders,
      ...jsonHeader,
      ...opts?.headers, // explicit caller headers win
    },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(body.detail || `Request failed: ${res.status}`, res.status);
  }
  return res.json();
}
