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
  /** Raw `detail` from the error body — a string for legacy errors, or a
   * structured object (e.g. credit-wall 402s carry {reason, price, ...}). */
  detail?: unknown;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Build an ApiError from a parsed error body and HTTP status. `detail` is a
 * plain string for legacy errors; structured 402s carry an object with a
 * human-readable `reason` (and, for org-seat walls, managedByOrg/requestUrl/…).
 * The raw `detail` is attached to the returned error so callers can branch on
 * the structured shape — never render the raw object. Message precedence:
 * string detail → detail.reason → `fallback`.
 */
export function apiErrorFromBody(
  body: { detail?: unknown } | null | undefined,
  status: number,
  fallback = `Request failed: ${status}`
): ApiError {
  const detail = body?.detail;
  const message =
    typeof detail === "string"
      ? detail
      : (detail as { reason?: string } | undefined)?.reason ?? fallback;
  const err = new ApiError(message, status);
  err.detail = detail;
  return err;
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
    throw apiErrorFromBody(body, res.status);
  }
  // 204 No Content (and 205 Reset Content) MUST NOT have a body — calling
  // res.json() on these throws SyntaxError, which surfaces as a false-failure
  // toast even though the underlying request succeeded. Hand back undefined
  // so DELETE-style mutations can resolve cleanly.
  if (res.status === 204 || res.status === 205) {
    return undefined as T;
  }
  return res.json();
}
