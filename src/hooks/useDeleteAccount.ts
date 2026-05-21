import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { apiFetch, ApiError, API_URL } from "@/lib/apiFetch";
import { useAuth } from "@/contexts/AuthContext";

export type DeleteAccountErrorCode =
  | "last_admin"
  | "subscription_cancel_failed"
  | "email_mismatch"
  | "unknown";

export class DeleteAccountError extends Error {
  code: DeleteAccountErrorCode;
  constructor(code: DeleteAccountErrorCode, message: string) {
    super(message);
    this.code = code;
  }
}

export function useDeleteAccount() {
  const { signOut } = useAuth();
  const navigate = useNavigate();

  return useMutation({
    mutationFn: async ({ confirmationEmail }: { confirmationEmail: string }) => {
      try {
        await apiFetch<void>(`${API_URL}/users/account`, {
          method: "DELETE",
          body: JSON.stringify({ confirmation_email: confirmationEmail }),
        });
      } catch (err) {
        if (err instanceof ApiError) {
          const detail = err.message || `Request failed: ${err.status}`;
          if (err.status === 409) throw new DeleteAccountError("last_admin", detail);
          if (err.status === 400) throw new DeleteAccountError("email_mismatch", detail);
          if (err.status === 502) throw new DeleteAccountError("subscription_cancel_failed", detail);
          throw new DeleteAccountError("unknown", detail);
        }
        throw new DeleteAccountError("unknown", (err as Error)?.message || "Unknown error");
      }
    },
    onSuccess: async () => {
      // signOut hits the server to invalidate the session, which will 401
      // because the user no longer exists. Local state still clears.
      await signOut().catch(() => {});
      navigate("/", { replace: true });
    },
  });
}
