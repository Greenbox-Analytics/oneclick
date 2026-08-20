// src/hooks/useBillingContext.ts
// Licensing Phase B (spec §5) — the "Working as: Personal / ⟨Org⟩" switcher.
// PUT /me/billing-context sets the caller's persistent billing context
// server-side (profiles.billing_context_org_id). Credits + entitlements are
// context-dependent, so both caches are invalidated on success. The endpoint
// 404s (identical body for "org doesn't exist" and "no active seat" — no
// existence oracle) when orgId isn't an org the caller holds an active seat
// in; switching to personal (orgId: null) always succeeds.
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { apiFetch, API_URL } from "@/lib/apiFetch";
import { useAuth } from "@/contexts/AuthContext";
import { useEntitlements, clearRememberedBillingContext } from "@/hooks/useEntitlements";
import type { BillingContextOption } from "@/hooks/useEntitlements";
import { orgContext } from "@/lib/credits";

export interface SetBillingContextArgs {
  /** Org id to switch into, or null to switch back to personal billing. */
  orgId: string | null;
}

export function useSetBillingContext() {
  const qc = useQueryClient();
  const { user } = useAuth();
  return useMutation<unknown, Error, SetBillingContextArgs>({
    mutationFn: async ({ orgId }) =>
      apiFetch(`${API_URL}/me/billing-context`, {
        method: "PUT",
        body: JSON.stringify({ orgId }),
      }),
    onSuccess: () => {
      // This switch was the user's own choice — forget the remembered context
      // so the payer-switch notice (useEntitlements) doesn't treat the
      // refetched context as a surprise.
      clearRememberedBillingContext(user?.id);
      qc.invalidateQueries({ queryKey: ["entitlements", user?.id] });
      qc.invalidateQueries({ queryKey: ["credit-usage", user?.id] });
    },
  });
}

/**
 * The switcher's shared brain: the options to offer, which one is current, and
 * a `switchTo` that reports the outcome. Both switchers (the Profile card and
 * the header pill) run through this so they can't drift on what a switch says —
 * in particular the PENDING case, where the write is accepted but billing does
 * NOT move until the org activates.
 */
export function useBillingContextSwitcher() {
  const { data: ent } = useEntitlements();
  const setBillingContext = useSetBillingContext();

  const available = ent?.availableContexts ?? [];
  const orgContexts = available.filter(
    (c): c is Extract<BillingContextOption, { type: "org" }> => c.type === "org",
  );
  // availableContexts is absent/personal-only unless LICENSING_ENABLED is on and
  // the caller holds a seat somewhere — so this is also the "show it at all" gate.
  const canSwitch = available.length > 1;

  const currentOrgId = orgContext(ent)?.orgId ?? null;
  const value = currentOrgId ?? "personal";
  const currentLabel = orgContexts.find((o) => o.orgId === currentOrgId)?.orgName ?? "Personal";

  const switchTo = (next: string) => {
    if (next === value) return;
    const orgId = next === "personal" ? null : next;
    const target = orgId ? orgContexts.find((o) => o.orgId === orgId) : null;
    setBillingContext.mutate(
      { orgId },
      {
        onSuccess: () => {
          if (!orgId) {
            toast.success("Switched to Personal", {
              description: "Your personal plan and credits apply again.",
            });
          } else if (target?.pending) {
            toast.success("Saved", {
              description: `${target.orgName} is still activating — billing switches over once it's active.`,
            });
          } else {
            toast.success("Switched", {
              description: `Now working as ${target?.orgName ?? "your organization"}.`,
            });
          }
        },
        onError: () =>
          toast.error("Couldn't switch", {
            description: "We couldn't update your billing context. Please try again.",
          }),
      },
    );
  };

  return {
    canSwitch,
    orgContexts,
    value,
    currentLabel,
    switchTo,
    isPending: setBillingContext.isPending,
  };
}
