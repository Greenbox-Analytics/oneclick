import { useState, type ReactNode } from "react";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { ApiError } from "@/lib/apiFetch";
import { PaywallModal } from "@/components/paywall/PaywallModal";
import type { GatedFeature, CountableResource } from "@/hooks/useEntitlements";

interface UseGatedActionOptions<TData, TVars> {
  /** The actual work: API call, supabase mutation, etc. */
  mutationFn: (vars: TVars) => Promise<TData>;
  /** Consumer's onSuccess (cache invalidation, navigation, toasts). Runs after the near-limit toast. */
  onSuccess?: (data: TData, vars: TVars) => void;
  /** Consumer's onError. Only invoked for non-402, non-storage-cap errors. 402 + storage-cap errors are swallowed and surface as the paywall modal instead. */
  onError?: (err: Error, vars: TVars) => void;

  resource?: CountableResource;
  feature?: GatedFeature;
  /** Current count (pre-action) — used for near-limit toast detection. */
  currentCount?: number;
  /** Cap (post-merge) — used for near-limit toast detection. -1 = unlimited. */
  cap?: number;
}

// Module-level: track which resources have already been toasted this session,
// so the near-limit toast doesn't re-fire on every page load.
const _toastedResources = new Set<string>();

interface UseGatedActionResult<TData, TVars> {
  mutate: (vars: TVars) => void;
  isPending: boolean;
  paywallElement: ReactNode;
}

export function useGatedAction<TData, TVars>(
  opts: UseGatedActionOptions<TData, TVars>,
): UseGatedActionResult<TData, TVars> {
  const [paywallOpen, setPaywallOpen] = useState(false);
  const [paywallReason, setPaywallReason] = useState<string | undefined>(undefined);

  const mutation = useMutation<TData, Error, TVars>({
    mutationFn: opts.mutationFn,
    onSuccess: (data, vars) => {
      // Near-limit toast: did currentCount + 1 cross 80%?
      if (opts.resource && opts.cap !== undefined && opts.cap !== -1 && opts.currentCount !== undefined) {
        const before = opts.currentCount;
        const after = before + 1;
        const cap = opts.cap;
        const beforePct = before / cap;
        const afterPct = after / cap;
        if (afterPct >= 0.8 && beforePct < 0.8 && !_toastedResources.has(opts.resource)) {
          _toastedResources.add(opts.resource);
          toast(`You're at ${after}/${cap} ${opts.resource}s — upgrade to Pro for unlimited.`);
        }
      }
      opts.onSuccess?.(data, vars);
    },
    onError: (err: Error, vars: TVars) => {
      // Best-effort: detect storage trigger errors that leak through as 400/409
      const looksLikeStorageCap =
        /storage cap exceeded/i.test(err.message) ||
        (err instanceof ApiError && (err.status === 400 || err.status === 409) &&
         /storage cap/i.test(err.message));

      if (err instanceof ApiError && err.status === 402) {
        setPaywallReason(err.message);
        setPaywallOpen(true);
        return; // swallow: paywall surfaces in the modal, not via consumer toast
      }
      if (looksLikeStorageCap) {
        setPaywallReason("Upload would exceed your storage cap. Upgrade to Pro for unlimited.");
        setPaywallOpen(true);
        return;
      }
      opts.onError?.(err, vars);
    },
  });

  const paywallElement = (
    <PaywallModal
      open={paywallOpen}
      onClose={() => setPaywallOpen(false)}
      reason={paywallReason}
      feature={opts.feature}
      resource={opts.resource}
    />
  );

  return {
    mutate: (vars: TVars) => mutation.mutate(vars),
    isPending: mutation.isPending,
    paywallElement,
  };
}
