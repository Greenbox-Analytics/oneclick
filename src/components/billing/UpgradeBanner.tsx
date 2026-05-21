import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Sparkles, X } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { peekCachedAnalyticsContext } from "@/hooks/useAnalyticsContext";
import { cn } from "@/lib/utils";

const DISMISS_KEY = "msanii.upgrade_banner.dismissed";

/**
 * Subtle one-line strip near the top of the Dashboard for free-tier users.
 *
 * - Only shown when cached analytics context says `plan === "free"`.
 * - Dismissible — sets a localStorage flag (per-user via user_id suffix) so
 *   the banner stays gone until the user clears storage or upgrades.
 * - Returns null while the cache is empty (first sign-in of the session) —
 *   we'd rather not flash the banner before we know the plan than show it
 *   for a user who's actually on Pro.
 */
export function UpgradeBanner({ className }: { className?: string }) {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [dismissed, setDismissed] = useState(false);

  const dismissStorageKey = user ? `${DISMISS_KEY}.${user.id}` : null;

  useEffect(() => {
    if (!dismissStorageKey) return;
    setDismissed(localStorage.getItem(dismissStorageKey) === "1");
  }, [dismissStorageKey]);

  if (!user || dismissed) return null;

  const ctx = peekCachedAnalyticsContext(user.id);
  if (!ctx || ctx.plan !== "free") return null;

  const dismiss = () => {
    if (dismissStorageKey) localStorage.setItem(dismissStorageKey, "1");
    setDismissed(true);
  };

  return (
    <div
      className={cn(
        "flex items-center justify-between gap-3 px-4 py-2 mb-6 rounded-md",
        "bg-gradient-to-r from-primary/[0.08] to-primary/[0.02]",
        "border border-primary/15",
        className,
      )}
    >
      <div className="flex items-center gap-2 text-sm min-w-0">
        <Sparkles className="w-3.5 h-3.5 text-primary shrink-0" />
        <span className="text-muted-foreground truncate">
          You're on the Free plan.
        </span>
        <button
          type="button"
          onClick={() => navigate("/pricing")}
          className="text-primary hover:underline font-medium shrink-0"
        >
          See what Pro unlocks →
        </button>
      </div>
      <button
        type="button"
        onClick={dismiss}
        aria-label="Dismiss"
        className="text-muted-foreground/60 hover:text-foreground shrink-0"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}
