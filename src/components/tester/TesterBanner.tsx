import { useEffect, useState } from "react";
import { Sparkles, X } from "lucide-react";
import { useTesterStatus } from "@/hooks/useTesterStatus";
import { Button } from "@/components/ui/button";

const DISMISS_KEY = "msanii.tester_banner.dismissed.v1";

function formatRelativeExpiry(iso: string | null): string | null {
  if (!iso) return null;
  const expires = new Date(iso).getTime();
  if (Number.isNaN(expires)) return null;
  const days = Math.round((expires - Date.now()) / (1000 * 60 * 60 * 24));
  if (days <= 0) return "today";
  if (days === 1) return "tomorrow";
  if (days < 30) return `in ${days} days`;
  const months = Math.round(days / 30);
  return `in ${months} month${months === 1 ? "" : "s"}`;
}

/**
 * Global banner shown to beta testers so they can confirm their elevated
 * access. Hidden when isTester=false. Dismissable for the session via X.
 */
export function TesterBanner() {
  const { isTester, expiresAt } = useTesterStatus();
  const [dismissed, setDismissed] = useState(() => {
    try {
      return sessionStorage.getItem(DISMISS_KEY) === "1";
    } catch {
      return false;
    }
  });

  // Re-evaluate dismissal when isTester flips (e.g. just-granted on first login)
  useEffect(() => {
    if (!isTester) setDismissed(false);
  }, [isTester]);

  if (!isTester || dismissed) return null;

  const expiryHint = formatRelativeExpiry(expiresAt);

  const handleDismiss = () => {
    try {
      sessionStorage.setItem(DISMISS_KEY, "1");
    } catch {
      /* ignore */
    }
    setDismissed(true);
  };

  return (
    <div className="w-full bg-primary/10 border-b border-primary/20 px-4 py-2 flex items-center justify-between text-sm">
      <div className="flex items-center gap-2 text-foreground">
        <Sparkles className="w-4 h-4 text-primary" />
        <span>
          Beta tester — you have full Pro access. Thanks for trying Msanii!
          {expiryHint && (
            <span className="text-muted-foreground ml-2">Access expires {expiryHint}.</span>
          )}
        </span>
      </div>
      <Button
        variant="ghost"
        size="icon"
        className="h-6 w-6"
        onClick={handleDismiss}
        aria-label="Dismiss tester banner"
      >
        <X className="w-4 h-4" />
      </Button>
    </div>
  );
}
