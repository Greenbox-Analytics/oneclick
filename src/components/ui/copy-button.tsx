import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/** Copy-to-clipboard icon button. Silently no-ops when the clipboard is
 * unavailable (non-secure origin) — the text it copies is always on screen and
 * selectable, so there is nothing to recover. */
export function CopyButton({ text, label, className }: { text: string; label: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      variant="ghost"
      size="sm"
      aria-label={label}
      className={cn("h-7 px-2 text-muted-foreground", className)}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          /* no clipboard on this origin */
        }
      }}
    >
      {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
    </Button>
  );
}
