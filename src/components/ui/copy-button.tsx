import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/** Copy `text`, and flag it briefly so a button can say "Copied". The chrome
 * is the caller's (an icon button, a pill, a code-block header). No-ops
 * silently on a non-secure origin — the text is on screen and selectable. */
export function useCopied(text: string): [boolean, () => void] {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* no clipboard on this origin */
    }
  };
  return [copied, copy];
}

/** Copy-to-clipboard icon button. */
export function CopyButton({ text, label, className }: { text: string; label: string; className?: string }) {
  const [copied, copy] = useCopied(text);
  return (
    <Button
      variant="ghost"
      size="sm"
      aria-label={label}
      className={cn("h-7 px-2 text-muted-foreground", className)}
      onClick={copy}
    >
      {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
    </Button>
  );
}
