// src/components/docs/apiBits.tsx
// Small shared pieces of the docs' API section: the console and the page both
// render them, so they live outside either.
import type { ReactNode } from "react";

export function Tag({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <span
      className={`inline-flex h-[22px] items-center gap-1.5 whitespace-nowrap rounded-full border border-border bg-card px-2 text-[11.5px] font-semibold text-muted-foreground ${className}`}
    >
      {children}
    </span>
  );
}

const METHOD_STYLES = {
  GET: "bg-primary/10 text-primary",
  // The scheduled-payment tones are the one blue pair defined in both themes.
  POST: "bg-[hsl(var(--pay-sched-bg))] text-[hsl(var(--pay-sched-fg))]",
} as const;

export function MethodBadge({ method }: { method: keyof typeof METHOD_STYLES }) {
  return (
    <span
      className={`inline-flex h-[22px] items-center rounded-[5px] px-1.5 font-mono text-[11px] font-semibold tracking-wide ${METHOD_STYLES[method]}`}
    >
      {method}
    </span>
  );
}
