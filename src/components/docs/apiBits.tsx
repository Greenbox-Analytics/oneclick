// Small pieces of the docs' API section. Tag and MethodBadge are shared by the
// page and the console; ResponseExample is the page's alone.
import type { ReactNode } from "react";
import type { ResponseSection } from "./partnerApiSamples";

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

/** One captioned block per part of a response: caption and note left, JSON
 * fragment right, a field table beneath where the fields need explaining. */
export function ResponseExample({ label, sections }: { label: string; sections: ResponseSection[] }) {
  return (
    <div className="my-5 overflow-hidden rounded-xl border border-border">
      <div className="border-b border-border bg-muted/50 px-3.5 py-2 font-mono text-[11px] text-muted-foreground">{label}</div>
      {sections.map((s) => (
        <div key={s.key} className="grid gap-x-5 gap-y-2 border-b border-border px-3.5 py-3 last:border-b-0 md:grid-cols-[180px_minmax(0,1fr)]">
          <div>
            <div className="font-mono text-[12.5px] font-semibold text-foreground">{s.key}</div>
            <p className="mt-0.5 text-[12.5px] leading-relaxed text-muted-foreground">{s.note}</p>
          </div>
          <div className="min-w-0">
            <pre tabIndex={0} className="overflow-x-auto rounded-lg bg-muted/40 px-3 py-2.5 font-mono text-[12px] leading-relaxed text-foreground/90">{s.json}</pre>
            {s.fields && (
              <table className="mt-2 w-full border-collapse text-[12.5px]">
                <caption className="sr-only">{s.key} fields — name, type, description</caption>
                <tbody>
                  {s.fields.map(([field, type, desc]) => (
                    <tr key={field} className="border-t border-border">
                      <td className="py-1.5 pr-3 align-top font-mono text-[12px] text-foreground">{field}</td>
                      <td className="py-1.5 pr-3 align-top font-mono text-[11.5px] text-muted-foreground">{type}</td>
                      <td className="py-1.5 align-top text-muted-foreground">{desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
