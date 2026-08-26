// src/components/billing/CreditRing.tsx
// Donut ring showing credits remaining, with a colored segment per tool.
// Ports the mockup's SVG ring (r=82, stroke 16, -90° start, 3px gaps).

export interface RingSegment {
  /** Credits spent for this segment (drives arc length as spent/grant). */
  value: number;
  /** CSS color (e.g. "var(--t-oneclick)"). */
  color: string;
}

interface CreditRingProps {
  /** Credits remaining, shown large in the center. */
  left: number;
  /** Monthly grant, denominator for the ring + "of N / mo" pill. */
  grant: number;
  /** Credits used this period. */
  used: number;
  segments: RingSegment[];
  resetLabel: string;
}

const R = 82;
const CIRC = 2 * Math.PI * R;
const GAP = 3;

export function CreditRing({ left, grant, used, segments, resetLabel }: CreditRingProps) {
  const usedPct = grant > 0 ? Math.round((used / grant) * 1000) / 10 : 0;

  // Lay out each segment as a dash arc, advancing the offset by its full fraction.
  let offset = 0;
  const arcs = segments
    .filter((s) => s.value > 0)
    .map((s, i) => {
      const frac = grant > 0 ? s.value / grant : 0;
      const len = Math.max(0, frac * CIRC - GAP);
      const dashoffset = -offset;
      offset += frac * CIRC;
      return (
        <circle
          key={i}
          cx={100}
          cy={100}
          r={R}
          stroke={s.color}
          strokeDasharray={`${len} ${CIRC - len}`}
          strokeDashoffset={dashoffset}
        />
      );
    });

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative w-[200px] h-[200px]">
        <svg width="200" height="200" viewBox="0 0 200 200" className="block -rotate-90">
          <circle cx={100} cy={100} r={R} fill="none" stroke="hsl(var(--muted))" strokeWidth={16} />
          <g fill="none" strokeWidth={16} strokeLinecap="round">
            {arcs}
          </g>
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
          <div className="text-[34px] font-bold tracking-tight tabular-nums leading-none">
            {left.toLocaleString()}
          </div>
          <div className="text-xs text-muted-foreground mt-1">credits left</div>
          <div className="mt-2 text-[11px] text-muted-foreground bg-muted/60 border border-border rounded-full px-2.5 py-0.5">
            of {grant.toLocaleString()} / mo
          </div>
        </div>
      </div>
      <div className="text-center">
        <div className="text-[13px]">
          <b className="font-semibold tabular-nums">{used.toLocaleString()}</b> used{" "}
          <span className="text-muted-foreground">· {usedPct}%</span>
        </div>
        <div className="text-xs text-muted-foreground/70 mt-0.5">Resets {resetLabel}</div>
      </div>
    </div>
  );
}
