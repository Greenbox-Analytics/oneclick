import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Card-stack primitives for turning a wide table into a readable, single-column
 * list of cards on mobile — the platform-wide convention for dense tables.
 *
 * Pattern (see `CalculationResults` for the original it's modeled on):
 *
 *   <div className="hidden md:block"><table>…the real table…</table></div>
 *   <div className="md:hidden">
 *     <ResponsiveCardList>
 *       {rows.map((r) => (
 *         <DataCard key={r.id}>
 *           <DataCardHeader title={r.name} subtitle={r.role} trailing={<Badge/>} />
 *           <DataCardGrid>
 *             <DataCardField label="Gross" value={fmt(r.gross)} />
 *             <DataCardField label="Owed" value={fmt(r.owed)} emphasized />
 *           </DataCardGrid>
 *         </DataCard>
 *       ))}
 *     </ResponsiveCardList>
 *   </div>
 *
 * Keep these mobile-only (`md:hidden` on the wrapper) so the desktop table is
 * never affected.
 */

/** Vertical stack of cards. Use inside a `md:hidden` wrapper. */
export function ResponsiveCardList({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("space-y-3", className)} {...props} />;
}

/** A single bordered card representing one table row. */
export function DataCard({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "space-y-2 rounded-lg border border-border bg-card p-3",
        className,
      )}
      {...props}
    />
  );
}

interface DataCardHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  /** Rendered flush-right, e.g. a status badge or amount. */
  trailing?: React.ReactNode;
}

/** Card title row: primary label + optional subtitle, with a trailing slot. */
export function DataCardHeader({
  title,
  subtitle,
  trailing,
  className,
  ...props
}: DataCardHeaderProps) {
  return (
    <div
      className={cn("flex items-start justify-between gap-2", className)}
      {...props}
    >
      <div className="min-w-0">
        <p className="truncate text-sm font-semibold text-foreground">{title}</p>
        {subtitle != null && (
          <p className="truncate text-xs text-muted-foreground">{subtitle}</p>
        )}
      </div>
      {trailing != null && <div className="shrink-0">{trailing}</div>}
    </div>
  );
}

/** Two-column grid of label/value fields inside a card. */
export function DataCardGrid({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("grid grid-cols-2 gap-2 text-xs", className)}
      {...props}
    />
  );
}

interface DataCardFieldProps extends React.HTMLAttributes<HTMLDivElement> {
  label: React.ReactNode;
  value: React.ReactNode;
  /** Render the value bolder/darker (e.g. for the "amount owed" column). */
  emphasized?: boolean;
}

/** A single label-over-value pair. */
export function DataCardField({
  label,
  value,
  emphasized,
  className,
  ...props
}: DataCardFieldProps) {
  return (
    <div className={cn("min-w-0", className)} {...props}>
      <p className="text-muted-foreground">{label}</p>
      <p
        className={cn(
          "truncate font-medium",
          emphasized && "font-semibold text-foreground",
        )}
      >
        {value}
      </p>
    </div>
  );
}
