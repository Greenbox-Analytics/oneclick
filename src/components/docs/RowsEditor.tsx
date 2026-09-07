// src/components/docs/RowsEditor.tsx
// Rows-of-fields editor for the console's list inputs (contributors, parties,
// works, shares, expenses), plus RequestPreview, the read-only pretty-printed
// JSON panel showing the exact body those rows produce. Every cell is a
// string in state; the console turns rows into the JSON the API takes
// (numbers cast by column kind), so nobody types JSON and "malformed JSON"
// cannot happen. validateRows / rowsToObjects are pure and unit-tested;
// errors render inline under the cell.
import { useId } from "react";
import { Plus, X } from "lucide-react";
import { CopyButton } from "@/components/ui/copy-button";

export type Row = Record<string, string>;

export interface Column {
  key: string;
  label: string;
  kind: "text" | "number" | "select";
  /** Fixed options, or a function for options that depend on other rows (a share's party). */
  options?: readonly string[] | (() => string[]);
  placeholder?: string;
  required?: boolean;
  min?: number;
  max?: number;
  /** Tailwind width class for the cell; defaults to flex-1. */
  width?: string;
}

export interface RowError {
  row: number;
  key: string;
  message: string;
}

const optionsOf = (c: Column): string[] => (typeof c.options === "function" ? c.options() : [...(c.options ?? [])]);

export function validateRows(columns: Column[], rows: Row[]): RowError[] {
  const errors: RowError[] = [];
  rows.forEach((row, i) => {
    for (const c of columns) {
      const v = (row[c.key] ?? "").trim();
      if (!v) {
        if (c.required) errors.push({ row: i, key: c.key, message: `${c.label} is required` });
        continue;
      }
      if (c.kind === "number") {
        const n = Number(v);
        if (!Number.isFinite(n)) errors.push({ row: i, key: c.key, message: `${c.label} must be a number` });
        else if ((c.min != null && n < c.min) || (c.max != null && n > c.max))
          errors.push({ row: i, key: c.key, message: `${c.label} must be between ${c.min ?? "…"} and ${c.max ?? "…"}` });
      } else if (c.kind === "select" && !optionsOf(c).includes(v)) {
        errors.push({ row: i, key: c.key, message: `${c.label} must be one of the choices` });
      }
    }
  });
  return errors;
}

/** rows -> API objects: numbers cast, empty optional cells dropped. */
export function rowsToObjects(columns: Column[], rows: Row[]): Record<string, string | number>[] {
  return rows.map((row) => {
    const out: Record<string, string | number> = {};
    for (const c of columns) {
      const v = (row[c.key] ?? "").trim();
      if (!v) continue;
      out[c.key] = c.kind === "number" ? Number(v) : v;
    }
    return out;
  });
}

// Denser than the console's other inputs (h-[30px]/rounded-md, not h-[34px]/rounded-lg) — rows repeat, so compact wins.
const CELL =
  "block h-[30px] w-full rounded-md border border-border bg-background px-2 font-mono text-[12px] text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-[3px] focus:ring-primary/15";

export function RowsEditor({ label, columns, rows, onChange, blank, addLabel, errors = [], min = 1 }: {
  label: string;
  columns: Column[];
  rows: Row[];
  onChange: (rows: Row[]) => void;
  blank: () => Row;
  addLabel: string;
  errors?: RowError[];
  /** Rows that cannot be removed (a split sheet needs one contributor). */
  min?: number;
}) {
  const uid = useId();
  const set = (i: number, key: string, value: string) => onChange(rows.map((r, j) => (j === i ? { ...r, [key]: value } : r)));
  const remove = (i: number) => onChange(rows.filter((_, j) => j !== i));
  const errorFor = (i: number, key: string) => errors.find((e) => e.row === i && e.key === key)?.message;
  return (
    <fieldset className="mb-2.5 min-w-0">
      <legend className="mb-1 block text-[11.5px] font-semibold text-foreground">{label}</legend>
      <div aria-hidden className="mb-1 flex gap-1.5 pr-7 text-[10.5px] font-semibold text-muted-foreground">
        {columns.map((c) => (
          <span key={c.key} className={`min-w-0 ${c.width ?? "flex-1"}`}>{c.label}</span>
        ))}
      </div>
      <div className="flex flex-col gap-1.5">
        {rows.map((row, i) => (
          <div key={i} className="flex items-start gap-1.5">
            {columns.map((c) => {
              const name = `${label} ${i + 1} ${c.label}`;
              const err = errorFor(i, c.key);
              const errId = err ? `${uid}-${i}-${c.key}` : undefined;
              return (
                <div key={c.key} className={`min-w-0 ${c.width ?? "flex-1"}`}>
                  {c.kind === "select" ? (() => {
                    const opts = optionsOf(c);
                    const v = row[c.key]?.trim();
                    return (
                      <select aria-label={name} value={row[c.key] ?? ""} onChange={(e) => set(i, c.key, e.target.value)} aria-invalid={!!err} aria-describedby={errId} className={`${CELL} ${err ? "border-destructive" : ""}`}>
                        <option value="">{c.placeholder ?? c.label}</option>
                        {v && !opts.includes(v) && <option value={row[c.key]}>{row[c.key]} (no longer listed)</option>}
                        {opts.map((o) => (
                          <option key={o} value={o}>{o}</option>
                        ))}
                      </select>
                    );
                  })() : (
                    <input
                      aria-label={name}
                      type={c.kind === "number" ? "number" : "text"}
                      min={c.kind === "number" ? c.min : undefined}
                      max={c.kind === "number" ? c.max : undefined}
                      placeholder={c.placeholder ?? c.label}
                      value={row[c.key] ?? ""}
                      onChange={(e) => set(i, c.key, e.target.value)}
                      aria-invalid={!!err}
                      aria-describedby={errId}
                      className={`${CELL} ${err ? "border-destructive" : ""}`}
                    />
                  )}
                  {err && <p id={errId} role="alert" className="mt-0.5 text-[11px] text-destructive">{err}</p>}
                </div>
              );
            })}
            <button
              type="button"
              aria-label={`Remove ${label} ${i + 1}`}
              disabled={rows.length <= min}
              onClick={() => remove(i)}
              className="mt-1 shrink-0 rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-30"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
      </div>
      <button type="button" onClick={() => onChange([...rows, blank()])} className="mt-1.5 inline-flex items-center gap-1 text-[11.5px] font-semibold text-primary hover:underline">
        <Plus className="h-3 w-3" /> {addLabel}
      </button>
    </fieldset>
  );
}

/** The exact body the console will send, read-only, so the form doubles as a
 * worked example of the request. */
export function RequestPreview({ body }: { body: unknown }) {
  const text = JSON.stringify(body, null, 2);
  return (
    <div className="mb-2.5 overflow-hidden rounded-lg border border-border">
      <div className="flex items-center border-b border-border bg-muted/50 px-2.5 py-1.5 text-[11px] font-semibold text-foreground">
        Request body — what the console sends
        <CopyButton text={text} label="Copy request body" className="ml-auto h-auto px-1.5 py-0.5" />
      </div>
      <pre role="region" aria-label="Request body" tabIndex={0} className="max-h-56 overflow-auto px-2.5 py-2 font-mono text-[11px] leading-relaxed text-foreground/90">{text}</pre>
    </div>
  );
}
