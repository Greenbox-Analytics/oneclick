import { useMemo } from "react";
import { AlertTriangle, CheckCircle2, Plus, Sparkles, Pencil, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { RegistryAvatar } from "./RegistryAvatar";
import { clampPct, SPLIT_PALETTE, splitTotals } from "./splitsShared";

export interface SplitRow {
  /** Stable id so React keys are predictable. */
  key: string;
  name: string;
  role: string;
  /** UI-only flag for the user's own row. */
  isYou?: boolean;
  /** UI-only "p/k/a Stage Name" note shown under the (legal) name. Not persisted. */
  aliasNote?: string;
  /** Master royalties percentage 0–100. */
  master: number;
  /** Publishing royalties percentage 0–100. */
  publishing: number;
  /** SoundExchange (US digital performance) % — paid directly by SoundExchange,
   *  tracked separately and never counted toward the master total. */
  soundexchange?: number;
  status?: string;
}

export interface SplitsSource {
  type: "ai" | "manual";
  file?: string;
  date: string; // YYYY-MM-DD
}

interface RoyaltySplitsTableProps {
  rows: SplitRow[];
  onChange?: (next: SplitRow[]) => void;
  /** Display-only when omitted; edit handles appear when true. */
  editable?: boolean;
  /** Show the "Edit / Done" header button. */
  showEditToggle?: boolean;
  isEditing?: boolean;
  onToggleEdit?: () => void;
  /** "Parsed from contract.pdf · Apr 12" eyebrow. */
  source?: SplitsSource | null;
  /** Render an "Add party" button at the bottom. */
  allowAddRow?: boolean;
  /** Hide totals warning until the user has interacted (wizard UX). */
  warnOnImbalance?: boolean;
  /** Show the summed "Master X% · Publishing Y%" line. Hidden for viewers who
   *  only receive their own slice of the splits, where a "total" is meaningless. */
  showTotals?: boolean;
  /** Force the SoundExchange section on/off. Defaults to auto: shown when any
   *  row carries a SoundExchange share. Pass an explicit value while editing
   *  so the section doesn't vanish when a value is zeroed mid-edit. */
  showSoundExchange?: boolean;
}

export const clampPct = (v: string | number): number => {
  const n = parseInt(String(v).replace(/[^0-9]/g, ""), 10);
  if (Number.isNaN(n)) return 0;
  return Math.max(0, Math.min(100, n));
};

export const COLLAB_PALETTE = [
  "#7c5cff",
  "#1f8a5b",
  "#d9762b",
  "#d24b6e",
  "#2a6fdb",
  "#0f6b43",
];

const fmtDate = (iso: string): string => {
  if (!iso) return "";
  try {
    return new Date(iso + (iso.length === 10 ? "T00:00:00" : "")).toLocaleDateString(
      "en-US",
      { month: "short", day: "numeric", year: "numeric" }
    );
  } catch {
    return iso;
  }
};

/** Shared editable splits table used by the WorkEditor sidebar AND the wizard. */
export function RoyaltySplitsTable({
  rows,
  onChange,
  editable = false,
  showEditToggle = false,
  isEditing = false,
  onToggleEdit,
  source,
  allowAddRow = false,
  warnOnImbalance = true,
  showTotals = true,
  showSoundExchange,
}: RoyaltySplitsTableProps) {
  const totals = useMemo(() => splitTotals(rows), [rows]);
  const balanced = totals.master === 100 && totals.publishing === 100;
  // SoundExchange shares render in their own section below the table — never
  // as a column, never in `balanced` (SoundExchange pays parties directly, so
  // there's no 100% expectation against the master split).
  const showSX = showSoundExchange ?? rows.some((r) => (r.soundexchange ?? 0) > 0);

  const setRow = (idx: number, patch: Partial<SplitRow>) => {
    if (!onChange) return;
    onChange(rows.map((r, i) => (i === idx ? { ...r, ...patch } : r)));
  };

  const removeRow = (idx: number) => {
    if (!onChange) return;
    onChange(rows.filter((_, i) => i !== idx));
  };

  const addRow = () => {
    if (!onChange) return;
    onChange([
      ...rows,
      {
        key: `new-${Date.now()}`,
        name: "",
        role: "",
        master: 0,
        publishing: 0,
      },
    ]);
  };

  return (
    <div>
      {/* header row */}
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs font-semibold tracking-wider uppercase text-muted-foreground">
          Royalty splits
        </div>
        {showEditToggle && onToggleEdit && (
          <button
            type="button"
            onClick={onToggleEdit}
            className="inline-flex items-center gap-1 text-xs font-semibold text-primary hover:text-primary/80"
          >
            {isEditing ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5" /> Done
              </>
            ) : (
              <>
                <Pencil className="w-3.5 h-3.5" /> Edit
              </>
            )}
          </button>
        )}
      </div>

      {source && (
        <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground mb-3">
          {source.type === "ai" ? (
            <Sparkles className="w-3 h-3" />
          ) : (
            <Pencil className="w-3 h-3" />
          )}
          {source.type === "ai" ? (
            <span>
              Parsed from <b className="text-foreground">{source.file}</b> · {fmtDate(source.date)}
            </span>
          ) : (
            <span>Edited manually · {fmtDate(source.date)}</span>
          )}
        </div>
      )}

      {/* column headings — edit mode labels the % inputs inline instead */}
      {!editable && (
        <div className="grid grid-cols-[1fr_72px_72px] gap-2 px-1 text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold mb-1">
          <span>Party</span>
          <span className="text-center">Master</span>
          <span className="text-center">Publishing</span>
        </div>
      )}

      {/* rows */}
      <div className="divide-y divide-border/60">
        {rows.map((r, idx) => {
          const palette = COLLAB_PALETTE[idx % COLLAB_PALETTE.length];

          if (editable) {
            // Edit mode stacks each party: full-width name/role inputs so
            // nothing gets clipped while typing, % fields labeled below.
            return (
              <div key={r.key} className="py-2.5 space-y-1.5">
                <div className="flex items-start gap-2">
                  <RegistryAvatar name={r.name || "?"} color={palette} size={26} />
                  <div className="flex-1 min-w-0 space-y-1">
                    {r.isYou ? (
                      <div className="text-xs font-semibold break-words leading-snug pt-1">
                        {r.name || "Unnamed"}
                      </div>
                    ) : (
                      <Input
                        value={r.name}
                        placeholder="Name"
                        onChange={(e) => setRow(idx, { name: e.target.value })}
                        className="h-7 text-xs px-2"
                      />
                    )}
                    {r.aliasNote && (
                      <div className="text-[11px] text-muted-foreground/80 italic break-words leading-snug">
                        {r.aliasNote}
                      </div>
                    )}
                    {r.isYou ? (
                      <div className="text-[11px] text-muted-foreground break-words leading-snug">
                        {r.role}
                      </div>
                    ) : (
                      <Input
                        value={r.role}
                        placeholder="Role"
                        onChange={(e) => setRow(idx, { role: e.target.value })}
                        className="h-7 text-[11px] px-2"
                      />
                    )}
                  </div>
                  {allowAddRow && (
                    <div className="w-6 flex justify-center pt-1">
                      {!r.isYou && (
                        <button
                          type="button"
                          onClick={() => removeRow(idx)}
                          className="text-muted-foreground hover:text-destructive p-1"
                          title="Remove party"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-3 pl-[34px]">
                  {(["master", "publishing"] as const).map((k) => (
                    <label key={k} className="flex items-center gap-1.5">
                      <span className="text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold">
                        {k === "master" ? "Master" : "Publishing"}
                      </span>
                      <div className="relative">
                        <input
                          inputMode="numeric"
                          value={r[k] ?? 0}
                          onChange={(e) => setRow(idx, { [k]: clampPct(e.target.value) })}
                          className="w-[58px] h-7 rounded-md border bg-background text-right pr-4 pl-2 text-xs tabular-nums font-mono focus:outline-none focus:ring-1 focus:ring-primary"
                        />
                        <span className="absolute right-1 top-1/2 -translate-y-1/2 text-[10px] text-muted-foreground">
                          %
                        </span>
                      </div>
                    </label>
                  ))}
                </div>
              </div>
            );
          }

          return (
            <div
              key={r.key}
              className="grid grid-cols-[1fr_72px_72px] gap-2 items-center py-2"
            >
              <div className="flex items-center gap-2 min-w-0">
                <RegistryAvatar name={r.name || "?"} color={palette} size={26} />
                <div className="min-w-0">
                  <div className="text-xs font-semibold break-words leading-snug">
                    {r.name || "Unnamed"}
                  </div>
                  {r.aliasNote && (
                    <div className="text-[11px] text-muted-foreground/80 italic break-words leading-snug">
                      {r.aliasNote}
                    </div>
                  )}
                  <div className="text-[11px] text-muted-foreground break-words leading-snug">{r.role}</div>
                </div>
              </div>

              {(["master", "publishing"] as const).map((k) => (
                <div key={k} className="flex items-center justify-center">
                  <span
                    className={cn(
                      "font-mono text-xs font-bold tabular-nums",
                      (r[k] || 0) === 0 && "text-muted-foreground"
                    )}
                  >
                    {r[k] ?? 0}%
                  </span>
                </div>
              ))}
            </div>
          );
        })}
      </div>

      {/* totals + add */}
      {(showTotals || (editable && allowAddRow)) && (
        <div className="flex items-center justify-between mt-3">
          {showTotals ? (
            <div
              className={cn(
                "inline-flex items-center gap-1.5 text-[11px] font-medium",
                balanced
                  ? "text-emerald-400"
                  : warnOnImbalance
                  ? "text-amber-400"
                  : "text-muted-foreground"
              )}
            >
              {balanced ? (
                <CheckCircle2 className="w-3.5 h-3.5" />
              ) : (
                <AlertTriangle className="w-3.5 h-3.5" />
              )}
              <span>
                Master {totals.master}% · Publishing {totals.publishing}%
                {!balanced && warnOnImbalance && " — should total 100%"}
              </span>
            </div>
          ) : (
            <span />
          )}
          {editable && allowAddRow && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs"
              onClick={addRow}
            >
              <Plus className="w-3 h-3 mr-1" /> Add party
            </Button>
          )}
        </div>
      )}
      {/* SoundExchange — its own section, never a column and never part of the
          master total. In edit mode every row gets an input so a share can be
          adjusted or zeroed (zeroing deletes the stake on save). */}
      {showSX && (
        <div className="mt-4 pt-3 border-t border-border/60">
          <div className="text-xs font-semibold tracking-wider uppercase text-muted-foreground mb-1">
            SoundExchange
          </div>
          <div className="divide-y divide-border/60">
            {rows.map((r, idx) => {
              if (!editable && (r.soundexchange ?? 0) <= 0) return null;
              const palette = COLLAB_PALETTE[idx % COLLAB_PALETTE.length];
              return (
                <div key={r.key} className="flex items-center gap-2 py-2">
                  <RegistryAvatar name={r.name || "?"} color={palette} size={26} />
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-semibold break-words leading-snug">
                      {r.name || "Unnamed"}
                    </div>
                    {r.role && (
                      <div className="text-[11px] text-muted-foreground break-words leading-snug">
                        {r.role}
                      </div>
                    )}
                  </div>
                  {editable ? (
                    <div className="relative shrink-0">
                      <input
                        inputMode="numeric"
                        value={r.soundexchange ?? 0}
                        onChange={(e) => setRow(idx, { soundexchange: clampPct(e.target.value) })}
                        className="w-[58px] h-7 rounded-md border bg-background text-right pr-4 pl-2 text-xs tabular-nums font-mono focus:outline-none focus:ring-1 focus:ring-primary"
                      />
                      <span className="absolute right-1 top-1/2 -translate-y-1/2 text-[10px] text-muted-foreground">
                        %
                      </span>
                    </div>
                  ) : (
                    <span className="font-mono text-xs font-bold tabular-nums shrink-0">
                      {r.soundexchange ?? 0}%
                    </span>
                  )}
                </div>
              );
            })}
          </div>
          <p className="text-[11px] text-muted-foreground mt-1.5">
            SoundExchange royalties are paid directly by SoundExchange and aren't
            counted in the master total.
          </p>
        </div>
      )}
    </div>
  );
}
