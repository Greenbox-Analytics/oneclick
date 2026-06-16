import { useMemo } from "react";
import { AlertTriangle, CheckCircle2, Plus, Sparkles, Pencil, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { RegistryAvatar } from "./RegistryAvatar";

export interface SplitRow {
  /** Stable id so React keys are predictable. */
  key: string;
  name: string;
  role: string;
  /** UI-only flag for the user's own row. */
  isYou?: boolean;
  /** Master royalties percentage 0–100. */
  master: number;
  /** Publishing royalties percentage 0–100. */
  publishing: number;
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
}

const clampPct = (v: string | number): number => {
  const n = parseInt(String(v).replace(/[^0-9]/g, ""), 10);
  if (Number.isNaN(n)) return 0;
  return Math.max(0, Math.min(100, n));
};

const COLLAB_PALETTE = [
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
}: RoyaltySplitsTableProps) {
  const totals = useMemo(
    () =>
      rows.reduce(
        (acc, r) => ({
          master: acc.master + (r.master || 0),
          publishing: acc.publishing + (r.publishing || 0),
        }),
        { master: 0, publishing: 0 }
      ),
    [rows]
  );
  const balanced = totals.master === 100 && totals.publishing === 100;

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

      {/* column headings */}
      <div className="grid grid-cols-[1fr_72px_72px_auto] gap-2 px-1 text-[10px] uppercase tracking-wider text-muted-foreground/70 font-semibold mb-1">
        <span>Party</span>
        <span className="text-center">Master</span>
        <span className="text-center">Publishing</span>
        {editable && allowAddRow && <span />}
      </div>

      {/* rows */}
      <div className="divide-y divide-border/60">
        {rows.map((r, idx) => {
          const palette = COLLAB_PALETTE[idx % COLLAB_PALETTE.length];
          return (
            <div
              key={r.key}
              className="grid grid-cols-[1fr_72px_72px_auto] gap-2 items-center py-2"
            >
              <div className="flex items-center gap-2 min-w-0">
                <RegistryAvatar name={r.name || "?"} color={palette} size={26} />
                <div className="min-w-0">
                  {editable && !r.isYou ? (
                    <Input
                      value={r.name}
                      placeholder="Name"
                      onChange={(e) => setRow(idx, { name: e.target.value })}
                      className="h-7 text-xs px-2"
                    />
                  ) : (
                    <div className="text-xs font-semibold truncate">
                      {r.name || "Unnamed"}
                    </div>
                  )}
                  {editable && !r.isYou ? (
                    <Input
                      value={r.role}
                      placeholder="Role"
                      onChange={(e) => setRow(idx, { role: e.target.value })}
                      className="h-6 text-[11px] px-2 mt-1"
                    />
                  ) : (
                    <div className="text-[11px] text-muted-foreground truncate">{r.role}</div>
                  )}
                </div>
              </div>

              {(["master", "publishing"] as const).map((k) => (
                <div key={k} className="flex items-center justify-center">
                  {editable ? (
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
                  ) : (
                    <span
                      className={cn(
                        "font-mono text-xs font-bold tabular-nums",
                        (r[k] || 0) === 0 && "text-muted-foreground"
                      )}
                    >
                      {r[k] ?? 0}%
                    </span>
                  )}
                </div>
              ))}

              {editable && allowAddRow && (
                <div className="flex justify-center">
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
          );
        })}
      </div>

      {/* totals + add */}
      <div className="flex items-center justify-between mt-3">
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
    </div>
  );
}
