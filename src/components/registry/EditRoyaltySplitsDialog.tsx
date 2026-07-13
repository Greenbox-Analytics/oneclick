import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Loader2, Plus, Trash2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { RegistryAvatar } from "./RegistryAvatar";
import { clampPct, COLLAB_PALETTE, type SplitRow } from "./RoyaltySplitsTable";

interface EditRoyaltySplitsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The rows to edit — the dialog keeps its own draft copy. */
  rows: SplitRow[];
  saving: boolean;
  onSave: (rows: SplitRow[]) => void | Promise<void>;
}

/** Labeled percentage input with an inline "%" suffix. */
function PctField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block min-w-0">
      <span className="text-[11px] font-medium text-muted-foreground mb-1 block">{label}</span>
      <div className="relative">
        <input
          inputMode="numeric"
          value={value}
          onChange={(e) => onChange(clampPct(e.target.value))}
          className="w-full h-9 rounded-md border bg-background text-right pr-6 pl-2 text-sm tabular-nums font-mono focus:outline-none focus:ring-1 focus:ring-primary"
        />
        <span className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">
          %
        </span>
      </div>
    </label>
  );
}

/**
 * Modal editor for a work's royalty splits (works page). One card per party;
 * the "You" row keeps its name/role fixed. Every card carries a SoundExchange
 * field so a share can be added by hand — zeroing one removes the stake on
 * save. Saving is delegated to the caller (SplitsSidebar's stake diff).
 */
export function EditRoyaltySplitsDialog({
  open,
  onOpenChange,
  rows,
  saving,
  onSave,
}: EditRoyaltySplitsDialogProps) {
  const [draft, setDraft] = useState<SplitRow[]>(rows);

  // Re-seed the draft each time the dialog opens so Cancel discards edits.
  useEffect(() => {
    if (open) setDraft(rows);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const totals = useMemo(
    () =>
      draft.reduce(
        (acc, r) => ({
          master: acc.master + (r.master || 0),
          publishing: acc.publishing + (r.publishing || 0),
        }),
        { master: 0, publishing: 0 }
      ),
    [draft]
  );
  const balanced = totals.master === 100 && totals.publishing === 100;

  const setRow = (idx: number, patch: Partial<SplitRow>) =>
    setDraft((prev) => prev.map((r, i) => (i === idx ? { ...r, ...patch } : r)));

  const removeRow = (idx: number) => setDraft((prev) => prev.filter((_, i) => i !== idx));

  const addRow = () =>
    setDraft((prev) => [
      ...prev,
      {
        key: `new-${Date.now()}`,
        name: "",
        role: "",
        master: 0,
        publishing: 0,
        soundexchange: 0,
      },
    ]);

  return (
    <Dialog open={open} onOpenChange={(v) => !saving && onOpenChange(v)}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Edit royalty splits</DialogTitle>
          <DialogDescription>
            Set each party's master and publishing share. Each should total 100%.
          </DialogDescription>
        </DialogHeader>

        <div className="max-h-[65vh] overflow-y-auto space-y-3 pr-1 -mr-1">
          {draft.map((r, idx) => {
            const palette = COLLAB_PALETTE[idx % COLLAB_PALETTE.length];
            return (
              <div key={r.key} className="rounded-xl border bg-muted/20 p-4 space-y-3">
                {r.isYou ? (
                  <div className="flex items-center gap-2.5">
                    <RegistryAvatar name={r.name || "?"} color={palette} size={30} />
                    <div className="min-w-0">
                      <div className="text-sm font-semibold break-words leading-snug">
                        {r.name || "Unnamed"}
                      </div>
                      <div className="text-xs text-muted-foreground break-words leading-snug">
                        {r.role ? `${r.role} · You` : "You"}
                      </div>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="flex items-center gap-2.5">
                      <RegistryAvatar name={r.name || "?"} color={palette} size={30} />
                      <span className="flex-1 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                        {r.name.trim() || `Party ${idx + 1}`}
                      </span>
                      <button
                        type="button"
                        onClick={() => removeRow(idx)}
                        className="text-muted-foreground hover:text-destructive p-1"
                        title="Remove party"
                        aria-label={`Remove ${r.name.trim() || `party ${idx + 1}`}`}
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <label className="block min-w-0">
                        <span className="text-[11px] font-medium text-muted-foreground mb-1 block">
                          Name
                        </span>
                        <Input
                          value={r.name}
                          placeholder="Name"
                          onChange={(e) => setRow(idx, { name: e.target.value })}
                          className="h-9 text-sm"
                        />
                      </label>
                      <label className="block min-w-0">
                        <span className="text-[11px] font-medium text-muted-foreground mb-1 block">
                          Role
                        </span>
                        <Input
                          value={r.role}
                          placeholder="e.g. producer"
                          onChange={(e) => setRow(idx, { role: e.target.value })}
                          className="h-9 text-sm"
                        />
                      </label>
                    </div>
                  </>
                )}
                <div className="grid grid-cols-3 gap-3">
                  <PctField
                    label="Master"
                    value={r.master ?? 0}
                    onChange={(v) => setRow(idx, { master: v })}
                  />
                  <PctField
                    label="Publishing"
                    value={r.publishing ?? 0}
                    onChange={(v) => setRow(idx, { publishing: v })}
                  />
                  <PctField
                    label="SoundExchange"
                    value={r.soundexchange ?? 0}
                    onChange={(v) => setRow(idx, { soundexchange: v })}
                  />
                </div>
              </div>
            );
          })}

          <Button type="button" variant="ghost" size="sm" className="h-8 text-xs" onClick={addRow}>
            <Plus className="w-3.5 h-3.5 mr-1" /> Add party
          </Button>
        </div>

        <DialogFooter className="gap-2 sm:justify-between items-center pt-2 border-t">
          {!balanced ? (
            <div className="flex items-start gap-1.5 text-[11px] font-medium text-amber-400">
              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
              <span>
                Master {totals.master}% · Publishing {totals.publishing}% — should total 100%
              </span>
            </div>
          ) : (
            <span />
          )}
          <div className="flex items-center gap-2">
            <Button variant="outline" onClick={() => onOpenChange(false)} disabled={saving}>
              Cancel
            </Button>
            <Button onClick={() => onSave(draft)} disabled={saving}>
              {saving && <Loader2 className="w-4 h-4 mr-1.5 animate-spin" />}
              Save changes
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
