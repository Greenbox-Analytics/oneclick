import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, Plus, Trash2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { RegistryAvatar } from "./RegistryAvatar";
import type { SplitRow } from "./RoyaltySplitsTable";
import { SPLIT_PALETTE, clampPct, splitTotals } from "./splitsShared";

interface EditSplitsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Rows to seed the editor with — re-seeded each time the dialog opens. */
  initialRows: SplitRow[];
  /** Persist the edited rows. Resolves once saving is complete. */
  onSave: (rows: SplitRow[]) => Promise<void> | void;
}

/**
 * Spacious modal for editing a work's royalty splits — one block per party with
 * full-width name/role fields, so editing is legible even in the narrow work
 * sidebar. Holds its own draft and hands the result to `onSave`; the caller owns
 * persistence (stake CRUD). Imbalanced totals only warn — saving is never blocked.
 */
export function EditSplitsDialog({
  open,
  onOpenChange,
  initialRows,
  onSave,
}: EditSplitsDialogProps) {
  const [draft, setDraft] = useState<SplitRow[]>(initialRows);
  const [saving, setSaving] = useState(false);

  // Re-seed from the persisted rows whenever the dialog (re)opens, so Cancel +
  // reopen starts fresh rather than from abandoned edits.
  useEffect(() => {
    if (open) setDraft(initialRows);
  }, [open, initialRows]);

  const totals = useMemo(() => splitTotals(draft), [draft]);
  const balanced = totals.master === 100 && totals.publishing === 100;

  const setRow = (idx: number, patch: Partial<SplitRow>) =>
    setDraft((prev) => prev.map((r, i) => (i === idx ? { ...r, ...patch } : r)));

  const removeRow = (idx: number) => setDraft((prev) => prev.filter((_, i) => i !== idx));

  const addRow = () =>
    setDraft((prev) => [
      ...prev,
      { key: `new-${Date.now()}`, name: "", role: "", master: 0, publishing: 0 },
    ]);

  const handleSave = async () => {
    setSaving(true);
    try {
      await onSave(draft);
      onOpenChange(false);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(v) => !saving && onOpenChange(v)}>
      <DialogContent className="max-w-lg max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Edit royalty splits</DialogTitle>
          <DialogDescription>
            Set each party's master and publishing share. Each should total 100%.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto -mx-1 px-1 py-1 space-y-3">
          {draft.map((r, idx) => {
            const color = SPLIT_PALETTE[idx % SPLIT_PALETTE.length];
            return (
              <div key={r.key} className="rounded-xl border bg-muted/20 p-3">
                <div className="flex items-center gap-2.5 mb-3">
                  <RegistryAvatar name={r.name || "?"} color={color} size={32} />
                  <div className="min-w-0 flex-1">
                    {r.isYou ? (
                      <>
                        <div className="text-sm font-semibold truncate">{r.name || "You"}</div>
                        <div className="text-[11px] text-muted-foreground truncate">
                          {r.role || "Primary Artist"} · You
                        </div>
                      </>
                    ) : (
                      <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                        Party {idx + 1}
                      </div>
                    )}
                  </div>
                  {!r.isYou && (
                    <button
                      type="button"
                      onClick={() => removeRow(idx)}
                      className="shrink-0 p-1 text-muted-foreground hover:text-destructive"
                      title="Remove party"
                      aria-label={`Remove ${r.name || "party"}`}
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>

                {!r.isYou && (
                  <div className="grid grid-cols-2 gap-2 mb-2">
                    <div className="space-y-1">
                      <Label className="text-[11px] text-muted-foreground">Name</Label>
                      <Input
                        value={r.name}
                        placeholder="e.g. Jordan Lee"
                        onChange={(e) => setRow(idx, { name: e.target.value })}
                        className="h-9"
                      />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-[11px] text-muted-foreground">Role</Label>
                      <Input
                        value={r.role}
                        placeholder="e.g. Producer"
                        onChange={(e) => setRow(idx, { role: e.target.value })}
                        className="h-9"
                      />
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-2">
                  {(["master", "publishing"] as const).map((k) => (
                    <div key={k} className="space-y-1">
                      <Label className="text-[11px] capitalize text-muted-foreground">{k}</Label>
                      <div className="relative">
                        <Input
                          inputMode="numeric"
                          value={r[k] ?? 0}
                          onChange={(e) => setRow(idx, { [k]: clampPct(e.target.value) })}
                          className="h-9 pr-7 text-right font-mono tabular-nums"
                        />
                        <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">
                          %
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}

          <Button variant="outline" size="sm" className="w-full" onClick={addRow}>
            <Plus className="w-4 h-4 mr-1" /> Add party
          </Button>
        </div>

        <div className="flex items-center justify-between gap-3 border-t pt-3">
          <div
            className={cn(
              "inline-flex items-center gap-1.5 text-xs font-medium",
              balanced ? "text-emerald-400" : "text-amber-400"
            )}
          >
            {balanced ? (
              <CheckCircle2 className="w-4 h-4 shrink-0" />
            ) : (
              <AlertTriangle className="w-4 h-4 shrink-0" />
            )}
            <span>
              Master {totals.master}% · Publishing {totals.publishing}%
              {!balanced && " — should total 100%"}
            </span>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={saving}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={saving}>
              {saving ? "Saving…" : "Save changes"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
