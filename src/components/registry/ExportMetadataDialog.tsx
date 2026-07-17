import { useEffect, useMemo, useState } from "react";
import { Download, Loader2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { RegistryAvatar } from "./RegistryAvatar";
import type { SplitRow } from "./RoyaltySplitsTable";
import { SPLIT_PALETTE } from "./splitsShared";

interface ExportMetadataDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Parties on the work — pass WorkEditor's `stakeRows`. */
  parties: SplitRow[];
  isExporting: boolean;
  /** Called with the names of parties whose splits should be hidden. */
  onExport: (hiddenNames: string[]) => void;
}

/**
 * Pre-export options modal for the metadata PDF. Lists every party on the work
 * with a per-party toggle (default ON = show splits). Parties switched OFF are
 * still listed in the PDF but their percentages are replaced with a "Withheld"
 * message — so a user can share metadata with one party without disclosing the
 * others' ownership splits. Visibility choices are per-export only; nothing is
 * persisted.
 */
export function ExportMetadataDialog({
  open,
  onOpenChange,
  parties,
  isExporting,
  onExport,
}: ExportMetadataDialogProps) {
  // Set of party names whose splits are visible. Seeded to all-visible each
  // time the dialog opens, so Cancel + reopen starts fresh.
  const [visible, setVisible] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (open) setVisible(new Set(parties.map((p) => p.name)));
  }, [open, parties]);

  const toggle = (name: string) =>
    setVisible((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });

  const hiddenCount = useMemo(
    () => parties.filter((p) => !visible.has(p.name)).length,
    [parties, visible]
  );

  const handleExport = () => {
    const hidden = parties.filter((p) => !visible.has(p.name)).map((p) => p.name);
    onExport(hidden);
  };

  return (
    <Dialog open={open} onOpenChange={(v) => !isExporting && onOpenChange(v)}>
      <DialogContent className="max-w-lg max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Export metadata</DialogTitle>
          <DialogDescription>
            Choose whose splits to include. Parties you switch off will appear in the
            PDF without their percentages.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto -mx-1 px-1 py-1 space-y-2">
          {parties.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4 text-center">
              No parties recorded on this work.
            </p>
          ) : (
            parties.map((p, idx) => {
              const color = SPLIT_PALETTE[idx % SPLIT_PALETTE.length];
              const isVisible = visible.has(p.name);
              return (
                <div
                  key={p.key}
                  className="flex items-center gap-3 rounded-xl border bg-muted/20 p-3"
                >
                  <RegistryAvatar name={p.name || "?"} color={color} size={32} />
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-semibold truncate">
                      {p.name || "Unnamed party"}
                      {p.isYou && (
                        <span className="text-[11px] font-normal text-muted-foreground">
                          {" "}
                          · You
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-muted-foreground truncate">
                      {p.role || "—"} · Master {p.master ?? 0}% · Publishing{" "}
                      {p.publishing ?? 0}%
                    </div>
                  </div>
                  <div className="flex shrink-0 flex-col items-end gap-0.5">
                    <Switch
                      checked={isVisible}
                      onCheckedChange={() => toggle(p.name)}
                      aria-label={`Show ${p.name || "party"}'s splits`}
                    />
                    <span className="text-[10px] text-muted-foreground">
                      {isVisible ? "Shown" : "Withheld"}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        <div className="flex items-center justify-between gap-3 border-t pt-3">
          <div className="text-xs text-muted-foreground">
            {hiddenCount === 0
              ? "All splits will be shown."
              : `${hiddenCount} part${hiddenCount === 1 ? "y" : "ies"} withheld.`}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Button variant="ghost" onClick={() => onOpenChange(false)} disabled={isExporting}>
              Cancel
            </Button>
            <Button onClick={handleExport} disabled={isExporting}>
              {isExporting ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Download className="w-4 h-4 mr-2" />
              )}
              Export
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
