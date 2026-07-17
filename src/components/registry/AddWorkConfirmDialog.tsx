import { useMemo } from "react";
import { AlertTriangle, CheckCircle2, Info, Loader2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { RoyaltySplitsTable, type SplitRow } from "./RoyaltySplitsTable";

interface AddWorkConfirmDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  workTitle: string;
  artistName: string;
  projectName?: string;
  /** The split rows that will be saved as stakes (0% rows are skipped). */
  rows: SplitRow[];
  submitting: boolean;
  onConfirm: () => void;
}

/**
 * Final review before the work + its royalty stakes are created. Shows every
 * party with their master/publishing split and flags totals that don't reach
 * 100% — the user can still proceed and fix splits later on the work page.
 */
export function AddWorkConfirmDialog({
  open,
  onOpenChange,
  workTitle,
  artistName,
  projectName,
  rows,
  submitting,
  onConfirm,
}: AddWorkConfirmDialogProps) {
  const namedRows = useMemo(() => rows.filter((r) => r.name.trim()), [rows]);
  const totals = useMemo(
    () =>
      namedRows.reduce(
        (acc, r) => ({
          master: acc.master + (r.master || 0),
          publishing: acc.publishing + (r.publishing || 0),
        }),
        { master: 0, publishing: 0 }
      ),
    [namedRows]
  );
  const hasSplits = namedRows.some(
    (r) => (r.master || 0) > 0 || (r.publishing || 0) > 0 || (r.soundexchange || 0) > 0
  );
  const balanced = totals.master === 100 && totals.publishing === 100;

  return (
    <Dialog open={open} onOpenChange={(v) => !submitting && onOpenChange(v)}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Confirm and add “{workTitle}”</DialogTitle>
          <DialogDescription>
            {artistName}
            {projectName ? ` · ${projectName}` : ""} — check the parties and splits below before
            the work is added.
          </DialogDescription>
        </DialogHeader>

        {namedRows.length > 0 ? (
          <>
            <div className="max-h-72 overflow-y-auto pr-1">
              <RoyaltySplitsTable rows={namedRows} />
            </div>
            {hasSplits && !balanced && (
              <div className="flex items-start gap-2 text-xs bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded-lg px-3 py-2">
                <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                <span>
                  Splits don't total 100% (Master {totals.master}% · Publishing {totals.publishing}
                  %). You can still add the work and fix them later on the work page.
                </span>
              </div>
            )}
          </>
        ) : null}

        {!hasSplits && (
          <div className="flex items-start gap-2 text-xs bg-muted/60 text-muted-foreground border rounded-lg px-3 py-2">
            <Info className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            <span>
              No royalty splits will be saved with this work — you can add them any time on the
              work page.
            </span>
          </div>
        )}

        <DialogFooter className="gap-2">
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={submitting}>
            Go back
          </Button>
          <Button onClick={onConfirm} disabled={submitting}>
            {submitting ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <CheckCircle2 className="w-4 h-4 mr-2" />
            )}
            Confirm &amp; add work
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
