import { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Loader2 } from "lucide-react";

export interface RevisionCandidate {
  statement_id: string;
  name: string | null;
  period_start: string;
  period_end: string;
  total: number;
}

export type RevisionDecision = { replace: string } | { none: true };

interface RevisionPromptDialogProps {
  open: boolean;
  candidates: RevisionCandidate[];
  isSubmitting?: boolean;
  onDecide: (decision: RevisionDecision) => void;
  onCancel: () => void;
}

// Sentinel radio value for "these are new earnings, not a revision" — kept
// out of statement_id space since real ids are UUIDs.
const NEW_EARNINGS_VALUE = "__keep_both__";

const formatCurrency = (n: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(n || 0);

const candidateLabel = (c: RevisionCandidate) =>
  c.name?.trim() || `Statement covering ${c.period_start} – ${c.period_end}`;

/**
 * Shown when a newly-uploaded statement covers the same songs and time period
 * as one already calculated. The user decides whether it's a corrected
 * version of an earlier statement (we swap in the new numbers) or genuinely
 * new earnings (we keep both on record).
 */
export function RevisionPromptDialog({
  open,
  candidates,
  isSubmitting = false,
  onDecide,
  onCancel,
}: RevisionPromptDialogProps) {
  const [selected, setSelected] = useState<string>("");

  // Reset the selection whenever a fresh set of candidates is shown.
  useEffect(() => {
    setSelected("");
  }, [candidates]);

  const handleSubmit = () => {
    if (!selected) return;
    onDecide(selected === NEW_EARNINGS_VALUE ? { none: true } : { replace: selected });
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onCancel()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Is this an updated version of an earlier statement?</DialogTitle>
          <DialogDescription>
            This statement covers the same songs and time period as one you calculated before. If
            it's a corrected version, we'll swap in the new numbers and keep your payment history.
          </DialogDescription>
        </DialogHeader>

        <RadioGroup value={selected} onValueChange={setSelected} className="py-2 space-y-2">
          {candidates.map((c) => (
            <div key={c.statement_id} className="flex items-center gap-2">
              <RadioGroupItem value={c.statement_id} id={`revision-${c.statement_id}`} />
              <Label htmlFor={`revision-${c.statement_id}`} className="font-normal cursor-pointer">
                {candidateLabel(c)} — {formatCurrency(c.total)}
              </Label>
            </div>
          ))}
          <div className="flex items-center gap-2 pt-2 mt-1 border-t border-border">
            <RadioGroupItem value={NEW_EARNINGS_VALUE} id="revision-new-earnings" />
            <Label htmlFor="revision-new-earnings" className="font-normal cursor-pointer">
              These are new earnings — keep both
            </Label>
          </div>
        </RadioGroup>

        <DialogFooter>
          <Button variant="outline" onClick={onCancel} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={!selected || isSubmitting}>
            {isSubmitting && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            {selected === NEW_EARNINGS_VALUE ? "Keep both" : "Replace old numbers"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default RevisionPromptDialog;
