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

export interface ConflictClaim {
  contract_ids: string[];
  contract_names?: string[];
  amount: number;
  percentage: number | null;
}

export interface Conflict {
  party_key: string;
  song_key: string;
  royalty_type_key: string;
  party_name: string;
  song_title: string;
  claims: ConflictClaim[];
}

export interface AssertedBy {
  id: string;
  name?: string;
  hash?: string;
}

/** One item of a `scope: "cross_run"` conflict payload — a saved value vs. a newly-calculated one. */
export interface CrossRunConflictItem {
  party_key: string;
  song_key: string;
  royalty_type_key: string;
  party_name: string;
  song_title: string;
  stored: { amount: number; percentage: number | null; asserted_by: AssertedBy[] };
  new: { amount: number; percentage: number | null; contract_ids: string[] };
}

export interface ConflictGatePayload {
  scope: "in_run" | "cross_run";
  conflicts: Conflict[] | CrossRunConflictItem[];
}

export interface ConflictResolution {
  party_key: string;
  song_key: string;
  royalty_type_key: string;
  governing_contract_id: string;
}

const conflictKey = (c: Pick<Conflict, "party_key" | "song_key" | "royalty_type_key">) =>
  `${c.party_key}|${c.song_key}|${c.royalty_type_key}`;

const claimLabel = (claim: ConflictClaim) =>
  claim.contract_names && claim.contract_names.length > 0
    ? claim.contract_names.join(", ")
    : claim.contract_ids.join(", ");

const formatCurrency = (n: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(n || 0);

/**
 * Turns a "cross_run" conflict payload (one value saved from an earlier
 * calculation vs. one just calculated) into the same two-claim shape as an
 * "in_run" conflict, so both scopes can render through this one dialog.
 */
export function normalizeConflictPayload(payload: ConflictGatePayload): Conflict[] {
  if (payload.scope !== "cross_run") return payload.conflicts as Conflict[];
  return (payload.conflicts as CrossRunConflictItem[]).map((item) => {
    const storedBy = item.stored.asserted_by?.[0];
    const storedClaim: ConflictClaim = {
      contract_ids: storedBy?.id ? [storedBy.id] : [],
      contract_names: [storedBy?.name || "a contract used earlier"],
      amount: item.stored.amount,
      percentage: item.stored.percentage,
    };
    const newClaim: ConflictClaim = {
      contract_ids: item.new.contract_ids,
      amount: item.new.amount,
      percentage: item.new.percentage,
    };
    return {
      party_key: item.party_key,
      song_key: item.song_key,
      royalty_type_key: item.royalty_type_key,
      party_name: item.party_name,
      song_title: item.song_title,
      claims: [storedClaim, newClaim],
    };
  });
}

interface ConflictResolutionDialogProps {
  open: boolean;
  conflicts: Conflict[];
  isSubmitting?: boolean;
  onResolve: (resolutions: ConflictResolution[]) => void;
  onCancel: () => void;
}

/**
 * Shown when two contracts disagree about a collaborator's share of a song —
 * either within the documents just uploaded, or against a split saved from an
 * earlier calculation (already normalized into the same shape by the caller
 * via `normalizeConflictPayload`). The user must pick the correct contract for
 * every conflict before continuing; nothing is saved until they do.
 */
export function ConflictResolutionDialog({
  open,
  conflicts,
  isSubmitting = false,
  onResolve,
  onCancel,
}: ConflictResolutionDialogProps) {
  const [picks, setPicks] = useState<Record<string, string>>({});

  // Reset selections whenever a fresh set of conflicts is shown.
  useEffect(() => {
    setPicks({});
  }, [conflicts]);

  const allPicked = conflicts.length > 0 && conflicts.every((c) => picks[conflictKey(c)]);

  const handleSubmit = () => {
    if (!allPicked) return;
    const resolutions: ConflictResolution[] = conflicts.map((c) => ({
      party_key: c.party_key,
      song_key: c.song_key,
      royalty_type_key: c.royalty_type_key,
      governing_contract_id: picks[conflictKey(c)],
    }));
    onResolve(resolutions);
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onCancel()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Your contracts disagree on a split</DialogTitle>
          <DialogDescription>
            Two contracts give the same person different shares of the same song. Pick which
            contract is correct — nothing is saved until you choose.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2 max-h-[55vh] overflow-y-auto">
          {conflicts.map((conflict) => {
            const key = conflictKey(conflict);
            return (
              <div key={key} className="rounded-md border border-border p-3 space-y-2">
                <p className="text-sm font-medium">
                  {conflict.party_name}
                  <span className="text-muted-foreground font-normal"> — {conflict.song_title}</span>
                </p>
                <RadioGroup
                  value={picks[key] ?? ""}
                  onValueChange={(value) => setPicks((p) => ({ ...p, [key]: value }))}
                >
                  {conflict.claims.map((claim, i) => {
                    const value = claim.contract_ids[0] ?? `${key}-${i}`;
                    const inputId = `${key}-claim-${i}`;
                    return (
                      <div key={inputId} className="flex items-center gap-2">
                        <RadioGroupItem value={value} id={inputId} />
                        <Label htmlFor={inputId} className="font-normal cursor-pointer">
                          {claimLabel(claim)} —{" "}
                          {claim.percentage != null ? `${claim.percentage}%` : "unknown %"} (
                          {formatCurrency(claim.amount)})
                        </Label>
                      </div>
                    );
                  })}
                </RadioGroup>
              </div>
            );
          })}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onCancel} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={!allPicked || isSubmitting}>
            {isSubmitting && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            Use selected splits
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default ConflictResolutionDialog;
