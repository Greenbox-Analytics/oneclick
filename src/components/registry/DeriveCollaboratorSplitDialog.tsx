import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, Loader2, Plus, Sparkles, X } from "lucide-react";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import { type DeriveResult, useDeriveFromContracts } from "@/hooks/useRegistry";
import { useWorkFiles } from "@/hooks/useWorkFiles";

interface TermRow {
  label: string;
  value: string;
}

interface Props {
  workId: string;
  collaboratorName: string;
  open: boolean;
  onOpenChange: (o: boolean) => void;
  onApply: (result: {
    masterPct: number;
    publishingPct: number;
    terms: Array<{ label: string; value: string }>;
    matchedFileIds: string[];
  }) => void;
}

export default function DeriveCollaboratorSplitDialog({
  workId,
  collaboratorName,
  open,
  onOpenChange,
  onApply,
}: Props) {
  const filesQuery = useWorkFiles(workId);
  const derive = useDeriveFromContracts();

  // The work's linked documents, multi-select. Default = all selected.
  const documents = useMemo(
    () =>
      (filesQuery.data || [])
        .filter((wf) => wf.project_files)
        .map((wf) => ({ id: wf.project_files!.id, label: wf.project_files!.file_name })),
    [filesQuery.data]
  );

  const [selectedContractIds, setSelectedContractIds] = useState<string[]>([]);
  const [result, setResult] = useState<DeriveResult | null>(null);
  const [masterPct, setMasterPct] = useState("");
  const [publishingPct, setPublishingPct] = useState("");
  const [terms, setTerms] = useState<TermRow[]>([]);

  // Reset everything when the dialog opens; default-select all linked documents.
  useEffect(() => {
    if (open) {
      setSelectedContractIds(documents.map((d) => d.id));
      setResult(null);
      setMasterPct("");
      setPublishingPct("");
      setTerms([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Once documents finish loading after the dialog is already open, default-select them.
  useEffect(() => {
    if (open && !result && selectedContractIds.length === 0 && documents.length > 0) {
      setSelectedContractIds(documents.map((d) => d.id));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documents.length, open]);

  const toggleContract = (id: string) =>
    setSelectedContractIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );

  const handleDerive = async () => {
    const res = await derive.mutateAsync({
      work_id: workId,
      name: collaboratorName,
      contract_file_ids: selectedContractIds,
    });
    setResult(res);
    setMasterPct(res.master_pct != null ? String(res.master_pct) : "0");
    setPublishingPct(res.publishing_pct != null ? String(res.publishing_pct) : "0");
    setTerms((res.terms || []).map((t) => ({ label: t.label, value: t.value })));
  };

  const handleEnterManually = () => {
    // Skip derivation entirely: close with a zeroed split so the user fills the
    // invite form manually.
    onApply({ masterPct: 0, publishingPct: 0, terms: [], matchedFileIds: [] });
    onOpenChange(false);
  };

  const handleApply = () => {
    onApply({
      masterPct: parseFloat(masterPct) || 0,
      publishingPct: parseFloat(publishingPct) || 0,
      terms: terms.filter((t) => t.label.trim() || t.value.trim()),
      matchedFileIds: result?.matched_file_ids || [],
    });
    onOpenChange(false);
  };

  const isHighConfidence = !!result && result.found && result.confidence === "high";
  const showAmberBanner = !!result && (!result.found || result.confidence !== "high");

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Fill in details from contracts</DialogTitle>
        </DialogHeader>

        <div className="space-y-4 pt-1">
          {/* Choose contracts */}
          <div className="space-y-2">
            <Label className="text-sm font-medium">Choose contracts</Label>
            <p className="text-xs text-muted-foreground">
              We'll scan these documents for {collaboratorName?.trim() || "this person"}'s split.
            </p>
            {filesQuery.isLoading ? (
              <div className="flex items-center gap-2 px-1 py-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading documents…
              </div>
            ) : documents.length === 0 ? (
              <div className="rounded-lg border border-dashed px-4 py-6 text-center text-sm text-muted-foreground">
                No documents are linked to this work yet.
              </div>
            ) : (
              <div className="flex max-h-[160px] flex-col gap-1 overflow-auto rounded-lg border border-border p-1.5">
                {documents.map((doc) => {
                  const checked = selectedContractIds.includes(doc.id);
                  return (
                    <label
                      key={doc.id}
                      className={cn(
                        "flex cursor-pointer items-center gap-2.5 rounded-md px-2 py-1.5 transition-colors hover:bg-muted/50",
                        checked && "bg-emerald-50 dark:bg-emerald-950/40"
                      )}
                    >
                      <Checkbox
                        checked={checked}
                        onCheckedChange={() => toggleContract(doc.id)}
                        className={cn(
                          checked &&
                            "border-emerald-600 bg-emerald-600 text-white data-[state=checked]:bg-emerald-600 data-[state=checked]:text-white"
                        )}
                      />
                      <span className="truncate text-sm font-medium text-foreground" title={doc.label}>
                        {doc.label}
                      </span>
                    </label>
                  );
                })}
              </div>
            )}
            <Button
              type="button"
              variant="outline"
              onClick={handleDerive}
              disabled={derive.isPending || selectedContractIds.length === 0}
              className="w-full"
            >
              {derive.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Sparkles className="mr-2 h-4 w-4" />
              )}
              {result ? "Re-run Derive" : "Derive"}
            </Button>
          </div>

          {/* Review state */}
          {result && (
            <div className="space-y-4 border-t border-border pt-4">
              {/* Confidence chip */}
              {isHighConfidence ? (
                <div className="inline-flex items-center gap-1.5 rounded-full bg-emerald-100 px-2.5 py-1 text-xs font-medium text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-300">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Found · high confidence
                </div>
              ) : (
                <div className="inline-flex items-center gap-1.5 rounded-full bg-amber-100 px-2.5 py-1 text-xs font-medium text-amber-800 dark:bg-amber-950/50 dark:text-amber-300">
                  <AlertTriangle className="h-3.5 w-3.5" />
                  {result.found ? "Low confidence" : "Not found"}
                </div>
              )}

              {/* Amber banner + escape hatches */}
              {showAmberBanner && (
                <div className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-3 dark:border-amber-900 dark:bg-amber-950/30">
                  <p className="text-sm text-amber-900 dark:text-amber-200">
                    We couldn't confidently match the contracts to this person.
                  </p>
                  <p className="mt-1 text-xs text-amber-800/90 dark:text-amber-300/80">
                    Try re-picking the contracts above and running Derive again, or edit the values
                    below by hand.
                  </p>
                  <div className="mt-2.5">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={handleEnterManually}
                      className="border-amber-300 text-amber-900 hover:bg-amber-100 dark:border-amber-800 dark:text-amber-200"
                    >
                      Enter manually
                    </Button>
                  </div>
                </div>
              )}

              {/* Editable split */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-sm font-medium">Master %</Label>
                  <Input
                    type="number"
                    min="0"
                    max="100"
                    step="0.01"
                    value={masterPct}
                    onChange={(e) => setMasterPct(e.target.value)}
                    placeholder="0"
                  />
                </div>
                <div>
                  <Label className="text-sm font-medium">Publishing %</Label>
                  <Input
                    type="number"
                    min="0"
                    max="100"
                    step="0.01"
                    value={publishingPct}
                    onChange={(e) => setPublishingPct(e.target.value)}
                    placeholder="0"
                  />
                </div>
              </div>

              {/* Editable terms */}
              <div className="space-y-2">
                <Label className="text-sm font-medium">Terms</Label>
                {terms.length === 0 && (
                  <p className="text-xs text-muted-foreground">
                    Add any deal terms for this person (e.g. "Advance", "$5,000").
                  </p>
                )}
                {terms.map((term, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <Input
                      value={term.label}
                      onChange={(e) =>
                        setTerms((prev) =>
                          prev.map((t, j) => (j === i ? { ...t, label: e.target.value } : t))
                        )
                      }
                      placeholder="Label"
                      className="flex-1"
                    />
                    <Input
                      value={term.value}
                      onChange={(e) =>
                        setTerms((prev) =>
                          prev.map((t, j) => (j === i ? { ...t, value: e.target.value } : t))
                        )
                      }
                      placeholder="Value"
                      className="flex-1"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      onClick={() => setTerms((prev) => prev.filter((_, j) => j !== i))}
                      aria-label="Remove term"
                      className="shrink-0"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setTerms((prev) => [...prev, { label: "", value: "" }])}
                >
                  <Plus className="mr-1.5 h-3.5 w-3.5" />
                  Add term
                </Button>
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button type="button" onClick={handleApply} disabled={!result}>
            Apply
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
