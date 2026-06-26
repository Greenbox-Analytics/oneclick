import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, FileText, Loader2, Plus, Sparkles, Upload, X } from "lucide-react";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { type DeriveResult, useDeriveFromContracts } from "@/hooks/useRegistry";
import { useWorkFiles } from "@/hooks/useWorkFiles";
import { useStorageStatus } from "@/hooks/useEntitlements";
import { useGatedAction } from "@/hooks/useGatedAction";
import { supabase } from "@/integrations/supabase/client";

interface TermRow {
  label: string;
  value: string;
}

interface Props {
  workId: string;
  projectId?: string;
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

interface ProjectContractOption {
  id: string;
  file_name: string;
}

export default function DeriveCollaboratorSplitDialog({
  workId,
  projectId,
  collaboratorName,
  open,
  onOpenChange,
  onApply,
}: Props) {
  const filesQuery = useWorkFiles(workId);
  const derive = useDeriveFromContracts();
  const queryClient = useQueryClient();
  const storageStatus = useStorageStatus();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // The work's linked documents, multi-select. Default = all selected.
  const documents = useMemo(
    () =>
      (filesQuery.data || [])
        .filter((wf) => wf.project_files)
        .map((wf) => ({ id: wf.project_files!.id, label: wf.project_files!.file_name })),
    [filesQuery.data]
  );

  // Contracts in the same project that aren't yet linked to this work. Lets
  // the user pull in a contract without leaving the dialog. Backend derive
  // accepts unlinked file_ids and auto-links matched ones afterwards.
  const linkedIds = useMemo(() => new Set(documents.map((d) => d.id)), [documents]);
  const projectContractsQuery = useQuery<ProjectContractOption[]>({
    queryKey: ["project-contracts-not-linked", projectId, Array.from(linkedIds).sort().join(",")],
    queryFn: async () => {
      if (!projectId) return [];
      const { data, error } = await supabase
        .from("project_files")
        .select("id, file_name")
        .eq("project_id", projectId)
        .eq("folder_category", "contract")
        .order("created_at", { ascending: false });
      if (error) throw error;
      return (data || []).filter((f) => !linkedIds.has(f.id)) as ProjectContractOption[];
    },
    enabled: open && !!projectId,
  });

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

  // Upload pipeline for the "Upload contract" button — mirrors the work-editor
  // dialog's pattern (SHA-256 dedup → storage upload → project_files insert).
  // On success the new file id gets added to selectedContractIds so the user
  // doesn't have to tick a separate checkbox; the backend will auto-link
  // matched contracts to the work after a successful derive.
  const { mutate: gatedUpload, isPending: isUploading, paywallElement } = useGatedAction<
    void,
    { file: File }
  >({
    mutationFn: async ({ file }) => {
      if (!projectId) {
        throw new Error("Cannot upload — work has no project.");
      }
      const hashBuffer = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
      const contentHash = Array.from(new Uint8Array(hashBuffer))
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");

      const { data: existing } = await supabase
        .from("project_files")
        .select("id")
        .eq("project_id", projectId)
        .eq("content_hash", contentHash)
        .limit(1);

      let fileId = existing && existing.length > 0 ? existing[0].id : null;
      let wasNew = false;

      if (!fileId) {
        const filePath = `${projectId}/contract/${Date.now()}_${file.name}`;
        const { error: uploadError } = await supabase.storage
          .from("project-files")
          .upload(filePath, file);
        if (uploadError) throw uploadError;
        const { data: urlData } = supabase.storage.from("project-files").getPublicUrl(filePath);
        const { data: insertedData, error: dbError } = await supabase
          .from("project_files")
          .insert({
            project_id: projectId,
            file_name: file.name,
            file_url: urlData.publicUrl,
            file_path: filePath,
            folder_category: "contract",
            file_size: file.size,
            file_type: file.type,
            content_hash: contentHash,
          })
          .select("id")
          .single();
        if (dbError) {
          await supabase.storage.from("project-files").remove([filePath]);
          throw dbError;
        }
        fileId = insertedData.id;
        wasNew = true;
      }

      setSelectedContractIds((prev) => (prev.includes(fileId!) ? prev : [...prev, fileId!]));
      queryClient.invalidateQueries({ queryKey: ["project-contracts-not-linked", projectId] });
      queryClient.invalidateQueries({ queryKey: ["project-files-tab", projectId] });
      toast.success(wasNew ? "Contract uploaded" : "Contract already in project — selected");
    },
    onError: (err) => {
      toast.error(err instanceof Error ? err.message : "Upload failed");
    },
  });

  const handleUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (storageStatus.cap !== -1 && storageStatus.used + file.size > storageStatus.cap) {
      toast.error("Uploading this file would exceed your storage cap.");
      return;
    }
    gatedUpload({ file });
  };

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
    // The backend auto-links matched contracts to the work — refresh the
    // Related Documents list so the new link shows up immediately.
    if (res.found && res.matched_file_ids?.length) {
      queryClient.invalidateQueries({ queryKey: ["work-files", workId] });
    }
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
              We'll scan the selected contracts for {collaboratorName?.trim() || "this person"}'s split.
            </p>
            {filesQuery.isLoading ? (
              <div className="flex items-center gap-2 px-1 py-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading documents…
              </div>
            ) : documents.length === 0 ? (
              <div className="rounded-lg border border-dashed px-4 py-3 text-xs text-muted-foreground">
                No contracts are linked to this work yet — pick one from the project below or upload a new contract.
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

            {/* Other contracts in this project — lets the user pull in a
                contract that lives in the project but isn't yet on the work. */}
            {projectId && (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label className="text-xs font-medium text-muted-foreground">
                    Other contracts in this project
                  </Label>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                    className="h-7 px-2 text-xs"
                  >
                    {isUploading ? (
                      <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Upload className="mr-1.5 h-3.5 w-3.5" />
                    )}
                    Upload contract
                  </Button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    onChange={handleUpload}
                  />
                </div>
                {projectContractsQuery.isLoading ? (
                  <div className="flex items-center gap-2 px-1 py-2 text-xs text-muted-foreground">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading project contracts…
                  </div>
                ) : (projectContractsQuery.data || []).length === 0 ? (
                  <div className="rounded-md border border-dashed px-3 py-2 text-xs text-muted-foreground">
                    No other contracts in this project. Upload one above.
                  </div>
                ) : (
                  <div className="flex max-h-[140px] flex-col gap-1 overflow-auto rounded-lg border border-border p-1.5">
                    {(projectContractsQuery.data || []).map((doc) => {
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
                          <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                          <span className="truncate text-sm font-medium text-foreground" title={doc.file_name}>
                            {doc.file_name}
                          </span>
                        </label>
                      );
                    })}
                  </div>
                )}
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
        {paywallElement}
      </DialogContent>
    </Dialog>
  );
}
