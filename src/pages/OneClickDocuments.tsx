import { useState, useEffect, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { HeaderDocsButton } from "@/components/layout/HeaderDocsButton";
import { Input } from "@/components/ui/input";
import { ArrowLeft, Loader2, AlertCircle } from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";
import { PageHeader } from "@/components/layout/PageHeader";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { useSmartBack } from "@/hooks/useSmartBack";
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
import { API_URL, apiFetch, getAuthHeaders, ApiError, apiErrorFromBody } from "@/lib/apiFetch";
import { parseCreditWallDetail, UnlinkProjectHint } from "@/components/paywall/creditWall";
import ContractSelector from "@/components/oneclick/ContractSelector";
import RoyaltyStatementSelector from "@/components/oneclick/RoyaltyStatementSelector";
import CalculationResults from "@/components/oneclick/CalculationResults";
import ExpenseReviewDialog from "@/components/oneclick/ExpenseReviewDialog";
import SongMismatchComparison from "@/components/oneclick/SongMismatchComparison";
import {
  ConflictResolutionDialog,
  normalizeConflictPayload,
  type Conflict,
  type ConflictGatePayload,
  type ConflictResolution,
} from "@/components/oneclick/ConflictResolutionDialog";
import {
  RevisionPromptDialog,
  type RevisionCandidate,
  type RevisionDecision,
} from "@/components/oneclick/RevisionPromptDialog";

interface RoyaltyPayment { song_title: string; party_name: string; role: string; royalty_type: string; percentage: number; total_royalty: number; amount_to_pay: number; terms?: string; basis?: string; gross_amount?: number; expenses_applied?: number; net_amount?: number; }
interface ReviewExpense { id: string; description?: string; amount: number; category?: string | null; incurred_on?: string | null; work_ids?: string[]; work_titles?: string[]; }
interface SplitFinding {
  party_name: string;
  royalty_type: string;
  extracted_percentage: number;
  extracted_basis?: string | null;
  verdict: "verified" | "mismatch" | "unverified";
  contract_percentage?: number | null;
  contract_basis?: string | null;
  contract_quote?: string;
  note?: string;
}
interface SplitReview { overall: "verified" | "needs_review" | "unavailable"; checked: number; flagged: number; findings: SplitFinding[]; }
interface CalculationResult { status: string; total_payments: number; payments: RoyaltyPayment[]; excel_file_url?: string; message: string; is_cached?: boolean; calculation_id?: string; expense_review_required?: boolean; expenses?: ReviewExpense[]; review?: SplitReview | null; }
interface CalculationErrorState {
  message: string;
  code?: string;
  suggestion?: string;
  details?: {
    contract_works?: string[];
    statement_songs?: string[];
    statement_song_total_count?: number;
    available_columns?: string[];
    looking_for?: string[];
    excluded_payor_count?: number;
  };
  /** Licensing Phase B (plan Task 13) — set when the calculation was denied by
   * a credit-402 whose seat wallet is managed by an organization: there's no
   * upgrade/pay-per-use path on a seat, so the alert offers a "Request
   * credits" link (to `requestUrl`, or /organization) instead. */
  managedByOrg?: boolean;
  requestUrl?: string;
  /** Licensing Phase C (spec §6/§11 rule 11c, plan Task 8) — set when the
   * dry-seat wall is on a project the caller OWNS and can unlink. Lands on
   * the 402 in Task 6 (running separately); rendered behind a presence-check
   * until then, alongside (never instead of) the "Request credits" link. */
  ownerCanUnlink?: boolean;
  projectId?: string;
  projectName?: string;
}
interface Project { id: string; name: string; }
interface ArtistFile { id: string; file_name: string; created_at: string; folder_category: string; file_path: string; project_id: string; }
interface Artist { id: string; name: string; }

type ConfirmGate = "conflict" | "revision" | "superseded";

/**
 * Thrown by postConfirm() when /oneclick/confirm responds 409 with a
 * `{gate, payload}` body — a contract conflict or a possible statement
 * revision the user needs to resolve before the calculation can be saved.
 */
class ConfirmGateError extends Error {
  gate: ConfirmGate;
  payload: unknown;
  constructor(gate: ConfirmGate, payload: unknown) {
    super(`oneclick/confirm needs resolution: ${gate}`);
    this.gate = gate;
    this.payload = payload;
  }
}

/**
 * POSTs to /oneclick/confirm with a raw fetch (rather than apiFetch) so a 409
 * gate response's structured `{gate, payload}` body can be inspected directly
 * instead of being collapsed into a plain error message.
 */
const postConfirm = async (body: Record<string, unknown>): Promise<{ id?: string }> => {
  const authHeaders = await getAuthHeaders();
  const res = await fetch(`${API_URL}/oneclick/confirm`, {
    method: "POST",
    headers: { ...authHeaders, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const json = await res.json().catch(() => ({}));
  if (res.status === 409 && json?.detail?.gate) {
    throw new ConfirmGateError(json.detail.gate, json.detail.payload);
  }
  if (!res.ok) {
    throw new Error(typeof json?.detail === "string" ? json.detail : `Request failed: ${res.status}`);
  }
  return json;
};

const OneClickDocuments = () => {
  const navigate = useNavigate();
  const goBack = useSmartBack("/tools/oneclick");
  const { artistId } = useParams<{ artistId: string }>();
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const invalidateRoyalties = () => {
    ["royalty-payees", "royalty-periods", "royalty-payouts"].forEach((k) =>
      queryClient.invalidateQueries({ queryKey: [k] }),
    );
  };
  const [contractFiles, setContractFiles] = useState<File[]>([]);
  const [royaltyStatementFile, setRoyaltyStatementFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [calculationResult, setCalculationResult] = useState<CalculationResult | null>(null);
  const [expenseReview, setExpenseReview] = useState<ReviewExpense[] | null>(null);
  const [isRecalculating, setIsRecalculating] = useState(false);
  const [error, setError] = useState<CalculationErrorState | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  // Set once the calculation is saved (or read off a cached result); the
  // Earnings Breakdown tab needs it to fetch per-dimension aggregates.
  const [savedCalculationId, setSavedCalculationId] = useState<string | null>(null);
  const [selectedExistingContracts, setSelectedExistingContracts] = useState<string[]>([]);
  const [selectedExistingRoyaltyStatement, setSelectedExistingRoyaltyStatement] = useState<string | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [existingContracts, setExistingContracts] = useState<ArtistFile[]>([]);
  const [existingRoyaltyStatements, setExistingRoyaltyStatements] = useState<ArtistFile[]>([]);
  const [projectFilesById, setProjectFilesById] = useState<Record<string, ArtistFile[]>>({});
  const [selectedContractProject, setSelectedContractProject] = useState<string | null>(null);
  const [selectedRoyaltyStatementProject, setSelectedRoyaltyStatementProject] = useState<string | null>(null);
  const [newContractProjectId, setNewContractProjectId] = useState<string>("");
  const [contractTabValue, setContractTabValue] = useState<string>("upload");
  const [royaltyStatementTabValue, setRoyaltyStatementTabValue] = useState<string>("upload");
  const [newRoyaltyStatementProjectId, setNewRoyaltyStatementProjectId] = useState<string>("");
  const [royaltyStatementCurrency, setRoyaltyStatementCurrency] = useState<string>("USD");
  // Artist name fetched via React Query so the result is cached across
  // remounts. Without this, navigating away from /tools/oneclick/.../documents
  // and back would re-fetch and (worse) re-trigger a full-page loader that
  // hid any in-progress calculation UI. Cache key includes artistId so a
  // different artist re-fetches correctly.
  const artistQuery = useQuery({
    queryKey: ["artist-name", artistId],
    queryFn: async (): Promise<string> => {
      const data = await apiFetch<Artist[]>(`${API_URL}/artists`);
      return data.find((a) => a.id === artistId)?.name ?? "";
    },
    enabled: !!artistId && !!user,
    staleTime: 5 * 60_000, // 5 min — artist names rarely change mid-session
  });
  const artistName = artistQuery.data ?? "";
  const isLoadingArtist = artistQuery.isLoading;
  const [isLoadingProjectFiles, setIsLoadingProjectFiles] = useState(false);
  const [calculationProgress, setCalculationProgress] = useState(0);
  const [calculationStage, setCalculationStage] = useState("");
  const [calculationMessage, setCalculationMessage] = useState("");
  const [showProgressModal, setShowProgressModal] = useState(false);
  const [contractSearchTerm, setContractSearchTerm] = useState("");
  const [royaltySearchTerm, setRoyaltySearchTerm] = useState("");
  const [isCreateProjectOpen, setIsCreateProjectOpen] = useState(false);
  const [newProjectNameInput, setNewProjectNameInput] = useState("");
  const [isCreatingProject, setIsCreatingProject] = useState(false);
  const [showReviewDialog, setShowReviewDialog] = useState(false);
  const [pendingForceRecalculate, setPendingForceRecalculate] = useState(false);
  // The exact confirm request body last sent — a gate retry re-sends this
  // plus conflict_resolutions / revision_decision, per the backend contract.
  const [pendingConfirmBody, setPendingConfirmBody] = useState<Record<string, unknown> | null>(null);
  const [conflictGateItems, setConflictGateItems] = useState<Conflict[] | null>(null);
  const [revisionGateCandidates, setRevisionGateCandidates] = useState<RevisionCandidate[] | null>(null);
  const autoSaveTriggeredRef = useRef<string | null>(null);
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(TOOL_CONFIGS.oneclick, {
    onComplete: () => markToolCompleted("oneclick"),
  });

  useEffect(() => {
    if (!onboardingLoading && !statuses.oneclick && walkthrough.phase === "idle") {
      const timer = setTimeout(() => walkthrough.startModal(), 500);
      return () => clearTimeout(timer);
    }
  }, [onboardingLoading, statuses.oneclick]);

  const normalizeFileName = (name: string) => name.trim().toLowerCase();

  const showPersistentDuplicateToast = (message: string) => {
    toast.error(message, {
      duration: Infinity, closeButton: true,
      style: { background: "hsl(var(--destructive))", color: "hsl(var(--destructive-foreground))", border: "1px solid hsl(var(--destructive))" },
    });
  };

  const findDuplicateFileNames = (files: File[]) => {
    const seen = new Set<string>();
    const duplicates = new Set<string>();
    for (const file of files) {
      const normalized = normalizeFileName(file.name);
      if (seen.has(normalized)) duplicates.add(file.name);
      else seen.add(normalized);
    }
    return Array.from(duplicates);
  };

  const fetchProjectFilesForValidation = async (projectId: string): Promise<ArtistFile[]> => {
    if (projectFilesById[projectId]) {
      return projectFilesById[projectId];
    }

    const data = await apiFetch<ArtistFile[]>(`${API_URL}/files/${projectId}`);
    setProjectFilesById(prev => ({ ...prev, [projectId]: data }));
    return data;
  };

  // (artist fetch moved to the useQuery above so its result persists across
  // remounts. Navigating away during a calculation and back no longer triggers
  // a full-page reload of the artist name.)

  useEffect(() => {
    if (artistId) {
      apiFetch<Project[]>(`${API_URL}/projects/${artistId}`)
        .then(data => setProjects(data))
        .catch(err => console.error("Error fetching projects:", err));
    }
  }, [artistId]);

  useEffect(() => {
    if (selectedContractProject) {
      setSelectedExistingContracts([]);
      setIsLoadingProjectFiles(true);
      apiFetch<ArtistFile[]>(`${API_URL}/files/${selectedContractProject}`)
        .then((data: ArtistFile[]) => {
            setProjectFilesById(prev => ({ ...prev, [selectedContractProject]: data }));
            setExistingContracts(data.filter(f => f.folder_category === 'contract' || f.folder_category === 'split_sheet'));
            setIsLoadingProjectFiles(false);
        })
        .catch(err => { console.error("Error fetching contract files:", err); setIsLoadingProjectFiles(false); });
    } else {
        setExistingContracts([]);
        setSelectedExistingContracts([]);
    }
  }, [selectedContractProject]);

  useEffect(() => {
    if (selectedRoyaltyStatementProject) {
      setSelectedExistingRoyaltyStatement(null);
      setIsLoadingProjectFiles(true);
      apiFetch<ArtistFile[]>(`${API_URL}/files/${selectedRoyaltyStatementProject}`)
        .then((data: ArtistFile[]) => {
            setProjectFilesById(prev => ({ ...prev, [selectedRoyaltyStatementProject]: data }));
            setExistingRoyaltyStatements(data.filter(f => f.folder_category === 'royalty_statement'));
            setIsLoadingProjectFiles(false);
        })
        .catch(err => { console.error("Error fetching royalty statement files:", err); setIsLoadingProjectFiles(false); });
    } else {
        setExistingRoyaltyStatements([]);
        setSelectedExistingRoyaltyStatement(null);
    }
  }, [selectedRoyaltyStatementProject]);

  const handleCreateProject = async () => {
    if (!artistId || !newProjectNameInput.trim()) return;
    setIsCreatingProject(true);
    try {
        const newProject = await apiFetch<Project>(`${API_URL}/projects`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ artist_id: artistId, name: newProjectNameInput })
        });
        setProjects([newProject, ...projects]);
        setNewContractProjectId(newProject.id);
        setNewRoyaltyStatementProjectId(newProject.id);
        setNewProjectNameInput(""); setIsCreateProjectOpen(false);
        toast.success("Project created successfully");
    } catch (err) {
        console.error("Error creating project:", err);
        toast.error(err instanceof ApiError ? err.message : "Failed to create project");
    }
    finally { setIsCreatingProject(false); }
  };

  const handleCalculateRoyalties = async (forceRecalculate = false) => {
    if (!artistId || !user) { toast.error("User or Artist not found."); return; }
    const hasContracts = contractFiles.length > 0 || selectedExistingContracts.length > 0;
    const hasRoyaltyStatement = royaltyStatementFile !== null || selectedExistingRoyaltyStatement !== null;
    if (!hasContracts || !hasRoyaltyStatement) {
        toast.error("Please provide both contracts/split sheets and a royalty statement."); return;
    }
    setIsUploading(true);
    setCalculationResult(null);
    setError(null);
    try {
        let finalContractIds: string[] = [];
        let finalStatementId = "";
        let finalProjectId = "";

        if (contractFiles.length > 0) {
            if (!newContractProjectId || newContractProjectId === "none") {
                 toast.error("Please select a project to save the new document.");
                 throw new Error("You must select a project to save the new document to.");
             }

           const duplicateNamesInContractSelection = findDuplicateFileNames(contractFiles);
           if (duplicateNamesInContractSelection.length > 0) {
             showPersistentDuplicateToast(`Duplicate file name(s) blocked: ${duplicateNamesInContractSelection.join(", ")}`);
             throw new Error("Please remove duplicate contract file names before uploading.");
           }

           const projectFiles = await fetchProjectFilesForValidation(newContractProjectId);
           const projectFileNames = new Set(projectFiles.map(existing => normalizeFileName(existing.file_name)));
           const contractDuplicatesInProject = contractFiles
             .filter(file => projectFileNames.has(normalizeFileName(file.name)))
             .map(file => file.name);

           if (contractDuplicatesInProject.length > 0) {
             showPersistentDuplicateToast(`These file name(s) already exist in this project: ${contractDuplicatesInProject.join(", ")}`);
             throw new Error("Duplicate file names found in selected project.");
           }

             // Upload each file sequentially to ensure order and error handling
             for (const file of contractFiles) {
                 const formData = new FormData();
                 formData.append("file", file);
                 formData.append("project_id", newContractProjectId);

                 const uploadAuthHeaders = await getAuthHeaders();
                 const uploadRes = await fetch(`${API_URL}/contracts/upload`, {
                     method: "POST",
                     headers: uploadAuthHeaders,
                     body: formData
                 });

                 if (!uploadRes.ok) {
                     const errData = await uploadRes.json();
                     throw new Error(errData.detail || `Failed to upload and process contract: ${file.name}`);
                 }

                 const uploadData = await uploadRes.json();
                 finalContractIds.push(uploadData.contract_id);
             }

             finalProjectId = newContractProjectId;
             toast.success(`${contractFiles.length} contract(s) uploaded and processed successfully!`);
        }
        if (selectedExistingContracts.length > 0) {
            finalContractIds = [...finalContractIds, ...selectedExistingContracts];
            if (!finalProjectId) {
                const contract = existingContracts.find(c => c.id === selectedExistingContracts[0]);
                if (contract) finalProjectId = contract.project_id;
            }
        }

        if (royaltyStatementFile) {
             const formData = new FormData();
             formData.append("file", royaltyStatementFile);
             formData.append("artist_id", artistId);
             formData.append("category", "royalty_statement");
             formData.append("currency", royaltyStatementCurrency);
             const targetProjectId = newRoyaltyStatementProjectId || finalProjectId;
             if (targetProjectId) formData.append("project_id", targetProjectId);

           if (targetProjectId) {
             const projectFiles = await fetchProjectFilesForValidation(targetProjectId);
             const projectFileNames = new Set(projectFiles.map(existing => normalizeFileName(existing.file_name)));
             const targetFileName = royaltyStatementFile.name;

             if (projectFileNames.has(normalizeFileName(targetFileName))) {
               showPersistentDuplicateToast(`A file named "${targetFileName}" already exists in this project.`);
               throw new Error("Duplicate file name found in selected project.");
             }
           }

             const stmtAuthHeaders = await getAuthHeaders();
             const uploadRes = await fetch(`${API_URL}/upload`, {
                 method: "POST",
                 headers: stmtAuthHeaders,
                 body: formData
             });

           if (!uploadRes.ok) {
             const errData = await uploadRes.json();
             throw new Error(errData.detail || "Failed to upload royalty statement");
           }
             const uploadData = await uploadRes.json();
             finalStatementId = uploadData.file.id;

        } else if (selectedExistingRoyaltyStatement) {
            finalStatementId = selectedExistingRoyaltyStatement;
        }

        if (finalContractIds.length === 0 || !finalStatementId) {
            throw new Error("Could not determine contracts or statement to process.");
        }

        setLastCalculationContext({ contractIds: finalContractIds, statementId: finalStatementId, projectId: finalProjectId });
        setShowProgressModal(true);
        setCalculationProgress(0);
        setCalculationStage("starting");
        setCalculationMessage("Starting calculation...");
        setSaveSuccess(false);
        // Clear any prior calculation's id so the breakdown tab doesn't show
        // stale data before this run is saved (or a cached id arrives).
        setSavedCalculationId(null);
        // Reset the auto-save guard so this run saves even when the inputs match
        // the previous one (e.g. force-recalculate of the same statement +
        // contracts) — otherwise the new calculation_id is never captured and the
        // breakdown is stuck on "Save the calculation to see the earnings breakdown".
        autoSaveTriggeredRef.current = null;

        // Use fetch + SSE for authenticated streaming
        const queryParams = new URLSearchParams({
            project_id: finalProjectId,
            royalty_statement_file_id: finalStatementId
        });
        if (forceRecalculate) queryParams.append("force_recalculate", "true");
        finalContractIds.forEach(id => queryParams.append("contract_ids", id));

        const authHeaders = await getAuthHeaders();
        const response = await fetch(`${API_URL}/oneclick/calculate-royalties-stream?` + queryParams.toString(), {
            headers: { ...authHeaders },
        });

        if (!response.ok) {
            const body = await response.json().catch(() => ({}));
            // Structured 402s carry `detail` (reason + org-seat managedByOrg/
            // requestUrl/…), preserved on the ApiError so the catch block can
            // offer the "Request credits" CTA instead of a dead-end message.
            throw apiErrorFromBody(body, response.status, `Calculation request failed: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response stream available");
        const decoder = new TextDecoder();

        const timeout = setTimeout(() => {
            reader.cancel(); setShowProgressModal(false);
            setError({ message: "Calculation timed out. Please try again." }); toast.error("Calculation timed out"); setIsUploading(false);
        }, 120000);

        let buffer = "";
        // Populated by the 'complete' branch below; the 'needs_confirmation'
        // event (if it follows) carries no results of its own, so this is
        // what's used to build the confirm retry body for it.
        let streamResult: CalculationResult | null = null;
        const processLine = (line: string) => {
            if (!line.startsWith("data: ")) return;
            try {
                const data = JSON.parse(line.slice(6));
                if (data.type === 'status') {
                    setCalculationProgress(data.progress || 0); setCalculationStage(data.stage || ""); setCalculationMessage(data.message || "");
                } else if (data.type === 'complete' || (data.status === 'success' && data.payments)) {
                    clearTimeout(timeout);
                    const needsReview = !!data.expense_review_required && !data.is_cached;
                    const result: CalculationResult = { status: data.status, total_payments: data.total_payments, payments: data.payments, message: data.message, is_cached: data.is_cached, calculation_id: data.calculation_id, expense_review_required: needsReview, expenses: data.expenses || [], review: data.review ?? null };
                    streamResult = result;
                    setCalculationResult(result);
                    setShowProgressModal(false);
                    if (needsReview) {
                        setExpenseReview(data.expenses || []);
                        toast.info("Some collaborators are paid on net income — review the expenses to finalize.");
                    } else {
                        toast.success(data.is_cached ? "Royalties loaded successfully!" : "Royalties calculated successfully!");
                    }
                    setContractFiles([]); setRoyaltyStatementFile(null); setIsUploading(false);
                } else if (data.type === 'needs_confirmation') {
                    clearTimeout(timeout);
                    setShowProgressModal(false); setIsUploading(false);
                    if (!streamResult) {
                        console.error("needs_confirmation event arrived without a prior result to attach it to");
                        toast.error("Couldn't finish this calculation. Please try again.");
                        return;
                    }
                    routeConfirmGate(data.gate, data.payload, {
                        contract_ids: finalContractIds,
                        royalty_statement_id: finalStatementId,
                        project_id: finalProjectId,
                        results: streamResult,
                    });
                } else if (data.type === 'error') {
                    clearTimeout(timeout); setShowProgressModal(false);
                    setError({
                        message: data.message || "An error occurred",
                        code: data.error_code,
                        suggestion: data.suggestion,
                        details: data.details,
                    });
                    toast.error(data.message || "Calculation failed");
                    setIsUploading(false);
                }
            } catch (err) { console.error("Error parsing SSE data:", err); }
        };

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop() || "";
                for (const line of lines) processLine(line.trim());
            }
            if (buffer.trim()) processLine(buffer.trim());
        } catch {
            clearTimeout(timeout); setShowProgressModal(false);
            setError({ message: "Connection error. Please try again." }); toast.error("Connection error during calculation"); setIsUploading(false);
        }

    } catch (error: unknown) {
        console.error("Error:", error);
        const errorMessage = error instanceof Error ? error.message : "An error occurred.";
        const cw = parseCreditWallDetail(error instanceof ApiError ? error.detail : undefined);
        setError({ message: errorMessage, ...cw });
        if (!String(errorMessage).toLowerCase().includes("duplicate file")) toast.error(errorMessage || "An error occurred during processing.");
        setShowProgressModal(false); setIsUploading(false);
    }
  };

  const [lastCalculationContext, setLastCalculationContext] = useState<{
      contractIds: string[], statementId: string, projectId: string
  } | null>(null);

  // Sends a confirm body to the backend; on success stores the calculation id
  // and clears any open gate dialogs, on a gate response opens the matching
  // dialog instead of failing, on any other error shows the generic toast.
  const submitConfirm = async (body: Record<string, unknown>) => {
      setIsSaving(true);
      try {
          const saved = await postConfirm(body);
          if (saved?.id) setSavedCalculationId(saved.id);
          setSaveSuccess(true);
          setConflictGateItems(null);
          setRevisionGateCandidates(null);
          setPendingConfirmBody(null);
          invalidateRoyalties();
      } catch (err) {
          if (err instanceof ConfirmGateError) {
              routeConfirmGate(err.gate, err.payload, body);
              return;
          }
          console.error("Error saving results:", err);
          toast.error("Failed to save results");
      }
      finally { setIsSaving(false); }
  };

  // Opens the dialog matching a gate (or, for 'superseded', just toasts) and
  // remembers the body that produced it so a resolved retry can re-send it.
  const routeConfirmGate = (gate: ConfirmGate, payload: unknown, body: Record<string, unknown>) => {
      setPendingConfirmBody(body);
      if (gate === "conflict") {
          setConflictGateItems(normalizeConflictPayload(payload as ConflictGatePayload));
          setRevisionGateCandidates(null);
      } else if (gate === "revision") {
          setRevisionGateCandidates((payload as { candidates: RevisionCandidate[] }).candidates);
          setConflictGateItems(null);
      } else {
          setConflictGateItems(null);
          setRevisionGateCandidates(null);
          toast.error("This statement was replaced by a newer file — run that one instead.");
      }
  };

  const handleResolveConflicts = (resolutions: ConflictResolution[]) => {
      if (!pendingConfirmBody) return;
      setConflictGateItems(null);
      submitConfirm({ ...pendingConfirmBody, conflict_resolutions: resolutions });
  };

  const handleDecideRevision = (decision: RevisionDecision) => {
      if (!pendingConfirmBody) return;
      setRevisionGateCandidates(null);
      submitConfirm({ ...pendingConfirmBody, revision_decision: decision });
  };

  const handleCancelConfirmGate = () => {
      setConflictGateItems(null);
      setRevisionGateCandidates(null);
      setPendingConfirmBody(null);
  };

  const handleConfirmResultsWithContext = async () => {
      if (!lastCalculationContext || !calculationResult || !user) return;
      await submitConfirm({
          contract_ids: lastCalculationContext.contractIds,
          royalty_statement_id: lastCalculationContext.statementId,
          project_id: lastCalculationContext.projectId,
          results: calculationResult,
      });
  };

  useEffect(() => {
      if (!calculationResult || !lastCalculationContext || !user) return;
      if (calculationResult.is_cached) {
        invalidateRoyalties();
        return;
      }
      // Hold off on saving until the user confirms expenses for net-basis rows.
      if (calculationResult.expense_review_required) return;
      const resultKey = `${lastCalculationContext.statementId}-${lastCalculationContext.contractIds.join(',')}`;
      if (autoSaveTriggeredRef.current === resultKey) return;
      autoSaveTriggeredRef.current = resultKey;
      handleConfirmResultsWithContext();
  }, [calculationResult, lastCalculationContext, user]);

  // Recompute net payouts after the user confirms/edits expenses in the review modal.
  const handleConfirmExpenses = async (editedExpenses: ReviewExpense[]) => {
      if (!calculationResult) return;
      setIsRecalculating(true);
      try {
          const resp = await apiFetch<{ payments: RoyaltyPayment[] }>(`${API_URL}/oneclick/recalculate-net`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                  payments: calculationResult.payments,
                  expenses: editedExpenses.map(e => ({ amount: e.amount, work_titles: e.work_titles || [] })),
              }),
          });
          setCalculationResult({ ...calculationResult, payments: resp.payments, expense_review_required: false });
          setExpenseReview(null);
          toast.success("Net royalties finalized");
      } catch (err) {
          toast.error(`Recalculation failed: ${(err as Error).message}`);
      } finally {
          setIsRecalculating(false);
      }
  };

  const openReviewDialog = (forceRecalculate: boolean) => {
    const hasContracts = contractFiles.length > 0 || selectedExistingContracts.length > 0;
    const hasRoyaltyStatement = royaltyStatementFile !== null || selectedExistingRoyaltyStatement !== null;
    if (!hasContracts || !hasRoyaltyStatement) {
      toast.error("Please provide both contracts/split sheets and a royalty statement.");
      return;
    }
    setPendingForceRecalculate(forceRecalculate);
    setShowReviewDialog(true);
  };

  const handleConfirmReview = () => {
    setShowReviewDialog(false);
    handleCalculateRoyalties(pendingForceRecalculate);
  };

  const reviewContractNames: string[] = [
    ...contractFiles.map(f => f.name),
    ...selectedExistingContracts
      .map(id => existingContracts.find(c => c.id === id)?.file_name)
      .filter((n): n is string => Boolean(n)),
  ];

  const reviewStatementName: string | null =
    royaltyStatementFile?.name ??
    (selectedExistingRoyaltyStatement
      ? existingRoyaltyStatements.find(s => s.id === selectedExistingRoyaltyStatement)?.file_name ?? null
      : null);

  // (Removed the full-page "Loading artist information..." gate that blocked
  // the whole route while the artist name was in-flight. It hid in-progress
  // calculation UI on every remount. Render the page content immediately; the
  // header shows an inline skeleton in place of the name while the React
  // Query fetch resolves.)

  return (
    <div className="min-h-screen bg-background">
      <PageHeader
        showBack={false}
        actions={
          <>
            <HeaderDocsButton />
            <ToolHelpButton onClick={walkthrough.replay} />
            <Button variant="outline" className="hidden md:inline-flex" onClick={goBack}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
          </>
        }
      />

      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-foreground mb-2">OneClick Royalty Calculator</h2>
          <p className="text-muted-foreground flex items-center gap-2">
            Upload documents for{" "}
            {isLoadingArtist ? (
              <span className="inline-block h-4 w-32 rounded bg-muted animate-pulse" aria-label="Loading artist name" />
            ) : (
              <span>{artistName || "Artist"}</span>
            )}
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <ContractSelector
            contractTabValue={contractTabValue}
            setContractTabValue={setContractTabValue}
            contractFiles={contractFiles}
            setContractFiles={setContractFiles}
            newContractProjectId={newContractProjectId}
            setNewContractProjectId={setNewContractProjectId}
            projects={projects}
            setIsCreateProjectOpen={setIsCreateProjectOpen}
            selectedContractProject={selectedContractProject}
            setSelectedContractProject={setSelectedContractProject}
            contractSearchTerm={contractSearchTerm}
            setContractSearchTerm={setContractSearchTerm}
            isLoadingProjectFiles={isLoadingProjectFiles}
            existingContracts={existingContracts}
            selectedExistingContracts={selectedExistingContracts}
            setSelectedExistingContracts={setSelectedExistingContracts}
            fetchProjectFilesForValidation={fetchProjectFilesForValidation}
          />

          <RoyaltyStatementSelector
            royaltyStatementTabValue={royaltyStatementTabValue}
            setRoyaltyStatementTabValue={setRoyaltyStatementTabValue}
            royaltyStatementFile={royaltyStatementFile}
            setRoyaltyStatementFile={setRoyaltyStatementFile}
            newRoyaltyStatementProjectId={newRoyaltyStatementProjectId}
            setNewRoyaltyStatementProjectId={setNewRoyaltyStatementProjectId}
            projects={projects}
            setIsCreateProjectOpen={setIsCreateProjectOpen}
            selectedRoyaltyStatementProject={selectedRoyaltyStatementProject}
            setSelectedRoyaltyStatementProject={setSelectedRoyaltyStatementProject}
            royaltySearchTerm={royaltySearchTerm}
            setRoyaltySearchTerm={setRoyaltySearchTerm}
            isLoadingProjectFiles={isLoadingProjectFiles}
            existingRoyaltyStatements={existingRoyaltyStatements}
            selectedExistingRoyaltyStatement={selectedExistingRoyaltyStatement}
            setSelectedExistingRoyaltyStatement={setSelectedExistingRoyaltyStatement}
            fetchProjectFilesForValidation={fetchProjectFilesForValidation}
            currency={royaltyStatementCurrency}
            onCurrencyChange={setRoyaltyStatementCurrency}
          />
        </div>

        <div className="flex gap-3 justify-center mb-8">
          <Button
            data-walkthrough="oneclick-calculate"
            onClick={() => openReviewDialog(false)}
            disabled={((contractFiles.length === 0 && selectedExistingContracts.length === 0) ||
                       (royaltyStatementFile === null && selectedExistingRoyaltyStatement === null)) ||
                       isUploading}
            size="lg"
            className="w-full max-w-sm"
          >
            {isUploading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Calculating...
              </>
            ) : (
              "Calculate Royalties"
            )}
          </Button>
        </div>

        {/* Error Alert */}
        {error && (
            <Alert variant="destructive" className="mb-6">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>{error.message}</AlertTitle>
                <AlertDescription>
                    {error.suggestion && <p className="mt-2">{error.suggestion}</p>}
                    {error.managedByOrg && (
                        <Button
                            size="sm"
                            variant="outline"
                            className="mt-3"
                            onClick={() => navigate(error.requestUrl || "/organization")}
                        >
                            Request credits
                        </Button>
                    )}
                    {error.managedByOrg && error.ownerCanUnlink && (
                        <UnlinkProjectHint
                            projectId={error.projectId}
                            projectName={error.projectName}
                            className="mt-2"
                        />
                    )}
                    {error.code === 'NO_SONG_MATCHES' && error.details?.contract_works && error.details?.statement_songs && (
                        <SongMismatchComparison
                            contractWorks={error.details.contract_works}
                            statementSongs={error.details.statement_songs}
                            statementTotalCount={error.details.statement_song_total_count ?? error.details.statement_songs.length}
                        />
                    )}
                    {error.code === 'STATEMENT_COLUMNS_UNDETECTABLE' && error.details?.available_columns && (
                        <p className="mt-3 text-sm">
                            <span className="font-medium">Columns we found in your statement:</span>{" "}
                            {error.details.available_columns.join(", ")}
                        </p>
                    )}
                </AlertDescription>
            </Alert>
        )}

        <CalculationResults
          showProgressModal={showProgressModal}
          calculationProgress={calculationProgress}
          calculationStage={calculationStage}
          calculationMessage={calculationMessage}
          calculationResult={calculationResult}
          isUploading={isUploading}
          handleCalculateRoyalties={openReviewDialog}
          calculationId={savedCalculationId}
        />

        <ExpenseReviewDialog
          open={expenseReview !== null}
          expenses={expenseReview ?? []}
          isSubmitting={isRecalculating}
          projectId={lastCalculationContext?.projectId}
          onConfirm={handleConfirmExpenses}
          onCancel={() => setExpenseReview(null)}
        />

        <ConflictResolutionDialog
          open={conflictGateItems !== null}
          conflicts={conflictGateItems ?? []}
          isSubmitting={isSaving}
          onResolve={handleResolveConflicts}
          onCancel={handleCancelConfirmGate}
        />

        <RevisionPromptDialog
          open={revisionGateCandidates !== null}
          candidates={revisionGateCandidates ?? []}
          isSubmitting={isSaving}
          onDecide={handleDecideRevision}
          onCancel={handleCancelConfirmGate}
        />

        {/* Review Selection Dialog */}
        <Dialog open={showReviewDialog} onOpenChange={setShowReviewDialog}>
          <DialogContent className="sm:max-w-[480px]">
            <DialogHeader>
              <DialogTitle>Review your selection</DialogTitle>
              <DialogDescription>
                Confirm the documents that will be used for this calculation. Stale selections from a previous run will be included if you don't remove them.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-2">
              <div className="space-y-2">
                <p className="text-sm font-semibold text-foreground">
                  Contracts ({reviewContractNames.length})
                </p>
                {reviewContractNames.length > 0 ? (
                  <ul className="space-y-1 max-h-48 overflow-y-auto rounded-md border border-border bg-secondary/30 p-2">
                    {reviewContractNames.map((name, i) => (
                      <li key={i} className="text-sm text-foreground truncate">• {name}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-muted-foreground">No contracts selected.</p>
                )}
              </div>
              <div className="space-y-2">
                <p className="text-sm font-semibold text-foreground">Royalty Statement</p>
                {reviewStatementName ? (
                  <div className="rounded-md border border-border bg-secondary/30 p-2 text-sm text-foreground truncate">
                    • {reviewStatementName}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No statement selected.</p>
                )}
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowReviewDialog(false)}>
                Cancel
              </Button>
              <Button
                onClick={handleConfirmReview}
                disabled={reviewContractNames.length === 0 || !reviewStatementName}
              >
                Confirm & Calculate
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Create Project Dialog */}
        <Dialog open={isCreateProjectOpen} onOpenChange={setIsCreateProjectOpen}>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Create New Project</DialogTitle>
              <DialogDescription>
                Create a new project to organize your files.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="name" className="text-right">
                  Name
                </Label>
                <Input
                  id="name"
                  value={newProjectNameInput}
                  onChange={(e) => setNewProjectNameInput(e.target.value)}
                  className="col-span-3"
                  placeholder="e.g. Summer 2024 Release"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsCreateProjectOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateProject} disabled={isCreatingProject || !newProjectNameInput.trim()}>
                {isCreatingProject ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Creating...
                  </>
                ) : (
                  "Create Project"
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <ToolIntroModal
          config={TOOL_CONFIGS.oneclick}
          isOpen={walkthrough.phase === "modal"}
          onStartTour={walkthrough.startSpotlight}
          onSkip={walkthrough.skip}
        />
        <WalkthroughProvider
          isActive={walkthrough.phase === "spotlight"}
          currentStep={walkthrough.currentStep}
          currentStepIndex={walkthrough.visibleStepIndex}
          totalSteps={walkthrough.totalSteps}
          onNext={walkthrough.next}
          onSkip={walkthrough.skip}
        />
      </main>
    </div>
  );
};

export default OneClickDocuments;
