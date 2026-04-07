import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Music, ArrowLeft, Loader2, AlertCircle, BookOpen } from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
import { useWorks } from "@/hooks/useRegistry";
import { type WorkFileLink } from "@/hooks/useWorkFiles";
import { API_URL, apiFetch, getAuthHeaders } from "@/lib/apiFetch";
import ContractSelector from "@/components/oneclick/ContractSelector";
import RoyaltyStatementSelector from "@/components/oneclick/RoyaltyStatementSelector";
import CalculationResults from "@/components/oneclick/CalculationResults";

interface RoyaltyPayment { song_title: string; party_name: string; role: string; royalty_type: string; percentage: number; total_royalty: number; amount_to_pay: number; terms?: string; }
interface CalculationResult { status: string; total_payments: number; payments: RoyaltyPayment[]; excel_file_url?: string; message: string; is_cached?: boolean; }
interface Project { id: string; name: string; }
interface ArtistFile { id: string; file_name: string; created_at: string; folder_category: string; file_path: string; project_id: string; }
interface Artist { id: string; name: string; }

const OneClickDocuments = () => {
  const navigate = useNavigate();
  const { artistId } = useParams<{ artistId: string }>();
  const { user } = useAuth();
  const [contractFiles, setContractFiles] = useState<File[]>([]);
  const [royaltyStatementFile, setRoyaltyStatementFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [calculationResult, setCalculationResult] = useState<CalculationResult | null>(null);
  const [error, setError] = useState<string>("");
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
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
  const [artistName, setArtistName] = useState<string>("");
  const [isLoadingArtist, setIsLoadingArtist] = useState(true);
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
  const { data: artistWorks = [], isLoading: isLoadingWorks } = useWorks(artistId);
  const [selectedWorkId, setSelectedWorkId] = useState<string | null>(null);
  const [workFiles, setWorkFiles] = useState<WorkFileLink[]>([]);
  const [loadingWorkFiles, setLoadingWorkFiles] = useState(false);
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

    const authHeaders = await getAuthHeaders();
    const response = await fetch(`${API_URL}/files/${projectId}`, { headers: authHeaders });
    if (!response.ok) {
      throw new Error("Failed to load existing files for duplicate check");
    }

    const data: ArtistFile[] = await response.json();
    setProjectFilesById(prev => ({ ...prev, [projectId]: data }));
    return data;
  };

  useEffect(() => {
    if (artistId && user) {
      setIsLoadingArtist(true);
      apiFetch<Artist[]>(`${API_URL}/artists`)
        .then((data: Artist[]) => {
          const artist = data.find(a => a.id === artistId);
          if (artist) setArtistName(artist.name);
          setIsLoadingArtist(false);
        })
        .catch(err => {
          console.error("Error fetching artist:", err);
          setIsLoadingArtist(false);
        });
    }
  }, [artistId, user]);

  useEffect(() => {
    if (artistId) {
      apiFetch<any>(`${API_URL}/projects/${artistId}`)
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

  useEffect(() => {
    if (!selectedWorkId || !user) { setWorkFiles([]); return; }
    setLoadingWorkFiles(true);
    apiFetch<any>(`${API_URL}/registry/works/${selectedWorkId}/files`)
      .then((data) => {
        setWorkFiles(data.files || []);
        setLoadingWorkFiles(false);
      })
      .catch((err) => {
        console.error("Error fetching work files:", err);
        setWorkFiles([]);
        setLoadingWorkFiles(false);
      });
  }, [selectedWorkId, user]);

  const handleCreateProject = async () => {
    if (!artistId || !newProjectNameInput.trim()) return;
    setIsCreatingProject(true);
    try {
        const response = await fetch(`${API_URL}/projects`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ artist_id: artistId, name: newProjectNameInput, description: "Created via OneClick", user_id: user?.id })
        });
        if (!response.ok) throw new Error("Failed to create project");
        const newProject = await response.json();
        setProjects([newProject, ...projects]);
        setNewContractProjectId(newProject.id);
        setNewRoyaltyStatementProjectId(newProject.id);
        setNewProjectNameInput(""); setIsCreateProjectOpen(false);
        toast.success("Project created successfully");
    } catch (err) { console.error("Error creating project:", err); toast.error("Failed to create project"); }
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
    setError("");
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

        // Use EventSource for SSE
        const queryParams = new URLSearchParams({
            project_id: finalProjectId,
            royalty_statement_file_id: finalStatementId
        });
        if (forceRecalculate) queryParams.append("force_recalculate", "true");
        finalContractIds.forEach(id => queryParams.append("contract_ids", id));

        const eventSource = new EventSource(`${API_URL}/oneclick/calculate-royalties-stream?` + queryParams.toString());
        const timeout = setTimeout(() => {
            eventSource.close(); setShowProgressModal(false);
            setError("Calculation timed out. Please try again."); toast.error("Calculation timed out"); setIsUploading(false);
        }, 120000);

        const handleSSEComplete = (data: any) => {
            clearTimeout(timeout); eventSource.close();
            setCalculationResult({ status: data.status, total_payments: data.total_payments, payments: data.payments, message: data.message, is_cached: data.is_cached });
            setShowProgressModal(false);
            toast.success(data.is_cached ? "Royalties loaded successfully!" : "Royalties calculated successfully!");
            setContractFiles([]); setRoyaltyStatementFile(null); setIsUploading(false);
        };

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'status') {
                    setCalculationProgress(data.progress || 0); setCalculationStage(data.stage || ""); setCalculationMessage(data.message || "");
                } else if (data.type === 'complete' || (data.status === 'success' && data.payments)) {
                    handleSSEComplete(data);
                } else if (data.type === 'error') {
                    clearTimeout(timeout); eventSource.close(); setShowProgressModal(false);
                    setError(data.message || "An error occurred"); toast.error(data.message || "Calculation failed"); setIsUploading(false);
                }
            } catch (err) { console.error("Error parsing SSE data:", err); }
        };

        eventSource.onerror = () => {
            clearTimeout(timeout); eventSource.close(); setShowProgressModal(false);
            setError("Connection error. Please try again."); toast.error("Connection error during calculation"); setIsUploading(false);
        };

    } catch (error: any) {
        console.error("Error:", error);
        const errorMessage = error?.message || "An error occurred.";
        setError(errorMessage);
        if (!String(errorMessage).toLowerCase().includes("duplicate file")) toast.error(errorMessage || "An error occurred during processing.");
        setShowProgressModal(false); setIsUploading(false);
    }
  };

  const [lastCalculationContext, setLastCalculationContext] = useState<{
      contractIds: string[], statementId: string, projectId: string
  } | null>(null);

  const handleConfirmResultsWithContext = async () => {
      if (!lastCalculationContext || !calculationResult || !user) return;
      setIsSaving(true);
      try {
          const confirmAuthHeaders = await getAuthHeaders();
          const response = await fetch(`${API_URL}/oneclick/confirm`, {
              method: "POST",
              headers: { "Content-Type": "application/json", ...confirmAuthHeaders },
              body: JSON.stringify({
                  contract_ids: lastCalculationContext.contractIds,
                  royalty_statement_id: lastCalculationContext.statementId,
                  project_id: lastCalculationContext.projectId,
                  results: calculationResult
              })
          });
          if (!response.ok) throw new Error("Failed to save results");
          setSaveSuccess(true);
      } catch (err) { console.error("Error saving results:", err); toast.error("Failed to save results"); }
      finally { setIsSaving(false); }
  };

  useEffect(() => {
      if (!calculationResult || !lastCalculationContext || !user) return;
      if (calculationResult.is_cached) return;
      const resultKey = `${lastCalculationContext.statementId}-${lastCalculationContext.contractIds.join(',')}`;
      if (autoSaveTriggeredRef.current === resultKey) return;
      autoSaveTriggeredRef.current = resultKey;
      handleConfirmResultsWithContext();
  }, [calculationResult, lastCalculationContext, user]);

  if (isLoadingArtist) {
    return (
      <div className="min-h-screen bg-background">
        <header className="border-b border-border bg-card">
          <div className="container mx-auto px-4 py-4 flex items-center justify-between">
            <div className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity" onClick={() => navigate("/dashboard")}>
              <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
                <Music className="w-6 h-6 text-primary-foreground" />
              </div>
              <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => navigate("/docs")}
                title="Documentation"
                className="text-muted-foreground hover:text-foreground"
              >
                <BookOpen className="w-4 h-4" />
              </Button>
              <Button variant="outline" onClick={() => navigate("/tools/oneclick")}>
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Artist Selection
              </Button>
            </div>
          </div>
        </header>
        <main className="container mx-auto px-4 py-8 max-w-4xl">
          <div className="flex flex-col items-center justify-center min-h-[60vh]">
            <Loader2 className="animate-spin h-16 w-16 text-primary mb-4" />
            <p className="text-lg text-muted-foreground">Loading artist information...</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity" onClick={() => navigate("/dashboard")}>
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/docs")}
              title="Documentation"
              className="text-muted-foreground hover:text-foreground"
            >
              <BookOpen className="w-4 h-4" />
            </Button>
            <ToolHelpButton onClick={walkthrough.replay} />
            <Button variant="outline" onClick={() => navigate("/tools/oneclick")}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Artist Selection
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-foreground mb-2">OneClick Royalty Calculator</h2>
          <p className="text-muted-foreground">
            Upload documents for {artistName || "Artist"}
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
            isLoadingWorks={isLoadingWorks}
            artistWorks={artistWorks}
            selectedWorkId={selectedWorkId}
            setSelectedWorkId={setSelectedWorkId}
            workFiles={workFiles}
            setWorkFiles={setWorkFiles}
            loadingWorkFiles={loadingWorkFiles}
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
          />
        </div>

        <div className="flex gap-3 justify-center mb-8">
          <Button
            data-walkthrough="oneclick-calculate"
            onClick={() => handleCalculateRoyalties(false)}
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
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
            </Alert>
        )}

        <CalculationResults
          showProgressModal={showProgressModal}
          calculationProgress={calculationProgress}
          calculationStage={calculationStage}
          calculationMessage={calculationMessage}
          calculationResult={calculationResult}
          isUploading={isUploading}
          handleCalculateRoyalties={handleCalculateRoyalties}
        />

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
