// React hooks for managing component state and side effects
import { useState, useEffect, useRef } from "react";
// UI components from shadcn/ui library for building the interface
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChartContainer, ChartTooltip } from "@/components/ui/chart";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
// Icons from lucide-react for visual elements
import { Music, ArrowLeft, Upload, FileText, X, FileSignature, Receipt, Users, DollarSign, Download, FileSpreadsheet, CheckCircle2, Folder, Loader2, AlertCircle, Search, Plus } from "lucide-react";
// React Router hooks for navigation and getting URL parameters
import { useNavigate, useParams } from "react-router-dom";
// Recharts for pie chart
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
// xlsx library for Excel export
import * as XLSX from "xlsx";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { toPng } from 'html-to-image';


// Backend API URL
const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

// Type definitions
interface RoyaltyPayment {
  song_title: string;
  party_name: string;
  role: string;
  royalty_type: string;
  percentage: number;
  total_royalty: number;
  amount_to_pay: number;
  terms?: string;
}

interface CalculationResult {
  status: string;
  total_payments: number;
  payments: RoyaltyPayment[];
  excel_file_url?: string;
  message: string;
}

interface Project {
  id: string;
  name: string;
}

interface ArtistFile {
  id: string;
  file_name: string;
  created_at: string;
  folder_category: string;
  file_path: string; // Needed for calculation
  project_id: string;
}

interface Artist {
  id: string;
  name: string;
}

const OneClickDocuments = () => {
  const navigate = useNavigate();
  const { artistId } = useParams<{ artistId: string }>();
  const { user } = useAuth();
  
  // File Upload State
  const [contractFiles, setContractFiles] = useState<File[]>([]);
  const [royaltyStatementFiles, setRoyaltyStatementFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  
  // Calculation Results State
  const [calculationResult, setCalculationResult] = useState<CalculationResult | null>(null);
  const [error, setError] = useState<string>("");

  // Existing Documents Selection State
  const [selectedExistingContracts, setSelectedExistingContracts] = useState<string[]>([]); // Array of file_ids (OneClick uses IDs)
  const [selectedExistingRoyaltyStatements, setSelectedExistingRoyaltyStatements] = useState<string[]>([]); // Array of file_ids (OneClick uses IDs)

  // Data from Backend
  const [projects, setProjects] = useState<Project[]>([]);
  const [existingContracts, setExistingContracts] = useState<ArtistFile[]>([]);
  const [existingRoyaltyStatements, setExistingRoyaltyStatements] = useState<ArtistFile[]>([]);
  
  // UI State - Separate project selection for each card
  const [selectedContractProject, setSelectedContractProject] = useState<string | null>(null);
  const [selectedRoyaltyStatementProject, setSelectedRoyaltyStatementProject] = useState<string | null>(null);
  const [newContractProjectId, setNewContractProjectId] = useState<string>("");
  
  // Tab state to maintain which tab is active in each card
  const [contractTabValue, setContractTabValue] = useState<string>("upload");
  const [royaltyStatementTabValue, setRoyaltyStatementTabValue] = useState<string>("upload");
  const [newRoyaltyStatementProjectId, setNewRoyaltyStatementProjectId] = useState<string>("");

  const [artistName, setArtistName] = useState<string>("");
  const [isLoadingArtist, setIsLoadingArtist] = useState(true);
  const [isLoadingProjectFiles, setIsLoadingProjectFiles] = useState(false);

  // Progress tracking state for SSE
  const [calculationProgress, setCalculationProgress] = useState(0);
  const [calculationStage, setCalculationStage] = useState("");
  const [calculationMessage, setCalculationMessage] = useState("");
  const [showProgressModal, setShowProgressModal] = useState(false);
  const [contractSearchTerm, setContractSearchTerm] = useState("");
  const [royaltySearchTerm, setRoyaltySearchTerm] = useState("");

  // Create Project State
  const [isCreateProjectOpen, setIsCreateProjectOpen] = useState(false);
  const [newProjectNameInput, setNewProjectNameInput] = useState("");
  const [isCreatingProject, setIsCreatingProject] = useState(false);

  // Ref for chart download (only the chart content, not the entire card)
  const chartContentRef = useRef<HTMLDivElement>(null);

  // Fetch Artist Name
  useEffect(() => {
    if (artistId && user) {
      setIsLoadingArtist(true);
      fetch(`${API_URL}/artists?user_id=${user.id}`)
        .then(res => res.json())
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

  // Fetch Projects on Mount
  useEffect(() => {
    if (artistId) {
      fetch(`${API_URL}/projects/${artistId}`)
        .then(res => res.json())
        .then(data => setProjects(data))
        .catch(err => console.error("Error fetching projects:", err));

      // Fetch all royalty statements for the artist (since view is "all")
      // fetch(`${API_URL}/files/artist/${artistId}/category/royalty_statement`)
      //   .then(res => res.json())
      //   .then(data => setExistingRoyaltyStatements(data))
      //   .catch(err => console.error("Error fetching royalty statements:", err));
    }
  }, [artistId]);

  // Fetch Contracts when Contract Project Selected
  useEffect(() => {
    if (selectedContractProject) {
      // Clear previous contract selections when changing projects
      setSelectedExistingContracts([]);
      
      setIsLoadingProjectFiles(true);
      fetch(`${API_URL}/files/${selectedContractProject}`)
        .then(res => res.json())
        .then((data: ArtistFile[]) => {
            // Filter contracts
            const contracts = data.filter(f => f.folder_category === 'contract');
            setExistingContracts(contracts);
            setIsLoadingProjectFiles(false);
        })
        .catch(err => {
          console.error("Error fetching contract files:", err);
          setIsLoadingProjectFiles(false);
        });
    } else {
        setExistingContracts([]);
        setSelectedExistingContracts([]);
    }
  }, [selectedContractProject]);

  // Fetch Royalty Statements when Royalty Statement Project Selected
  useEffect(() => {
    if (selectedRoyaltyStatementProject) {
      // Clear previous royalty statement selections when changing projects
      setSelectedExistingRoyaltyStatements([]);
      
      setIsLoadingProjectFiles(true);
      fetch(`${API_URL}/files/${selectedRoyaltyStatementProject}`)
        .then(res => res.json())
        .then((data: ArtistFile[]) => {
            // Filter royalty statements
            const statements = data.filter(f => f.folder_category === 'royalty_statement');
            setExistingRoyaltyStatements(statements);
            setIsLoadingProjectFiles(false);
        })
        .catch(err => {
          console.error("Error fetching royalty statement files:", err);
          setIsLoadingProjectFiles(false);
        });
    } else {
        setExistingRoyaltyStatements([]);
        setSelectedExistingRoyaltyStatements([]);
    }
  }, [selectedRoyaltyStatementProject]);

  const handleContractFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) setContractFiles(prev => [...prev, ...Array.from(files)]);
  };

  const handleRoyaltyStatementFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) setRoyaltyStatementFiles(prev => [...prev, ...Array.from(files)]);
  };

  const handleRemoveContractFile = (index: number) => {
    setContractFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleRemoveRoyaltyStatementFile = (index: number) => {
    setRoyaltyStatementFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleCreateProject = async () => {
    if (!artistId || !newProjectNameInput.trim()) return;
    
    setIsCreatingProject(true);
    try {
        const response = await fetch(`${API_URL}/projects`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                artist_id: artistId,
                name: newProjectNameInput,
                description: "Created via OneClick"
            })
        });
        
        if (!response.ok) throw new Error("Failed to create project");
        
        const newProject = await response.json();
        setProjects([newProject, ...projects]);
        setNewContractProjectId(newProject.id); // Auto-select for contract
        setNewRoyaltyStatementProjectId(newProject.id); // Auto-select for royalty statement
        setNewProjectNameInput("");
        setIsCreateProjectOpen(false);
        toast.success("Project created successfully");
    } catch (err) {
        console.error("Error creating project:", err);
        toast.error("Failed to create project");
    } finally {
        setIsCreatingProject(false);
    }
  };

  // Toggle selection store ID instead of file_path for easier backend processing with existing endpoints
  const handleToggleExistingContract = (fileId: string) => {
    setSelectedExistingContracts(prev =>
      prev.includes(fileId) ? prev.filter(p => p !== fileId) : [...prev, fileId]
    );
  };

  const handleToggleExistingRoyaltyStatement = (fileId: string) => {
    setSelectedExistingRoyaltyStatements(prev =>
      prev.includes(fileId) ? prev.filter(p => p !== fileId) : [...prev, fileId]
    );
  };

  const handleCalculateRoyalties = async () => {
    if (!artistId || !user) {
        toast.error("User or Artist not found.");
        return;
    }

    const hasContracts = contractFiles.length > 0 || selectedExistingContracts.length > 0;
    const hasRoyaltyStatements = royaltyStatementFiles.length > 0 || selectedExistingRoyaltyStatements.length > 0;
    
    if (!hasContracts || !hasRoyaltyStatements) {
        toast.error("Please provide both contracts and royalty statements.");
        return;
    }

    // OneClick currently supports multiple contracts but ONE statement at a time.
    
    // Check if multiple statements selected - warn user (or just use first)
    if ((royaltyStatementFiles.length + selectedExistingRoyaltyStatements.length) > 1) {
          toast.warning("Note: OneClick currently processes one royalty statement at a time. Using the first selected.");
    }

    setIsUploading(true);
    setCalculationResult(null);
    setError("");
    
    try {
        let finalContractIds: string[] = [];
        let finalStatementId = "";
        let finalProjectId = "";

        // 1. Determine Contract IDs
        if (contractFiles.length > 0) {
            // Upload new contracts
            if (!newContractProjectId || newContractProjectId === "none") {
                 toast.error("Please select a project to save the new contract.");
                 throw new Error("You must select a project to save the new contract to.");
             }

             // Upload each file sequentially to ensure order and error handling
             for (const file of contractFiles) {
                 const formData = new FormData();
                 formData.append("file", file);
                 formData.append("project_id", newContractProjectId);
                 formData.append("user_id", user.id); 
                 
                 const uploadRes = await fetch(`${API_URL}/contracts/upload`, {
                     method: "POST",
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
            // Find project ID for this contract if not set
            if (!finalProjectId) {
                const contract = existingContracts.find(c => c.id === selectedExistingContracts[0]);
                if (contract) finalProjectId = contract.project_id;
            }
        }

        // 2. Determine Statement ID
        if (royaltyStatementFiles.length > 0) {
             const formData = new FormData();
             formData.append("file", royaltyStatementFiles[0]);
             formData.append("artist_id", artistId);
             formData.append("category", "royalty_statement");
             // Associate with explicitly selected project, or contract's project if available
             const targetProjectId = newRoyaltyStatementProjectId || finalProjectId;
             if (targetProjectId) formData.append("project_id", targetProjectId);
             
             const uploadRes = await fetch(`${API_URL}/upload`, {
                 method: "POST",
                 body: formData
             });

             if (!uploadRes.ok) throw new Error("Failed to upload royalty statement");
             const uploadData = await uploadRes.json();
             finalStatementId = uploadData.file.id;

        } else if (selectedExistingRoyaltyStatements.length > 0) {
            finalStatementId = selectedExistingRoyaltyStatements[0];
        }

        if (finalContractIds.length === 0 || !finalStatementId) {
             throw new Error("Could not determine contracts or statement to process.");
        }

        // 3. Show progress modal and start SSE connection
        setShowProgressModal(true);
        setCalculationProgress(0);
        setCalculationStage("starting");
        setCalculationMessage("Starting calculation...");

        // Build request body
        const requestBody = {
            contract_ids: finalContractIds,
            user_id: user.id,
            project_id: finalProjectId,
            royalty_statement_file_id: finalStatementId
        };

        // Use EventSource for SSE
        const queryParams = new URLSearchParams({
            user_id: requestBody.user_id,
            project_id: requestBody.project_id,
            royalty_statement_file_id: requestBody.royalty_statement_file_id
        });
        
        // Append each contract_id
        finalContractIds.forEach(id => queryParams.append("contract_ids", id));

        const eventSource = new EventSource(
            `${API_URL}/oneclick/calculate-royalties-stream?` + queryParams.toString()
        );

        // Set timeout for 2 minutes
        const timeout = setTimeout(() => {
            eventSource.close();
            setShowProgressModal(false);
            setError("Calculation timed out. Please try again.");
            toast.error("Calculation timed out");
            setIsUploading(false);
        }, 120000);

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                if (data.type === 'status') {
                    setCalculationProgress(data.progress || 0);
                    setCalculationStage(data.stage || "");
                    setCalculationMessage(data.message || "");
                } else if (data.type === 'complete') {
                    clearTimeout(timeout);
                    eventSource.close();
                    
                    // Set results
                    const result: CalculationResult = {
                        status: data.status,
                        total_payments: data.total_payments,
                        payments: data.payments,
                        message: data.message
                    };
                    
                    setCalculationResult(result);
                    setShowProgressModal(false);
                    toast.success("Royalties calculated successfully!");
                    
                    // Clear uploaded files from state
                    setContractFiles([]);
                    setRoyaltyStatementFiles([]);
                    setIsUploading(false);
                } else if (data.type === 'error') {
                    clearTimeout(timeout);
                    eventSource.close();
                    setShowProgressModal(false);
                    setError(data.message || "An error occurred");
                    toast.error(data.message || "Calculation failed");
                    setIsUploading(false);
                }
            } catch (err) {
                console.error("Error parsing SSE data:", err);
            }
        };

        eventSource.onerror = (err) => {
            console.error("EventSource error:", err);
            clearTimeout(timeout);
            eventSource.close();
            setShowProgressModal(false);
            setError("Connection error. Please try again.");
            toast.error("Connection error during calculation");
            setIsUploading(false);
        };

    } catch (error: any) {
        console.error("Error:", error);
        setError(error.message || "An error occurred.");
        toast.error(error.message || "An error occurred during processing.");
        setShowProgressModal(false);
        setIsUploading(false);
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  const handleExportCSV = () => {
    if (!calculationResult) return;
    const headers = ["Song Title", "Payee", "Role", "Royalty Type", "Total Revenue", "Share %", "Amount to Pay"];
    const rows = calculationResult.payments.map(p => [
      p.song_title,
      p.party_name,
      p.role,
      p.royalty_type,
      p.total_royalty,
      `${p.percentage}%`,
      p.amount_to_pay
    ]);
    const csvContent = [headers.join(","), ...rows.map(row => row.map(cell => `"${cell}"`).join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `royalty_breakdown.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleExportExcel = () => {
    if (!calculationResult) return;
    const excelData = [
      ["Song Title", "Payee", "Role", "Royalty Type", "Total Revenue", "Share %", "Amount to Pay"],
      ...calculationResult.payments.map(p => [
        p.song_title, p.party_name, p.role, p.royalty_type, p.total_royalty, p.percentage, p.amount_to_pay
      ])
    ];
    const ws = XLSX.utils.aoa_to_sheet(excelData);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Royalty Breakdown");
    XLSX.writeFile(wb, `royalty_breakdown.xlsx`);
  };

  const handleDownloadChart = async () => {
    if (!chartContentRef.current || !calculationResult) return;
    
    try {
      const dataUrl = await toPng(chartContentRef.current, {
        quality: 1,
        pixelRatio: 2,
        backgroundColor: '#ffffff'
      });
      
      const link = document.createElement('a');
      link.download = `royalty-distribution.png`;
      link.href = dataUrl;
      link.click();
      toast.success("Chart downloaded successfully!");
    } catch (error) {
      console.error("Error downloading chart:", error);
      toast.error("Failed to download chart. Please try again.");
    }
  };

  if (isLoadingArtist) {
    return (
      <div className="min-h-screen bg-background">
        <header className="border-b border-border bg-card">
          <div className="container mx-auto px-4 py-4 flex items-center justify-between">
            <div className="flex items-center gap-2 cursor-pointer" onClick={() => navigate("/dashboard")}>
              <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
                <Music className="w-6 h-6 text-primary-foreground" />
              </div>
              <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
            </div>
            <Button variant="outline" onClick={() => navigate("/tools/oneclick")}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Artist Selection
            </Button>
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
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => navigate("/dashboard")}>
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
          </div>
          <Button variant="outline" onClick={() => navigate("/tools/oneclick")}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Artist Selection
          </Button>
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
          {/* Contract Upload Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileSignature className="w-5 h-5 text-primary" />
                Upload Contract
              </CardTitle>
              <CardDescription>Upload artist contract documents</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Tabs value={contractTabValue} onValueChange={setContractTabValue} className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="upload">Upload New</TabsTrigger>
                  <TabsTrigger value="existing">Select Existing</TabsTrigger>
                </TabsList>
                
                <TabsContent value="upload" className="space-y-4 mt-4">
                  <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
                    <FileSignature className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                    <p className="text-foreground font-medium mb-2 text-sm">Upload Contract</p>
                    <div className="flex items-center justify-center gap-2 mb-4">
                      <div className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 rounded-md">
                        <FileText className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
                        <span className="text-xs font-medium text-blue-700 dark:text-blue-300">PDF accepted</span>
                      </div>
                    </div>
                    <Input
                      id="contract-upload"
                      type="file"
                      // Removed multiple for OneClick simplicity
                      accept=".pdf,application/pdf"
                      onChange={handleContractFileChange}
                      className="hidden"
                    />
                    <label htmlFor="contract-upload">
                      <Button variant="outline" size="sm" asChild className="cursor-pointer">
                        <span><Upload className="w-4 h-4 mr-2" />Select File</span>
                      </Button>
                    </label>
                  </div>

                  {contractFiles.length > 0 && (
                    <div className="space-y-2 pt-2 border-t border-border">
                      <div className="flex items-center gap-2">
                        <Folder className="w-4 h-4 text-muted-foreground" />
                        <label htmlFor="project-select" className="text-sm font-medium text-foreground">
                          Save to Project (Required)
                        </label>
                      </div>
                      <div className="flex items-center gap-2 w-full">
                        <Select value={newContractProjectId} onValueChange={setNewContractProjectId}>
                          <SelectTrigger id="project-select" className="w-full">
                            <SelectValue placeholder="Select a project..." />
                          </SelectTrigger>
                          <SelectContent>
                            {projects.map((project) => (
                              <SelectItem key={project.id} value={project.id}>{project.name}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <Button
                          variant="outline"
                          size="icon"
                          onClick={() => setIsCreateProjectOpen(true)}
                          title="Create New Project"
                        >
                          <Plus className="h-4 w-4" />
                        </Button>
                      </div>
                      <p className="text-xs text-muted-foreground">Select a project to save the contract.</p>
                    </div>
                  )}

                  {contractFiles.length > 0 && (
                    <div className="space-y-2 mt-4">
                      {contractFiles.map((file, index) => (
                        <div key={index} className="flex items-center justify-between p-2 border border-border rounded-lg bg-secondary/50">
                          <div className="flex items-center gap-2 flex-1 min-w-0">
                            <FileText className="w-4 h-4 text-primary flex-shrink-0" />
                            <p className="text-xs font-medium text-foreground truncate">{file.name}</p>
                          </div>
                          <Button variant="ghost" size="sm" onClick={() => handleRemoveContractFile(index)} className="text-destructive hover:text-destructive flex-shrink-0">
                            <X className="w-3 h-3" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="existing" className="space-y-4 mt-4">
                  {!selectedContractProject ? (
                    <div className="space-y-4">
                      <p className="text-sm font-medium text-foreground">Select a Project:</p>
                      <div className="grid gap-3">
                        {projects.length > 0 ? projects.map((project) => (
                          <div key={project.id} className="flex items-center gap-3 p-4 border border-border rounded-lg hover:bg-secondary/50 transition-colors cursor-pointer" onClick={() => setSelectedContractProject(project.id)}>
                            <Folder className="w-5 h-5 text-primary" />
                            <span className="font-medium text-foreground">{project.name}</span>
                          </div>
                        )) : (
                           <p className="text-sm text-muted-foreground text-center py-4">No projects found.</p>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-sm font-medium text-foreground">
                          Contracts in {projects.find(p => p.id === selectedContractProject)?.name}:
                        </p>
                        <Button variant="ghost" size="sm" onClick={() => setSelectedContractProject(null)} className="text-xs h-8">
                          Change Project
                        </Button>
                      </div>
                      
                      <div className="relative">
                        <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                        <Input
                          placeholder="Search contracts..."
                          value={contractSearchTerm}
                          onChange={(e) => setContractSearchTerm(e.target.value)}
                          className="pl-8"
                        />
                      </div>

                      {isLoadingProjectFiles ? (
                        <div className="text-center py-8">
                          <Loader2 className="inline-block animate-spin h-8 w-8 text-primary mb-2" />
                          <p className="text-sm text-muted-foreground">Loading contracts...</p>
                        </div>
                      ) : existingContracts.length > 0 ? (
                        <div className="space-y-2 max-h-[300px] overflow-y-auto">
                          {existingContracts
                            .filter(contract => 
                              contract.file_name.toLowerCase().includes(contractSearchTerm.toLowerCase())
                            )
                            .map((contract) => (
                              <div key={contract.id} className="flex items-center justify-between p-3 border border-border rounded-lg hover:bg-secondary/50 transition-colors">
                                <div className="flex items-center gap-3 flex-1">
                                  <Checkbox
                                    id={`existing-contract-${contract.id}`}
                                    checked={selectedExistingContracts.includes(contract.id)}
                                    onCheckedChange={() => handleToggleExistingContract(contract.id)}
                                  />
                                  <label htmlFor={`existing-contract-${contract.id}`} className="flex items-center gap-2 flex-1 cursor-pointer">
                                    <FileText className="w-4 h-4 text-primary" />
                                    <div>
                                      <p className="text-sm font-medium text-foreground">{contract.file_name}</p>
                                      <p className="text-xs text-muted-foreground">{new Date(contract.created_at).toLocaleDateString()}</p>
                                    </div>
                                  </label>
                                </div>
                              </div>
                            ))}
                          {existingContracts.filter(contract => 
                            contract.file_name.toLowerCase().includes(contractSearchTerm.toLowerCase())
                          ).length === 0 && (
                             <p className="text-sm text-muted-foreground text-center py-4">No matching contracts found.</p>
                          )}
                        </div>
                      ) : (
                        <p className="text-sm text-muted-foreground text-center py-4">No contracts found in this project.</p>
                      )}
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Royalty Statement Upload Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Receipt className="w-5 h-5 text-primary" />
                Upload Royalty Statement
              </CardTitle>
              <CardDescription>Upload royalty statement documents</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Tabs value={royaltyStatementTabValue} onValueChange={setRoyaltyStatementTabValue} className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="upload">Upload New</TabsTrigger>
                  <TabsTrigger value="existing">Select Existing</TabsTrigger>
                </TabsList>
                
                <TabsContent value="upload" className="space-y-4 mt-4">
                  <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
                    <Receipt className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                    <p className="text-foreground font-medium mb-2 text-sm">Upload Royalty Statement</p>
                    <div className="flex items-center justify-center gap-2 mb-4">
                      <div className="flex items-center gap-1.5 px-3 py-1.5 bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800 rounded-md">
                        <FileSpreadsheet className="w-3.5 h-3.5 text-green-600 dark:text-green-400" />
                        <span className="text-xs font-medium text-green-700 dark:text-green-300">XLSX or CSV only</span>
                      </div>
                    </div>
                    <Input
                      id="royalty-upload"
                      type="file"
                      // Removed multiple
                      accept=".xlsx,.csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,text/csv"
                      onChange={handleRoyaltyStatementFileChange}
                      className="hidden"
                    />
                    <label htmlFor="royalty-upload">
                      <Button variant="outline" size="sm" asChild className="cursor-pointer">
                        <span><Upload className="w-4 h-4 mr-2" />Select File</span>
                      </Button>
                    </label>
                  </div>

                  {royaltyStatementFiles.length > 0 && (
                    <div className="space-y-2 pt-2 border-t border-border">
                      <div className="flex items-center gap-2">
                        <Folder className="w-4 h-4 text-muted-foreground" />
                        <label htmlFor="royalty-project-select" className="text-sm font-medium text-foreground">
                          Save to Project (Optional)
                        </label>
                      </div>
                      <div className="flex items-center gap-2 w-full">
                        <Select value={newRoyaltyStatementProjectId} onValueChange={setNewRoyaltyStatementProjectId}>
                          <SelectTrigger id="royalty-project-select" className="w-full">
                            <SelectValue placeholder="Select a project..." />
                          </SelectTrigger>
                          <SelectContent>
                            {projects.map((project) => (
                              <SelectItem key={project.id} value={project.id}>{project.name}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <Button
                          variant="outline"
                          size="icon"
                          onClick={() => setIsCreateProjectOpen(true)}
                          title="Create New Project"
                        >
                          <Plus className="h-4 w-4" />
                        </Button>
                      </div>
                      <p className="text-xs text-muted-foreground">Select a project to save the royalty statement.</p>
                    </div>
                  )}

                  {royaltyStatementFiles.length > 0 && (
                    <div className="space-y-2 mt-4">
                      {royaltyStatementFiles.map((file, index) => (
                        <div key={index} className="flex items-center justify-between p-2 border border-border rounded-lg bg-secondary/50">
                          <div className="flex items-center gap-2 flex-1 min-w-0">
                            <FileText className="w-4 h-4 text-primary flex-shrink-0" />
                            <p className="text-xs font-medium text-foreground truncate">{file.name}</p>
                          </div>
                          <Button variant="ghost" size="sm" onClick={() => handleRemoveRoyaltyStatementFile(index)} className="text-destructive hover:text-destructive flex-shrink-0">
                            <X className="w-3 h-3" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="existing" className="space-y-4 mt-4">
                  {!selectedRoyaltyStatementProject ? (
                    <div className="space-y-4">
                      <p className="text-sm font-medium text-foreground">Select a Project:</p>
                      <div className="grid gap-3">
                        {projects.length > 0 ? projects.map((project) => (
                          <div key={project.id} className="flex items-center gap-3 p-4 border border-border rounded-lg hover:bg-secondary/50 transition-colors cursor-pointer" onClick={() => setSelectedRoyaltyStatementProject(project.id)}>
                            <Folder className="w-5 h-5 text-primary" />
                            <span className="font-medium text-foreground">{project.name}</span>
                          </div>
                        )) : (
                           <p className="text-sm text-muted-foreground text-center py-4">No projects found.</p>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-sm font-medium text-foreground">
                          Royalty Statements in {projects.find(p => p.id === selectedRoyaltyStatementProject)?.name}:
                        </p>
                        <Button variant="ghost" size="sm" onClick={() => setSelectedRoyaltyStatementProject(null)} className="text-xs h-8">
                          Change Project
                        </Button>
                      </div>
                      
                      <div className="relative">
                        <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                        <Input
                          placeholder="Search royalty statements..."
                          value={royaltySearchTerm}
                          onChange={(e) => setRoyaltySearchTerm(e.target.value)}
                          className="pl-8"
                        />
                      </div>
                      
                      {isLoadingProjectFiles ? (
                        <div className="text-center py-8">
                          <Loader2 className="inline-block animate-spin h-8 w-8 text-primary mb-2" />
                          <p className="text-sm text-muted-foreground">Loading royalty statements...</p>
                        </div>
                      ) : existingRoyaltyStatements.length > 0 ? (
                        <div className="space-y-2 max-h-[300px] overflow-y-auto">
                          {existingRoyaltyStatements
                            .filter(statement => 
                              statement.file_name.toLowerCase().includes(royaltySearchTerm.toLowerCase())
                            )
                            .map((statement) => (
                              <div key={statement.id} className="flex items-center justify-between p-3 border border-border rounded-lg hover:bg-secondary/50 transition-colors">
                                <div className="flex items-center gap-3 flex-1">
                                  <Checkbox
                                    id={`existing-royalty-${statement.id}`}
                                    checked={selectedExistingRoyaltyStatements.includes(statement.id)}
                                    onCheckedChange={() => handleToggleExistingRoyaltyStatement(statement.id)}
                                  />
                                  <label htmlFor={`existing-royalty-${statement.id}`} className="flex items-center gap-2 flex-1 cursor-pointer">
                                    <FileText className="w-4 h-4 text-primary" />
                                    <div>
                                      <p className="text-sm font-medium text-foreground">{statement.file_name}</p>
                                      <p className="text-xs text-muted-foreground">{new Date(statement.created_at).toLocaleDateString()}</p>
                                    </div>
                                  </label>
                                </div>
                              </div>
                            ))}
                          {existingRoyaltyStatements.filter(statement => 
                            statement.file_name.toLowerCase().includes(royaltySearchTerm.toLowerCase())
                          ).length === 0 && (
                             <p className="text-sm text-muted-foreground text-center py-4">No matching royalty statements found.</p>
                          )}
                        </div>
                      ) : (
                        <p className="text-sm text-muted-foreground text-center py-4">No royalty statements found in this project.</p>
                      )}
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>

        <div className="flex gap-3 justify-center mb-8">
          <Button
            onClick={handleCalculateRoyalties}
            disabled={((contractFiles.length === 0 && selectedExistingContracts.length === 0) || 
                       (royaltyStatementFiles.length === 0 && selectedExistingRoyaltyStatements.length === 0)) || 
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

        {/* Inline Progress Display */}
        {showProgressModal && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="text-center">Calculating Royalties</CardTitle>
              <CardDescription className="text-center">
                Please wait while we process your documents
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6 py-4">
              {/* Circular Progress */}
              <div className="flex flex-col items-center justify-center">
                <div className="relative w-32 h-32">
                  {/* Background circle */}
                  <svg className="w-32 h-32 transform -rotate-90">
                    <circle
                      cx="64"
                      cy="64"
                      r="56"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="none"
                      className="text-muted"
                    />
                    {/* Progress circle */}
                    <circle
                      cx="64"
                      cy="64"
                      r="56"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="none"
                      strokeDasharray={`${2 * Math.PI * 56}`}
                      strokeDashoffset={`${2 * Math.PI * 56 * (1 - calculationProgress / 100)}`}
                      className="text-primary transition-all duration-500 ease-out"
                      strokeLinecap="round"
                    />
                  </svg>
                  {/* Percentage text */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-2xl font-bold text-foreground">{Math.round(calculationProgress)}%</span>
                  </div>
                </div>
                
                {/* Current stage message */}
                <p className="mt-4 text-sm font-medium text-center text-muted-foreground">
                  {calculationMessage}
                </p>
              </div>

              {/* Stage indicators */}
              <div className="space-y-3">
                {[
                  { id: 'downloading', label: 'Downloading files', icon: FileText },
                  { id: 'extracting', label: 'Extracting data', icon: FileText },
                  { id: 'calculating', label: 'Calculating payments', icon: DollarSign },
                ].map((stageItem) => {
                  const getStageStatus = (stageId: string) => {
                    if (calculationStage.includes('download')) {
                      if (stageId === 'downloading') return 'active';
                      return 'pending';
                    } else if (calculationStage.includes('extract') || calculationStage.includes('parties') || calculationStage.includes('works') || calculationStage.includes('royalty') || calculationStage.includes('summary')) {
                      if (stageId === 'downloading') return 'complete';
                      if (stageId === 'extracting') return 'active';
                      return 'pending';
                    } else if (calculationStage.includes('processing') || calculationStage.includes('calculating')) {
                      if (stageId === 'downloading' || stageId === 'extracting') return 'complete';
                      if (stageId === 'calculating') return 'active';
                      return 'pending';
                    }
                    return 'pending';
                  };
                  
                  const status = getStageStatus(stageItem.id);
                  const Icon = stageItem.icon;
                  
                  return (
                    <div key={stageItem.id} className="flex items-center gap-3">
                      <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center ${
                        status === 'complete' 
                          ? 'bg-primary text-primary-foreground' 
                          : status === 'active'
                          ? 'bg-primary/20 text-primary'
                          : 'bg-muted text-muted-foreground'
                      }`}>
                        {status === 'complete' ? (
                          <CheckCircle2 className="w-4 h-4" />
                        ) : status === 'active' ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <Icon className="w-4 h-4" />
                        )}
                      </div>
                      <span className={`text-sm ${
                        status === 'complete' || status === 'active'
                          ? 'text-foreground font-medium'
                          : 'text-muted-foreground'
                      }`}>
                        {stageItem.label}
                      </span>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        )}

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

        {calculationResult && (
          <div className="mt-8 space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-foreground mb-2">Royalty Calculation Results</h2>
              <p className="text-muted-foreground">{calculationResult.message}</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Songs Processed</CardTitle>
                  <Music className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent><div className="text-2xl font-bold text-foreground">{new Set(calculationResult.payments.map(p => p.song_title)).size}</div></CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Payees</CardTitle>
                  <Users className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent><div className="text-2xl font-bold text-foreground">{new Set(calculationResult.payments.map(p => p.party_name)).size}</div></CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Revenue</CardTitle>
                  <DollarSign className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-foreground">
                    {formatCurrency(
                      Array.from(new Map(calculationResult.payments.map(p => [p.song_title, p.total_royalty])).values())
                        .reduce((sum, val) => sum + val, 0)
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Royalty Distribution</CardTitle>
                  <Button variant="outline" size="sm" onClick={handleDownloadChart}>
                    <Download className="mr-2 h-4 w-4" />
                    Download Chart
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div ref={chartContentRef} className="h-[450px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={Object.values(calculationResult.payments.reduce((acc, curr) => {
                            if (!acc[curr.party_name]) acc[curr.party_name] = { name: curr.party_name, value: 0 };
                            acc[curr.party_name].value += curr.amount_to_pay;
                            return acc;
                        }, {} as Record<string, {name: string, value: number}>)).sort((a, b) => b.value - a.value)}
                        cx="50%" 
                        cy="50%" 
                        outerRadius={130} 
                        dataKey="value"
                        nameKey="name"
                        label={({ name, value, percent }) => `${name} (${(percent * 100).toFixed(1)}%): ${formatCurrency(value)}`}
                        labelLine={true}
                        style={{ fontSize: '11px' }}
                      >
                         {Object.values(calculationResult.payments.reduce((acc, curr) => {
                            if (!acc[curr.party_name]) acc[curr.party_name] = { name: curr.party_name, value: 0 };
                            acc[curr.party_name].value += curr.amount_to_pay;
                            return acc;
                        }, {} as Record<string, {name: string, value: number}>))
                        .sort((a, b) => b.value - a.value)
                        .map((entry, index, arr) => (
                          <Cell key={`cell-${index}`} fill={`hsl(150, ${50 + (index / (arr.length || 1)) * 10}%, ${25 + (index / (arr.length || 1)) * 40}%)`} />
                        ))}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <div className="flex items-center justify-between">
                        <CardTitle>Royalty Breakdown</CardTitle>
                        <DropdownMenu>
                            <DropdownMenuTrigger asChild><Button variant="outline" size="sm"><Download className="mr-2 h-4 w-4"/> Export</Button></DropdownMenuTrigger>
                            <DropdownMenuContent>
                                <DropdownMenuItem onClick={handleExportCSV}>Export as CSV</DropdownMenuItem>
                                <DropdownMenuItem onClick={handleExportExcel}>Export as Excel</DropdownMenuItem>
                            </DropdownMenuContent>
                        </DropdownMenu>
                    </div>
                </CardHeader>
                <CardContent>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Song</TableHead><TableHead>Payee</TableHead><TableHead>Role</TableHead><TableHead>Royalty Type</TableHead><TableHead>Total Revenue</TableHead><TableHead>Share %</TableHead><TableHead>Amount</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {calculationResult.payments.map((row, i) => (
                                <TableRow key={i}>
                                    <TableCell>{row.song_title}</TableCell>
                                    <TableCell>{row.party_name}</TableCell>
                                    <TableCell className="capitalize">{row.role}</TableCell>
                                    <TableCell className="capitalize">{row.royalty_type}</TableCell>
                                    <TableCell>{formatCurrency(row.total_royalty)}</TableCell>
                                    <TableCell>{row.percentage}%</TableCell>
                                    <TableCell>{formatCurrency(row.amount_to_pay)}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>
          </div>
        )}
      </main>
    </div>
  );
};

export default OneClickDocuments;
