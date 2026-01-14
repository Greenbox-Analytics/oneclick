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
import { Music, ArrowLeft, Upload, FileText, X, FileSignature, Receipt, Users, DollarSign, Download, FileSpreadsheet, CheckCircle2, Folder, Loader2, AlertCircle } from "lucide-react";
// React Router hooks for navigation and getting URL parameters
import { useNavigate, useParams } from "react-router-dom";
// Recharts for pie chart
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
// xlsx library for Excel export
import * as XLSX from "xlsx";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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
  
  // UI State
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [newContractProjectId, setNewContractProjectId] = useState<string>("");

  const [artistName, setArtistName] = useState<string>("");
  const [isLoadingArtist, setIsLoadingArtist] = useState(true);
  const [isLoadingProjectFiles, setIsLoadingProjectFiles] = useState(false);

  // Ref for chart download (only the chart content, not the entire card)
  const chartContentRef = useRef<HTMLDivElement>(null);

  // Fetch Artist Name
  useEffect(() => {
    if (artistId) {
      setIsLoadingArtist(true);
      fetch(`${API_URL}/artists`)
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
  }, [artistId]);

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

  // Fetch Contracts and Royalty Statements when Project Selected
  useEffect(() => {
    if (selectedProject) {
      // Clear previous selections when changing projects
      setSelectedExistingContracts([]);
      setSelectedExistingRoyaltyStatements([]);
      
      setIsLoadingProjectFiles(true);
      fetch(`${API_URL}/files/${selectedProject}`)
        .then(res => res.json())
        .then((data: ArtistFile[]) => {
            // Filter contracts
            const contracts = data.filter(f => f.folder_category === 'contract');
            setExistingContracts(contracts);
            
            // Filter royalty statements
            const statements = data.filter(f => f.folder_category === 'royalty_statement');
            setExistingRoyaltyStatements(statements);
            
            setIsLoadingProjectFiles(false);
        })
        .catch(err => {
          console.error("Error fetching project files:", err);
          setIsLoadingProjectFiles(false);
        });
    } else {
        setExistingContracts([]);
        setExistingRoyaltyStatements([]);
        setSelectedExistingContracts([]);
        setSelectedExistingRoyaltyStatements([]);
        setIsLoadingProjectFiles(false);
    }
  }, [selectedProject]);

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

    // OneClick currently supports ONE contract and ONE statement at a time in the backend endpoint.
    // We will take the first one available.
    
    // Check if multiple files selected - warn user (or just use first)
    if ((contractFiles.length + selectedExistingContracts.length) > 1 || 
        (royaltyStatementFiles.length + selectedExistingRoyaltyStatements.length) > 1) {
          toast.warning("Note: OneClick currently processes one contract and one statement at a time. Using the first selected.");
    }

    setIsUploading(true);
    setCalculationResult(null);
    setError("");
    
    try {
        let finalContractId = "";
        let finalStatementId = "";
        let finalProjectId = "";

        // 1. Determine Contract ID
        if (contractFiles.length > 0) {
            // Upload new contract
            if (!newContractProjectId || newContractProjectId === "none") {
                 toast.error("Please select a project to save the new contract.");
                 throw new Error("You must select a project to save the new contract to.");
             }

             const formData = new FormData();
             formData.append("file", contractFiles[0]);
             formData.append("project_id", newContractProjectId);
             formData.append("user_id", user.id); 
             
             const uploadRes = await fetch(`${API_URL}/contracts/upload`, {
                 method: "POST",
                 body: formData
             });
             
             if (!uploadRes.ok) {
                 const errData = await uploadRes.json();
                 throw new Error(errData.detail || "Failed to upload and process contract");
             }
             
             const uploadData = await uploadRes.json();
             finalContractId = uploadData.contract_id;
             finalProjectId = newContractProjectId;
             toast.success("Contract uploaded and processed successfully!");

        } else if (selectedExistingContracts.length > 0) {
            finalContractId = selectedExistingContracts[0];
            // Find project ID for this contract
            const contract = existingContracts.find(c => c.id === finalContractId);
            if (contract) finalProjectId = contract.project_id;
        }

        // 2. Determine Statement ID
        if (royaltyStatementFiles.length > 0) {
             const formData = new FormData();
             formData.append("file", royaltyStatementFiles[0]);
             formData.append("artist_id", artistId);
             formData.append("category", "royalty_statement");
             // Associate with contract's project if available
             if (finalProjectId) formData.append("project_id", finalProjectId);
             
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

        if (!finalContractId || !finalStatementId) {
             throw new Error("Could not determine contract or statement to process.");
        }

        // 3. Call Calculate Endpoint
        const calcRes = await fetch(`${API_URL}/oneclick/calculate-royalties`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                contract_id: finalContractId,
                user_id: user.id,
                project_id: finalProjectId,
                royalty_statement_file_id: finalStatementId
            })
        });

        if (!calcRes.ok) {
            const errData = await calcRes.json();
            throw new Error(errData.detail || "Calculation failed");
        }

        const result: CalculationResult = await calcRes.json();
        setCalculationResult(result);
        toast.success("Royalties calculated successfully!");

        // Clear uploaded files from state
        setContractFiles([]);
        setRoyaltyStatementFiles([]);

    } catch (error: any) {
        console.error("Error:", error);
        setError(error.message || "An error occurred.");
        toast.error(error.message || "An error occurred during processing.");
    } finally {
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
              <Tabs defaultValue="upload" className="w-full">
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
                  {!selectedProject ? (
                    <div className="space-y-4">
                      <p className="text-sm font-medium text-foreground">Select a Project:</p>
                      <div className="grid gap-3">
                        {projects.length > 0 ? projects.map((project) => (
                          <div key={project.id} className="flex items-center gap-3 p-4 border border-border rounded-lg hover:bg-secondary/50 transition-colors cursor-pointer" onClick={() => setSelectedProject(project.id)}>
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
                          Contracts in {projects.find(p => p.id === selectedProject)?.name}:
                        </p>
                        <Button variant="ghost" size="sm" onClick={() => setSelectedProject(null)} className="text-xs h-8">
                          Change Project
                        </Button>
                      </div>
                      
                      {isLoadingProjectFiles ? (
                        <div className="text-center py-8">
                          <Loader2 className="inline-block animate-spin h-8 w-8 text-primary mb-2" />
                          <p className="text-sm text-muted-foreground">Loading contracts...</p>
                        </div>
                      ) : existingContracts.length > 0 ? (
                        <div className="space-y-2">
                          {existingContracts.map((contract) => (
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
              <Tabs defaultValue="upload" className="w-full">
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
                  {!selectedProject ? (
                    <div className="space-y-4">
                      <p className="text-sm font-medium text-foreground">Select a Project:</p>
                      <div className="grid gap-3">
                        {projects.length > 0 ? projects.map((project) => (
                          <div key={project.id} className="flex items-center gap-3 p-4 border border-border rounded-lg hover:bg-secondary/50 transition-colors cursor-pointer" onClick={() => setSelectedProject(project.id)}>
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
                          Royalty Statements in {projects.find(p => p.id === selectedProject)?.name}:
                        </p>
                        <Button variant="ghost" size="sm" onClick={() => setSelectedProject(null)} className="text-xs h-8">
                          Change Project
                        </Button>
                      </div>
                      
                      {isLoadingProjectFiles ? (
                        <div className="text-center py-8">
                          <Loader2 className="inline-block animate-spin h-8 w-8 text-primary mb-2" />
                          <p className="text-sm text-muted-foreground">Loading royalty statements...</p>
                        </div>
                      ) : existingRoyaltyStatements.length > 0 ? (
                        <div className="space-y-2">
                          {existingRoyaltyStatements.map((statement) => (
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
