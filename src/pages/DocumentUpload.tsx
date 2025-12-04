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
import { Music, ArrowLeft, Upload, FileText, X, FileSignature, Receipt, Users, DollarSign, Download, FileSpreadsheet, CheckCircle2, Folder } from "lucide-react";
// React Router hooks for navigation and getting URL parameters
import { useNavigate, useParams } from "react-router-dom";
// Recharts for pie chart
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
// xlsx library for Excel export
import * as XLSX from "xlsx";
import { toast } from "sonner";

// Backend API URL
const API_URL = "http://localhost:8000";

// Type declaration for html-to-image
declare module 'html-to-image' {
  export function toPng(node: HTMLElement, options?: {
    quality?: number;
    pixelRatio?: number;
    backgroundColor?: string;
  }): Promise<string>;
}

// Type definitions
interface RoyaltyBreakdown {
  songName: string;
  contributorName: string;
  role: string;
  royaltyPercentage: number;
  amount: number;
}

interface RoyaltyResults {
  songTitle: string;
  totalContributors: number;
  totalRevenue: number;
  breakdown: RoyaltyBreakdown[];
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

const DocumentUpload = () => {
  const navigate = useNavigate();
  const { artistId } = useParams<{ artistId: string }>();
  
  // File Upload State
  const [contractFiles, setContractFiles] = useState<File[]>([]);
  const [royaltyStatementFiles, setRoyaltyStatementFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  
  // Calculation Results State
  const [royaltyResults, setRoyaltyResults] = useState<RoyaltyResults | null>(null);

  // Existing Documents Selection State
  const [selectedExistingContracts, setSelectedExistingContracts] = useState<string[]>([]); // Array of file_paths
  const [selectedExistingRoyaltyStatements, setSelectedExistingRoyaltyStatements] = useState<string[]>([]); // Array of file_paths

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
      fetch(`${API_URL}/files/artist/${artistId}/category/royalty_statement`)
        .then(res => res.json())
        .then(data => setExistingRoyaltyStatements(data))
        .catch(err => console.error("Error fetching royalty statements:", err));
    }
  }, [artistId]);

  // Fetch Contracts when Project Selected
  useEffect(() => {
    if (selectedProject) {
      // Clear previous selections when changing projects
      setSelectedExistingContracts([]);
      
      setIsLoadingProjectFiles(true);
      fetch(`${API_URL}/files/${selectedProject}`)
        .then(res => res.json())
        .then((data: ArtistFile[]) => {
            // Filter only contracts
            const contracts = data.filter(f => f.folder_category === 'contract');
            setExistingContracts(contracts);
            setIsLoadingProjectFiles(false);
        })
        .catch(err => {
          console.error("Error fetching project files:", err);
          setIsLoadingProjectFiles(false);
        });
    } else {
        setExistingContracts([]);
        setSelectedExistingContracts([]);
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

  // Toggle selection store file_path instead of ID for easier backend processing
  const handleToggleExistingContract = (filePath: string) => {
    setSelectedExistingContracts(prev =>
      prev.includes(filePath) ? prev.filter(p => p !== filePath) : [...prev, filePath]
    );
  };

  const handleToggleExistingRoyaltyStatement = (filePath: string) => {
    setSelectedExistingRoyaltyStatements(prev =>
      prev.includes(filePath) ? prev.filter(p => p !== filePath) : [...prev, filePath]
    );
  };

  const handleCalculateRoyalties = async () => {
    if (!artistId) return;

    const hasContracts = contractFiles.length > 0 || selectedExistingContracts.length > 0;
    const hasRoyaltyStatements = royaltyStatementFiles.length > 0 || selectedExistingRoyaltyStatements.length > 0;
    
    if (!hasContracts || !hasRoyaltyStatements) {
        toast.error("Please provide both contracts and royalty statements.");
        return;
    }

    setIsUploading(true);
    
    try {
        const uploadedContractPaths: string[] = [];
        const uploadedRoyaltyPaths: string[] = [];

        // 1. Upload New Contracts
        for (const file of contractFiles) {
            const formData = new FormData();
            formData.append("file", file);
            formData.append("artist_id", artistId);
            formData.append("category", "contract");
            if (newContractProjectId && newContractProjectId !== "none") {
                formData.append("project_id", newContractProjectId);
            }
            
            const res = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
            if (!res.ok) throw new Error("Failed to upload contract");
            const data = await res.json();
            // Backend returns { "status": "success", "file": {...} }
            uploadedContractPaths.push(data.file.file_path);
        }

        // 2. Upload New Royalty Statements
        for (const file of royaltyStatementFiles) {
            const formData = new FormData();
            formData.append("file", file);
            formData.append("artist_id", artistId);
            formData.append("category", "royalty_statement");
            
            const res = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
            if (!res.ok) throw new Error("Failed to upload royalty statement");
            const data = await res.json();
            uploadedRoyaltyPaths.push(data.file.file_path);
        }

        // 3. Combine with Selected Existing Files
        const allContractPaths = [...selectedExistingContracts, ...uploadedContractPaths];
        const allRoyaltyPaths = [...selectedExistingRoyaltyStatements, ...uploadedRoyaltyPaths];

        // 4. Call Calculate Endpoint
        // We need to send these lists as form data or JSON. Since the backend expects Form parameters (List[str]),
        // we can construct formData with multiple values for the same key.
        const calcFormData = new FormData();
        allContractPaths.forEach(path => calcFormData.append("contract_files", path));
        allRoyaltyPaths.forEach(path => calcFormData.append("royalty_files", path));

        const calcRes = await fetch(`${API_URL}/calculate`, {
            method: "POST",
            body: calcFormData
        });

        if (!calcRes.ok) throw new Error("Calculation failed");
        
        const results: RoyaltyResults = await calcRes.json();
        setRoyaltyResults(results);
        toast.success("Royalties calculated successfully!");

        // Clear uploaded files from state (optional, user might want to re-run)
        setContractFiles([]);
        setRoyaltyStatementFiles([]);

    } catch (error) {
        console.error("Error:", error);
        toast.error("An error occurred during processing.");
    } finally {
        setIsUploading(false);
    }
  };

  const handleExportCSV = () => {
    if (!royaltyResults) return;
    const headers = ["Song Name", "Contributor Name", "Role", "Royalty Share %", "Amount"];
    const rows = royaltyResults.breakdown.map(row => [
      row.songName,
      row.contributorName,
      row.role,
      `${row.royaltyPercentage}%`,
      `$${row.amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
    ]);
    const csvContent = [headers.join(","), ...rows.map(row => row.map(cell => `"${cell}"`).join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `royalty-breakdown-${royaltyResults.songTitle.replace(/\s+/g, "-")}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleExportExcel = () => {
    if (!royaltyResults) return;
    const excelData = [
      ["Song Name", "Contributor Name", "Role", "Royalty Share %", "Amount"],
      ...royaltyResults.breakdown.map(row => [
        row.songName, row.contributorName, row.role, row.royaltyPercentage, row.amount
      ])
    ];
    const ws = XLSX.utils.aoa_to_sheet(excelData);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Royalty Breakdown");
    XLSX.writeFile(wb, `royalty-breakdown-${royaltyResults.songTitle.replace(/\s+/g, "-")}.xlsx`);
  };

  const handleDownloadChart = async () => {
    if (!chartContentRef.current || !royaltyResults) return;
    
    try {
      // Dynamically import html-to-image
      const { toPng } = await import('html-to-image');
      
      const dataUrl = await toPng(chartContentRef.current, {
        quality: 1,
        pixelRatio: 2,
        backgroundColor: '#ffffff'
      });
      
      const link = document.createElement('a');
      link.download = `royalty-distribution-${royaltyResults.songTitle.replace(/\s+/g, "-")}.png`;
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
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
                <Music className="w-6 h-6 text-primary-foreground" />
              </div>
              <h1 className="text-2xl font-bold text-foreground">Msanii AI</h1>
            </div>
            <Button variant="outline" onClick={() => navigate("/tools/oneclick")}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Artist Selection
            </Button>
          </div>
        </header>
        <main className="container mx-auto px-4 py-8 max-w-4xl">
          <div className="flex flex-col items-center justify-center min-h-[60vh]">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary mb-4"></div>
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
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii AI</h1>
          </div>
          <Button variant="outline" onClick={() => navigate("/tools/oneclick")}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Artist Selection
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-foreground mb-2">Upload Documents</h2>
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
                    <p className="text-muted-foreground mb-4 text-xs">PDF or Word DOCX files only</p>
                    <Input
                      id="contract-upload"
                      type="file"
                      multiple
                      accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                      onChange={handleContractFileChange}
                      className="hidden"
                    />
                    <label htmlFor="contract-upload">
                      <Button variant="outline" size="sm" asChild>
                        <span><Upload className="w-4 h-4 mr-2" />Select Files</span>
                      </Button>
                    </label>
                  </div>

                  {contractFiles.length > 0 && (
                    <div className="space-y-2 pt-2 border-t border-border">
                      <div className="flex items-center gap-2">
                        <Folder className="w-4 h-4 text-muted-foreground" />
                        <label htmlFor="project-select" className="text-sm font-medium text-foreground">
                          Save to Project (Optional)
                        </label>
                      </div>
                      <Select value={newContractProjectId} onValueChange={setNewContractProjectId}>
                        <SelectTrigger id="project-select" className="w-full">
                          <SelectValue placeholder="Select a project..." />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">None (Don't save to project)</SelectItem>
                          {projects.map((project) => (
                            <SelectItem key={project.id} value={project.id}>{project.name}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">Associate these files with an existing project.</p>
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
                          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                          <p className="text-sm text-muted-foreground mt-2">Loading contracts...</p>
                        </div>
                      ) : existingContracts.length > 0 ? (
                        <div className="space-y-2">
                          {existingContracts.map((contract) => (
                            <div key={contract.id} className="flex items-center justify-between p-3 border border-border rounded-lg hover:bg-secondary/50 transition-colors">
                              <div className="flex items-center gap-3 flex-1">
                                <Checkbox
                                  id={`existing-contract-${contract.id}`}
                                  checked={selectedExistingContracts.includes(contract.file_path)}
                                  onCheckedChange={() => handleToggleExistingContract(contract.file_path)}
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
                    <p className="text-muted-foreground mb-4 text-xs">Excel or CSV files only</p>
                    <Input
                      id="royalty-upload"
                      type="file"
                      multiple
                      accept=".xlsx,.xls,.csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel,text/csv"
                      onChange={handleRoyaltyStatementFileChange}
                      className="hidden"
                    />
                    <label htmlFor="royalty-upload">
                      <Button variant="outline" size="sm" asChild>
                        <span><Upload className="w-4 h-4 mr-2" />Select Files</span>
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
                  {existingRoyaltyStatements.length > 0 ? (
                    <div className="space-y-2">
                      <p className="text-sm font-medium text-foreground mb-2">Available Royalty Statements:</p>
                      {existingRoyaltyStatements.map((statement) => (
                        <div key={statement.id} className="flex items-center justify-between p-3 border border-border rounded-lg hover:bg-secondary/50 transition-colors">
                          <div className="flex items-center gap-3 flex-1">
                            <Checkbox
                              id={`existing-royalty-${statement.id}`}
                              checked={selectedExistingRoyaltyStatements.includes(statement.file_path)}
                              onCheckedChange={() => handleToggleExistingRoyaltyStatement(statement.file_path)}
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
                    <p className="text-sm text-muted-foreground text-center py-4">No royalty statements found.</p>
                  )}
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>

        <div className="flex gap-3 justify-center">
          <Button
            onClick={handleCalculateRoyalties}
            disabled={((contractFiles.length === 0 && selectedExistingContracts.length === 0) || 
                       (royaltyStatementFiles.length === 0 && selectedExistingRoyaltyStatements.length === 0)) || 
                       isUploading}
          >
            {isUploading ? "Calculating..." : "Calculate Royalties"}
          </Button>
          <Button variant="outline" onClick={() => navigate("/tools/oneclick")} disabled={isUploading}>
            Cancel
          </Button>
        </div>

        {royaltyResults && (
          <div className="mt-8 space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-foreground mb-2">Royalty Calculation Results</h2>
              <p className="text-muted-foreground">Breakdown of royalty distribution</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Song Title</CardTitle>
                  <Music className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent><div className="text-2xl font-bold text-foreground">{royaltyResults.songTitle}</div></CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Contributors</CardTitle>
                  <Users className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent><div className="text-2xl font-bold text-foreground">{royaltyResults.totalContributors}</div></CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Revenue</CardTitle>
                  <DollarSign className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-foreground">
                    ${royaltyResults.totalRevenue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
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
                        data={royaltyResults.breakdown.sort((a, b) => b.royaltyPercentage - a.royaltyPercentage).map((row, index, arr) => ({
                          name: row.contributorName,
                          value: row.royaltyPercentage,
                          amount: row.amount,
                          fill: `hsl(150, ${50 + (index / (arr.length || 1)) * 10}%, ${25 + (index / (arr.length || 1)) * 40}%)`
                        }))}
                        cx="50%" 
                        cy="50%" 
                        outerRadius={130} 
                        dataKey="value"
                        nameKey="name"
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                        labelLine={true}
                        style={{ fontSize: '11px' }}
                      />
                      <Tooltip 
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="rounded-lg border bg-background p-3 shadow-sm">
                                <div className="font-bold text-foreground mb-1">{data.name}</div>
                                <div className="text-sm">
                                  <div className="text-muted-foreground">
                                    Royalty Share: <span className="text-foreground font-semibold">{data.value}%</span>
                                  </div>
                                  <div className="text-muted-foreground">
                                    Amount: <span className="text-foreground font-semibold">${data.amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                                  </div>
                                </div>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
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
                                <TableHead>Song</TableHead><TableHead>Contributor</TableHead><TableHead>Role</TableHead><TableHead>Share %</TableHead><TableHead>Amount</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {royaltyResults.breakdown.map((row, i) => (
                                <TableRow key={i}>
                                    <TableCell>{row.songName}</TableCell>
                                    <TableCell>{row.contributorName}</TableCell>
                                    <TableCell>{row.role}</TableCell>
                                    <TableCell>{row.royaltyPercentage}%</TableCell>
                                    <TableCell>${row.amount.toLocaleString()}</TableCell>
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

export default DocumentUpload;
