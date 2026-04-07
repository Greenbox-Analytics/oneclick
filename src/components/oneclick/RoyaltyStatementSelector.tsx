import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Upload, FileText, X, Receipt, Folder, Loader2, Search, Plus, FileSpreadsheet } from "lucide-react";
import { toast } from "sonner";

interface Project {
  id: string;
  name: string;
}

interface ArtistFile {
  id: string;
  file_name: string;
  created_at: string;
  folder_category: string;
  file_path: string;
  project_id: string;
}

interface RoyaltyStatementSelectorProps {
  royaltyStatementTabValue: string;
  setRoyaltyStatementTabValue: (value: string) => void;
  royaltyStatementFile: File | null;
  setRoyaltyStatementFile: React.Dispatch<React.SetStateAction<File | null>>;
  newRoyaltyStatementProjectId: string;
  setNewRoyaltyStatementProjectId: (value: string) => void;
  projects: Project[];
  setIsCreateProjectOpen: (value: boolean) => void;
  selectedRoyaltyStatementProject: string | null;
  setSelectedRoyaltyStatementProject: (value: string | null) => void;
  royaltySearchTerm: string;
  setRoyaltySearchTerm: (value: string) => void;
  isLoadingProjectFiles: boolean;
  existingRoyaltyStatements: ArtistFile[];
  selectedExistingRoyaltyStatement: string | null;
  setSelectedExistingRoyaltyStatement: React.Dispatch<React.SetStateAction<string | null>>;
  fetchProjectFilesForValidation: (projectId: string) => Promise<ArtistFile[]>;
}

const normalizeFileName = (name: string) => name.trim().toLowerCase();

const showPersistentDuplicateToast = (message: string) => {
  toast.error(message, {
    duration: Infinity,
    closeButton: true,
    style: {
      background: "hsl(var(--destructive))",
      color: "hsl(var(--destructive-foreground))",
      border: "1px solid hsl(var(--destructive))",
    },
  });
};

const RoyaltyStatementSelector = ({
  royaltyStatementTabValue,
  setRoyaltyStatementTabValue,
  royaltyStatementFile,
  setRoyaltyStatementFile,
  newRoyaltyStatementProjectId,
  setNewRoyaltyStatementProjectId,
  projects,
  setIsCreateProjectOpen,
  selectedRoyaltyStatementProject,
  setSelectedRoyaltyStatementProject,
  royaltySearchTerm,
  setRoyaltySearchTerm,
  isLoadingProjectFiles,
  existingRoyaltyStatements,
  selectedExistingRoyaltyStatement,
  setSelectedExistingRoyaltyStatement,
  fetchProjectFilesForValidation,
}: RoyaltyStatementSelectorProps) => {
  const handleRoyaltyStatementFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    const projectFileNames = new Set<string>();

    if (newRoyaltyStatementProjectId && newRoyaltyStatementProjectId !== "none") {
      try {
        const projectFiles = await fetchProjectFilesForValidation(newRoyaltyStatementProjectId);
        projectFiles.forEach(f => projectFileNames.add(normalizeFileName(f.file_name)));
      } catch (err) {
        console.error("Error checking royalty statement duplicates:", err);
      }
    }

    if (projectFileNames.has(normalizeFileName(file.name))) {
      showPersistentDuplicateToast(`Duplicate file name blocked: ${file.name}`);
    } else {
      setRoyaltyStatementFile(file);
    }

    e.target.value = "";
  };

  const handleRemoveRoyaltyStatementFile = () => {
    setRoyaltyStatementFile(null);
  };

  const handleToggleExistingRoyaltyStatement = (fileId: string) => {
    setSelectedExistingRoyaltyStatement(prev => prev === fileId ? null : fileId);
  };

  return (
    <Card data-walkthrough="oneclick-royalty">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Receipt className="w-5 h-5 text-primary" />
          Upload Royalty Statement
        </CardTitle>
        <CardDescription>Upload or select royalty statement documents</CardDescription>
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

            {royaltyStatementFile && (
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

            {royaltyStatementFile && (
              <div className="space-y-2 mt-4">
                <div className="flex items-center justify-between p-2 border border-border rounded-lg bg-secondary/50">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <FileText className="w-4 h-4 text-primary flex-shrink-0" />
                    <p className="text-xs font-medium text-foreground truncate">{royaltyStatementFile.name}</p>
                  </div>
                  <Button variant="ghost" size="sm" onClick={() => handleRemoveRoyaltyStatementFile()} className="text-destructive hover:text-destructive flex-shrink-0">
                    <X className="w-3 h-3" />
                  </Button>
                </div>
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
                              checked={selectedExistingRoyaltyStatement === statement.id}
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
  );
};

export default RoyaltyStatementSelector;
