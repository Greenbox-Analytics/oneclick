import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Upload, FileText, X, FileSignature, Folder, Loader2, Search, Plus } from "lucide-react";
import { toast } from "sonner";
import { type Work } from "@/hooks/useRegistry";
import { type WorkFileLink } from "@/hooks/useWorkFiles";

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

interface ContractSelectorProps {
  contractTabValue: string;
  setContractTabValue: (value: string) => void;
  contractFiles: File[];
  setContractFiles: React.Dispatch<React.SetStateAction<File[]>>;
  newContractProjectId: string;
  setNewContractProjectId: (value: string) => void;
  projects: Project[];
  setIsCreateProjectOpen: (value: boolean) => void;
  selectedContractProject: string | null;
  setSelectedContractProject: (value: string | null) => void;
  contractSearchTerm: string;
  setContractSearchTerm: (value: string) => void;
  isLoadingProjectFiles: boolean;
  existingContracts: ArtistFile[];
  selectedExistingContracts: string[];
  setSelectedExistingContracts: React.Dispatch<React.SetStateAction<string[]>>;
  isLoadingWorks: boolean;
  artistWorks: Work[];
  selectedWorkId: string | null;
  setSelectedWorkId: (value: string | null) => void;
  workFiles: WorkFileLink[];
  setWorkFiles: (value: WorkFileLink[]) => void;
  loadingWorkFiles: boolean;
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

const ContractSelector = ({
  contractTabValue,
  setContractTabValue,
  contractFiles,
  setContractFiles,
  newContractProjectId,
  setNewContractProjectId,
  projects,
  setIsCreateProjectOpen,
  selectedContractProject,
  setSelectedContractProject,
  contractSearchTerm,
  setContractSearchTerm,
  isLoadingProjectFiles,
  existingContracts,
  selectedExistingContracts,
  setSelectedExistingContracts,
  isLoadingWorks,
  artistWorks,
  selectedWorkId,
  setSelectedWorkId,
  workFiles,
  setWorkFiles,
  loadingWorkFiles,
  fetchProjectFilesForValidation,
}: ContractSelectorProps) => {
  const handleContractFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const incomingFiles = Array.from(files);
    const blockedNames = new Set<string>();
    const existingSelectedNames = new Set(contractFiles.map(file => normalizeFileName(file.name)));
    const seenInBatch = new Set<string>();
    const projectFileNames = new Set<string>();

    if (newContractProjectId && newContractProjectId !== "none") {
      try {
        const projectFiles = await fetchProjectFilesForValidation(newContractProjectId);
        projectFiles.forEach(file => projectFileNames.add(normalizeFileName(file.file_name)));
      } catch (err) {
        console.error("Error checking contract duplicates:", err);
      }
    }

    const allowedFiles: File[] = [];
    incomingFiles.forEach(file => {
      const normalizedName = normalizeFileName(file.name);

      if (seenInBatch.has(normalizedName) || existingSelectedNames.has(normalizedName) || projectFileNames.has(normalizedName)) {
        blockedNames.add(file.name);
        return;
      }

      seenInBatch.add(normalizedName);
      allowedFiles.push(file);
    });

    if (blockedNames.size > 0) {
      showPersistentDuplicateToast(`Duplicate file name(s) blocked: ${Array.from(blockedNames).join(", ")}`);
    }

    if (allowedFiles.length > 0) {
      setContractFiles(prev => [...prev, ...allowedFiles]);
    }

    e.target.value = "";
  };

  const handleRemoveContractFile = (index: number) => {
    setContractFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleToggleExistingContract = (fileId: string) => {
    setSelectedExistingContracts(prev =>
      prev.includes(fileId) ? prev.filter(p => p !== fileId) : [...prev, fileId]
    );
  };

  return (
    <Card data-walkthrough="oneclick-contracts">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileSignature className="w-5 h-5 text-primary" />
          Upload Contracts
        </CardTitle>
        <CardDescription>Upload or select contract documents</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Tabs value={contractTabValue} onValueChange={setContractTabValue} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="upload">Upload New</TabsTrigger>
            <TabsTrigger value="existing">Select Existing</TabsTrigger>
            <TabsTrigger value="works">From Works</TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-4 mt-4">
            <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
              <FileSignature className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
              <p className="text-foreground font-medium mb-2 text-sm">Upload Contract or Split Sheet</p>
              <div className="flex items-center justify-center gap-2 mb-4">
                <div className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 rounded-md">
                  <FileText className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
                  <span className="text-xs font-medium text-blue-700 dark:text-blue-300">PDF accepted</span>
                </div>
              </div>
              <Input
                id="contract-upload"
                type="file"
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
                <p className="text-xs text-muted-foreground">Select a project to save the document.</p>
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
                    Documents in {projects.find(p => p.id === selectedContractProject)?.name}:
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
                    <p className="text-sm text-muted-foreground">Loading documents...</p>
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
                       <p className="text-sm text-muted-foreground text-center py-4">No matching documents found.</p>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground text-center py-4">No contracts or split sheets found in this project.</p>
                )}
              </div>
            )}
          </TabsContent>

          <TabsContent value="works" className="space-y-4 mt-4">
            {isLoadingWorks ? (
              <div className="text-center py-8">
                <Loader2 className="inline-block animate-spin h-8 w-8 text-primary mb-2" />
                <p className="text-sm text-muted-foreground">Loading works...</p>
              </div>
            ) : artistWorks.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">No registered works found for this artist.</p>
            ) : !selectedWorkId ? (
              <div className="space-y-4">
                <p className="text-sm font-medium text-foreground">Select a Work:</p>
                <div className="grid gap-3 max-h-[300px] overflow-y-auto">
                  {artistWorks.map((work: Work) => (
                    <div
                      key={work.id}
                      className="flex items-center gap-3 p-4 border border-border rounded-lg hover:bg-secondary/50 transition-colors cursor-pointer"
                      onClick={() => setSelectedWorkId(work.id)}
                    >
                      <FileText className="w-5 h-5 text-primary" />
                      <div>
                        <p className="font-medium text-foreground">{work.title}</p>
                        <p className="text-xs text-muted-foreground">{work.work_type}{work.release_date ? ` \u00b7 ${new Date(work.release_date).toLocaleDateString()}` : ''}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium text-foreground">
                    Files in {artistWorks.find((w: Work) => w.id === selectedWorkId)?.title}:
                  </p>
                  <Button variant="ghost" size="sm" onClick={() => { setSelectedWorkId(null); setWorkFiles([]); }} className="text-xs h-8">
                    Change Work
                  </Button>
                </div>
                {loadingWorkFiles ? (
                  <div className="text-center py-8">
                    <Loader2 className="inline-block animate-spin h-8 w-8 text-primary mb-2" />
                    <p className="text-sm text-muted-foreground">Loading files...</p>
                  </div>
                ) : workFiles.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-4">No files linked to this work.</p>
                ) : (
                  <div className="space-y-2 max-h-[300px] overflow-y-auto">
                    {workFiles.map((link: WorkFileLink) => {
                      const file = link.project_files;
                      if (!file) return null;
                      return (
                        <div key={link.id} className="flex items-center justify-between p-3 border border-border rounded-lg hover:bg-secondary/50 transition-colors">
                          <div className="flex items-center gap-3 flex-1">
                            <Checkbox
                              id={`work-file-${file.id}`}
                              checked={selectedExistingContracts.includes(file.id)}
                              onCheckedChange={() => handleToggleExistingContract(file.id)}
                            />
                            <label htmlFor={`work-file-${file.id}`} className="flex items-center gap-2 flex-1 cursor-pointer">
                              <FileText className="w-4 h-4 text-primary" />
                              <div>
                                <p className="text-sm font-medium text-foreground">{file.file_name}</p>
                                <p className="text-xs text-muted-foreground">{file.folder_category}</p>
                              </div>
                            </label>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default ContractSelector;
