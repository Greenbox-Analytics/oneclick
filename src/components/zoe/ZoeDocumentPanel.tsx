import { RefObject } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { ArrowLeft, Upload, Trash2, ChevronDown, Plus, Loader2, FileText, FolderOpen, Users, GripVertical } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { ContractUploadModal } from "@/components/ContractUploadModal";
import { cn } from "@/lib/utils";
import { type Work } from "@/hooks/useRegistry";
import { type WorkFileLink } from "@/hooks/useWorkFiles";
import type { Artist, Project, Contract } from "@/components/zoe/types";

interface ZoeDocumentPanelProps {
  sidebarRef: RefObject<HTMLDivElement>;
  sidebarOpen: boolean;
  sidebarWidth: number;
  isResizing: boolean;
  onMouseDown: (e: React.MouseEvent) => void;
  artists: Artist[];
  selectedArtist: string;
  onArtistChange: (value: string) => void;
  projects: Project[];
  selectedProject: string;
  onProjectChange: (value: string) => void;
  contracts: Contract[];
  selectedContracts: string[];
  onSelectedContractsChange: (contracts: string[]) => void;
  contractsOpen: boolean;
  onContractsOpenChange: (open: boolean) => void;
  sharedWorks: Work[];
  isLoadingSharedWorks: boolean;
  sharedWorksOpen: boolean;
  onSharedWorksOpenChange: (open: boolean) => void;
  selectedSharedWork: string | null;
  onSelectedSharedWorkChange: (workId: string | null) => void;
  sharedWorkFiles: WorkFileLink[];
  loadingWorkFiles: boolean;
  onSharedWorkFilesReset: () => void;
  uploadModalOpen: boolean;
  onUploadModalOpenChange: (open: boolean) => void;
  onUploadComplete: () => void;
  isCreateProjectOpen: boolean;
  onCreateProjectOpenChange: (open: boolean) => void;
  newProjectNameInput: string;
  onNewProjectNameInputChange: (value: string) => void;
  isCreatingProject: boolean;
  onCreateProject: () => void;
  deleteDialogOpen: boolean;
  onDeleteDialogOpenChange: (open: boolean) => void;
  contractToDelete: Contract | null;
  deleting: boolean;
  onDeleteClick: (contract: Contract) => void;
  onDeleteConfirm: () => void;
}

export function ZoeDocumentPanel({
  sidebarRef,
  sidebarOpen,
  sidebarWidth,
  isResizing,
  onMouseDown,
  artists,
  selectedArtist,
  onArtistChange,
  projects,
  selectedProject,
  onProjectChange,
  contracts,
  selectedContracts,
  onSelectedContractsChange,
  contractsOpen,
  onContractsOpenChange,
  sharedWorks,
  isLoadingSharedWorks,
  sharedWorksOpen,
  onSharedWorksOpenChange,
  selectedSharedWork,
  onSelectedSharedWorkChange,
  sharedWorkFiles,
  loadingWorkFiles,
  onSharedWorkFilesReset,
  uploadModalOpen,
  onUploadModalOpenChange,
  onUploadComplete,
  isCreateProjectOpen,
  onCreateProjectOpenChange,
  newProjectNameInput,
  onNewProjectNameInputChange,
  isCreatingProject,
  onCreateProject,
  deleteDialogOpen,
  onDeleteDialogOpenChange,
  contractToDelete,
  deleting,
  onDeleteClick,
  onDeleteConfirm,
}: ZoeDocumentPanelProps) {
  return (
    <>
    <aside
      ref={sidebarRef}
      data-walkthrough="zoe-sidebar"
      className={cn(
        "border-r border-border bg-muted/30 flex-shrink-0 overflow-hidden relative",
        !isResizing && "transition-all duration-300 ease-in-out",
        !sidebarOpen && "!w-0"
      )}
      style={{ width: sidebarOpen ? sidebarWidth : 0 }}
    >
      <div className="h-full flex flex-col" style={{ width: sidebarWidth }}>
        {/* Resize Handle */}
        <div
          className={cn(
            "absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-primary/20 active:bg-primary/30 z-10 group",
            isResizing && "bg-primary/30"
          )}
          onMouseDown={onMouseDown}
        >
          <div className="absolute right-0 top-1/2 -translate-y-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
            <GripVertical className="w-3 h-3 text-muted-foreground" />
          </div>
        </div>
        <ScrollArea className="flex-1">
          <div className="p-4 space-y-6">
            {/* Artist Selection */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                <Users className="w-3.5 h-3.5" />
                Artist
              </div>
              <Select value={selectedArtist} onValueChange={onArtistChange}>
                <SelectTrigger className="w-full bg-background">
                  <SelectValue placeholder="Select artist..." />
                </SelectTrigger>
                <SelectContent>
                  {artists.map((artist) => (
                    <SelectItem key={artist.id} value={artist.id}>
                      {artist.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Project Selection */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                <FolderOpen className="w-3.5 h-3.5" />
                Project
              </div>
              <div className="flex items-center gap-2">
                <Select value={selectedProject} onValueChange={onProjectChange} disabled={!selectedArtist}>
                  <SelectTrigger className="flex-1 bg-background">
                    <SelectValue placeholder="Select project..." />
                  </SelectTrigger>
                  <SelectContent>
                    {projects.map((project) => (
                      <SelectItem key={project.id} value={project.id}>
                        {project.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button
                  variant="outline"
                  size="icon"
                  className="h-9 w-9 flex-shrink-0"
                  onClick={() => onCreateProjectOpenChange(true)}
                  disabled={!selectedArtist}
                  title="Create New Project"
                >
                  <Plus className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Documents Section (Contracts & Split Sheets) */}
            {selectedProject && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    <FileText className="w-3.5 h-3.5" />
                    Documents {contracts.length > 0 && `(${contracts.length})`}
                  </div>
                  <Button
                    data-walkthrough="zoe-upload"
                    variant="ghost"
                    size="sm"
                    onClick={() => onUploadModalOpenChange(true)}
                    className="h-7 text-xs px-2"
                  >
                    <Upload className="w-3 h-3 mr-1" />
                    Upload
                  </Button>
                </div>

                {/* Collapsible documents list */}
                <Collapsible open={contractsOpen} onOpenChange={onContractsOpenChange}>
                  <CollapsibleTrigger asChild>
                    <Button variant="secondary" size="sm" className="w-full justify-between h-9">
                      {selectedContracts.length > 0
                        ? `${selectedContracts.length} selected`
                        : "All documents"}
                      <ChevronDown className={cn("ml-2 h-4 w-4 transition-transform", contractsOpen && "rotate-180")} />
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <div className="bg-background rounded-lg border mt-2 max-h-64 overflow-y-auto">
                      {contracts.length > 0 ? (
                        <div className="p-2 space-y-1">
                          {selectedContracts.length > 0 && (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="w-full h-7 text-xs justify-start text-muted-foreground hover:text-foreground"
                              onClick={() => onSelectedContractsChange([])}
                            >
                              Clear selection
                            </Button>
                          )}
                          {/* Contracts group */}
                          {contracts.filter(c => c.folder_category === "contract").length > 0 && (
                            <>
                              <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider px-2 pt-1">Contracts</p>
                              {contracts.filter(c => c.folder_category === "contract").map((contract) => (
                                <div key={contract.id} className="flex items-center justify-between gap-2 group px-2 py-1.5 rounded hover:bg-muted/50">
                                  <div className="flex items-center space-x-2 flex-1 min-w-0">
                                    <Checkbox
                                      id={contract.id}
                                      checked={selectedContracts.includes(contract.id)}
                                      onCheckedChange={(checked) => {
                                        if (checked) {
                                          onSelectedContractsChange([...selectedContracts, contract.id]);
                                        } else {
                                          onSelectedContractsChange(selectedContracts.filter(id => id !== contract.id));
                                        }
                                      }}
                                    />
                                    <label
                                      htmlFor={contract.id}
                                      className="text-xs font-medium leading-none cursor-pointer truncate"
                                    >
                                      {contract.file_name}
                                    </label>
                                  </div>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => onDeleteClick(contract)}
                                    className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                                  >
                                    <Trash2 className="w-3 h-3 text-destructive" />
                                  </Button>
                                </div>
                              ))}
                            </>
                          )}
                          {/* Split Sheets group */}
                          {contracts.filter(c => c.folder_category === "split_sheet").length > 0 && (
                            <>
                              <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider px-2 pt-2">Split Sheets</p>
                              {contracts.filter(c => c.folder_category === "split_sheet").map((contract) => (
                                <div key={contract.id} className="flex items-center justify-between gap-2 group px-2 py-1.5 rounded hover:bg-muted/50">
                                  <div className="flex items-center space-x-2 flex-1 min-w-0">
                                    <Checkbox
                                      id={contract.id}
                                      checked={selectedContracts.includes(contract.id)}
                                      onCheckedChange={(checked) => {
                                        if (checked) {
                                          onSelectedContractsChange([...selectedContracts, contract.id]);
                                        } else {
                                          onSelectedContractsChange(selectedContracts.filter(id => id !== contract.id));
                                        }
                                      }}
                                    />
                                    <label
                                      htmlFor={contract.id}
                                      className="text-xs font-medium leading-none cursor-pointer truncate"
                                    >
                                      {contract.file_name}
                                    </label>
                                  </div>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => onDeleteClick(contract)}
                                    className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                                  >
                                    <Trash2 className="w-3 h-3 text-destructive" />
                                  </Button>
                                </div>
                              ))}
                            </>
                          )}
                        </div>
                      ) : (
                        <div className="p-4 text-center">
                          <p className="text-xs text-muted-foreground mb-2">No documents yet</p>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => onUploadModalOpenChange(true)}
                            className="h-7 text-xs"
                          >
                            <Upload className="w-3 h-3 mr-1" />
                            Upload First
                          </Button>
                        </div>
                      )}
                    </div>
                  </CollapsibleContent>
                </Collapsible>

                {/* Search scope indicator */}
                <Badge variant="outline" className="w-full justify-center text-xs py-1.5">
                  {selectedContracts.length > 0
                    ? `Searching ${selectedContracts.length} document${selectedContracts.length > 1 ? 's' : ''}`
                    : `All project documents`}
                </Badge>
              </div>
            )}

            {/* Empty state when no project */}
            {!selectedProject && selectedArtist && (
              <div className="rounded-lg border border-dashed border-muted-foreground/25 p-4 text-center">
                <FolderOpen className="w-8 h-8 mx-auto mb-2 text-muted-foreground/50" />
                <p className="text-xs text-muted-foreground">Select a project to view documents</p>
              </div>
            )}

            {!selectedArtist && (
              <div className="rounded-lg border border-dashed border-muted-foreground/25 p-4 text-center">
                <Users className="w-8 h-8 mx-auto mb-2 text-muted-foreground/50" />
                <p className="text-xs text-muted-foreground">Select an artist to get started</p>
              </div>
            )}

            {/* From Shared Works — documents from works where user is a collaborator */}
            <div className="space-y-2">
              <Collapsible open={sharedWorksOpen} onOpenChange={onSharedWorksOpenChange}>
                <CollapsibleTrigger asChild>
                  <Button variant="secondary" size="sm" className="w-full justify-between h-9">
                    <span className="flex items-center gap-2 text-xs font-medium">
                      <Users className="w-3.5 h-3.5" />
                      From Shared Works
                    </span>
                    <ChevronDown className={cn("ml-2 h-4 w-4 transition-transform", sharedWorksOpen && "rotate-180")} />
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <div className="bg-background rounded-lg border mt-2 max-h-72 overflow-y-auto">
                    {isLoadingSharedWorks ? (
                      <div className="p-4 text-center">
                        <Loader2 className="w-4 h-4 animate-spin mx-auto mb-1" />
                        <p className="text-xs text-muted-foreground">Loading shared works...</p>
                      </div>
                    ) : sharedWorks.length === 0 ? (
                      <div className="p-4 text-center">
                        <p className="text-xs text-muted-foreground">No shared works found</p>
                      </div>
                    ) : !selectedSharedWork ? (
                      <div className="p-2 space-y-1">
                        {sharedWorks.map((work: Work) => (
                          <div
                            key={work.id}
                            className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-muted/50 cursor-pointer"
                            onClick={() => onSelectedSharedWorkChange(work.id)}
                          >
                            <FileText className="w-3.5 h-3.5 text-primary flex-shrink-0" />
                            <div className="min-w-0 flex-1">
                              <p className="text-xs font-medium truncate">{work.title}</p>
                              <p className="text-[10px] text-muted-foreground truncate">{work.work_type}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="p-2 space-y-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="w-full h-7 text-xs justify-start text-muted-foreground hover:text-foreground mb-1"
                          onClick={() => { onSelectedSharedWorkChange(null); onSharedWorkFilesReset(); }}
                        >
                          <ArrowLeft className="w-3 h-3 mr-1" />
                          Back to works
                        </Button>
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider px-2">
                          Files in {sharedWorks.find((w: Work) => w.id === selectedSharedWork)?.title}
                        </p>
                        {loadingWorkFiles ? (
                          <div className="py-3 text-center">
                            <Loader2 className="w-4 h-4 animate-spin mx-auto" />
                          </div>
                        ) : sharedWorkFiles.length === 0 ? (
                          <p className="text-xs text-muted-foreground text-center py-3">No files linked to this work</p>
                        ) : (
                          sharedWorkFiles.map((link: WorkFileLink) => {
                            const file = link.project_files;
                            if (!file) return null;
                            return (
                              <div key={link.id} className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-muted/50">
                                <Checkbox
                                  id={`shared-file-${file.id}`}
                                  checked={selectedContracts.includes(file.id)}
                                  onCheckedChange={(checked) => {
                                    if (checked) {
                                      onSelectedContractsChange([...selectedContracts, file.id]);
                                    } else {
                                      onSelectedContractsChange(selectedContracts.filter(id => id !== file.id));
                                    }
                                  }}
                                />
                                <label
                                  htmlFor={`shared-file-${file.id}`}
                                  className="text-xs font-medium leading-none cursor-pointer truncate flex-1"
                                >
                                  {file.file_name}
                                </label>
                              </div>
                            );
                          })
                        )}
                      </div>
                    )}
                  </div>
                </CollapsibleContent>
              </Collapsible>
            </div>
          </div>
        </ScrollArea>
        <p className="text-[11px] text-center text-muted-foreground px-4 py-2 border-t border-border">
          Select documents to ask Zoe about them
        </p>
      </div>
    </aside>

    {selectedProject && (
      <ContractUploadModal
        open={uploadModalOpen}
        onOpenChange={onUploadModalOpenChange}
        projectId={selectedProject}
        onUploadComplete={onUploadComplete}
      />
    )}

    <Dialog open={isCreateProjectOpen} onOpenChange={onCreateProjectOpenChange}>
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
              onChange={(e) => onNewProjectNameInputChange(e.target.value)}
              className="col-span-3"
              placeholder="e.g. Summer 2024 Release"
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onCreateProjectOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={onCreateProject} disabled={isCreatingProject || !newProjectNameInput.trim()}>
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

    <AlertDialog open={deleteDialogOpen} onOpenChange={onDeleteDialogOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete Contract</AlertDialogTitle>
          <AlertDialogDescription>
            Are you sure you want to delete "{contractToDelete?.file_name}"? This will remove the
            contract file and all its indexed data from the AI search. This action cannot be undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={onDeleteConfirm}
            disabled={deleting}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
          >
            {deleting ? "Deleting..." : "Delete"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
    </>
  );
}
