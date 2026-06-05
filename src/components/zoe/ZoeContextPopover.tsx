import { useState } from "react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Plus, X, Loader2, Users, FolderOpen, FileText } from "lucide-react";
import type { Artist, Project, Contract } from "@/components/zoe/types";
import { ContractUploadModal } from "@/components/ContractUploadModal";

interface ZoeContextPopoverProps {
  children: React.ReactNode;
  artists: Artist[];
  selectedArtist: string;
  onArtistChange: (value: string) => void;
  projects: Project[];
  selectedProject: string;
  onProjectChange: (value: string) => void;
  contracts: Contract[];
  selectedContracts: string[];
  onSelectedContractsChange: (contracts: string[]) => void;
  uploadModalOpen: boolean;
  onUploadModalOpenChange: (open: boolean) => void;
  onUploadComplete: () => void;
  isCreateProjectOpen: boolean;
  onCreateProjectOpenChange: (open: boolean) => void;
  newProjectNameInput: string;
  onNewProjectNameInputChange: (value: string) => void;
  isCreatingProject: boolean;
  onCreateProject: () => void;
}

export function ZoeContextPopover({
  children,
  artists,
  selectedArtist,
  onArtistChange,
  projects,
  selectedProject,
  onProjectChange,
  contracts,
  selectedContracts,
  onSelectedContractsChange,
  uploadModalOpen,
  onUploadModalOpenChange,
  onUploadComplete,
  isCreateProjectOpen,
  onCreateProjectOpenChange,
  newProjectNameInput,
  onNewProjectNameInputChange,
  isCreatingProject,
  onCreateProject,
}: ZoeContextPopoverProps) {
  const [open, setOpen] = useState(false);

  const contractContracts = contracts.filter((c) => c.folder_category === "contract");
  const splitSheets = contracts.filter((c) => c.folder_category === "split_sheet");
  const otherContracts = contracts.filter(
    (c) => c.folder_category !== "contract" && c.folder_category !== "split_sheet"
  );

  return (
    <>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>{children}</PopoverTrigger>
        <PopoverContent
          align="end"
          sideOffset={8}
          className="w-80 p-0 shadow-xl border"
          style={{ zIndex: 50 }}
        >
          <div className="p-4 space-y-4">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              Context
            </p>

            {/* Artist */}
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                <Users className="w-3 h-3" />
                Artist
              </div>
              <div className="flex items-center gap-2">
                <Select value={selectedArtist} onValueChange={onArtistChange}>
                  <SelectTrigger className="flex-1 h-8 text-xs">
                    <SelectValue placeholder="Select artist…" />
                  </SelectTrigger>
                  <SelectContent>
                    {artists.map((a) => (
                      <SelectItem key={a.id} value={a.id} className="text-xs">
                        {a.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {selectedArtist && (
                  <button
                    onClick={() => onArtistChange("")}
                    className="w-7 h-7 flex items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                    title="Clear artist"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </div>

            {/* Project */}
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                <FolderOpen className="w-3 h-3" />
                Project
              </div>
              <div className="flex items-center gap-2">
                <Select
                  value={selectedProject}
                  onValueChange={onProjectChange}
                  disabled={!selectedArtist}
                >
                  <SelectTrigger className="flex-1 h-8 text-xs">
                    <SelectValue placeholder="Select project…" />
                  </SelectTrigger>
                  <SelectContent>
                    {projects.map((p) => (
                      <SelectItem key={p.id} value={p.id} className="text-xs">
                        {p.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <button
                  onClick={() => onCreateProjectOpenChange(true)}
                  disabled={!selectedArtist}
                  className="w-7 h-7 flex items-center justify-center rounded border border-border text-muted-foreground hover:text-foreground hover:bg-muted transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  title="Create new project"
                >
                  <Plus className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Contracts */}
            {selectedProject && (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                    <FileText className="w-3 h-3" />
                    Contracts {contracts.length > 0 && `(${contracts.length})`}
                  </div>
                  <button
                    onClick={() => onUploadModalOpenChange(true)}
                    className="text-xs text-muted-foreground hover:text-foreground underline underline-offset-2"
                  >
                    Upload
                  </button>
                </div>

                {contracts.length === 0 ? (
                  <p className="text-xs text-muted-foreground text-center py-2">
                    No documents yet
                  </p>
                ) : (
                  <ScrollArea className="max-h-52">
                    <div className="space-y-0.5 pr-1">
                      {selectedContracts.length > 0 && (
                        <button
                          onClick={() => onSelectedContractsChange([])}
                          className="w-full text-left text-xs text-muted-foreground hover:text-foreground px-1 py-1 rounded transition-colors"
                        >
                          Clear selection
                        </button>
                      )}

                      {contractContracts.length > 0 && (
                        <>
                          <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider px-1 pt-1 pb-0.5">
                            Contracts
                          </p>
                          {contractContracts.map((c) => (
                            <ContractCheckbox
                              key={c.id}
                              contract={c}
                              checked={selectedContracts.includes(c.id)}
                              onChange={(checked) => {
                                if (checked) {
                                  onSelectedContractsChange([...selectedContracts, c.id]);
                                } else {
                                  onSelectedContractsChange(
                                    selectedContracts.filter((id) => id !== c.id)
                                  );
                                }
                              }}
                            />
                          ))}
                        </>
                      )}

                      {splitSheets.length > 0 && (
                        <>
                          <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider px-1 pt-2 pb-0.5">
                            Split Sheets
                          </p>
                          {splitSheets.map((c) => (
                            <ContractCheckbox
                              key={c.id}
                              contract={c}
                              checked={selectedContracts.includes(c.id)}
                              onChange={(checked) => {
                                if (checked) {
                                  onSelectedContractsChange([...selectedContracts, c.id]);
                                } else {
                                  onSelectedContractsChange(
                                    selectedContracts.filter((id) => id !== c.id)
                                  );
                                }
                              }}
                            />
                          ))}
                        </>
                      )}

                      {otherContracts.length > 0 &&
                        otherContracts.map((c) => (
                          <ContractCheckbox
                            key={c.id}
                            contract={c}
                            checked={selectedContracts.includes(c.id)}
                            onChange={(checked) => {
                              if (checked) {
                                onSelectedContractsChange([...selectedContracts, c.id]);
                              } else {
                                onSelectedContractsChange(
                                  selectedContracts.filter((id) => id !== c.id)
                                );
                              }
                            }}
                          />
                        ))}
                    </div>
                  </ScrollArea>
                )}
              </div>
            )}
          </div>
        </PopoverContent>
      </Popover>

      {/* Upload modal — rendered outside Popover so it doesn't close when opening */}
      {selectedProject && (
        <ContractUploadModal
          open={uploadModalOpen}
          onOpenChange={onUploadModalOpenChange}
          projectId={selectedProject}
          onUploadComplete={onUploadComplete}
        />
      )}

      {/* Create project dialog */}
      <Dialog open={isCreateProjectOpen} onOpenChange={onCreateProjectOpenChange}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Create New Project</DialogTitle>
            <DialogDescription>Create a new project to organize your files.</DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="zcp-name" className="text-right">
                Name
              </Label>
              <Input
                id="zcp-name"
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
            <Button
              onClick={onCreateProject}
              disabled={isCreatingProject || !newProjectNameInput.trim()}
            >
              {isCreatingProject ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Creating…
                </>
              ) : (
                "Create Project"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

function ContractCheckbox({
  contract,
  checked,
  onChange,
}: {
  contract: Contract;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <div className="flex items-center gap-2 px-1 py-1 rounded hover:bg-muted/50">
      <Checkbox
        id={`ctx-${contract.id}`}
        checked={checked}
        onCheckedChange={(v) => onChange(!!v)}
        className="w-3.5 h-3.5"
      />
      <label
        htmlFor={`ctx-${contract.id}`}
        className="text-xs leading-none cursor-pointer truncate flex-1"
      >
        {contract.file_name}
      </label>
    </div>
  );
}
