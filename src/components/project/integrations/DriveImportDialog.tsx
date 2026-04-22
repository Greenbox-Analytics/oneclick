import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Folder,
  FileText,
  FileAudio,
  FileImage,
  File,
  ChevronRight,
  Download,
  ArrowLeft,
  Search,
  Loader2,
} from "lucide-react";
import { useDriveBrowse, useDriveImport } from "@/hooks/useGoogleDrive";
import { toast } from "sonner";
import type { DriveFile } from "@/types/integrations";

const CATEGORY_ALLOWED_TYPES: Record<string, { mimeTypes: string[]; label: string }> = {
  contract: {
    mimeTypes: ["application/pdf"],
    label: "PDFs only",
  },
  royalty_statement: {
    mimeTypes: [
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "application/vnd.ms-excel",
      "text/csv",
      "application/vnd.google-apps.spreadsheet",
    ],
    label: "Excel, CSV, Google Sheets",
  },
  split_sheet: {
    mimeTypes: [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "application/msword",
      "application/vnd.google-apps.document",
    ],
    label: "PDF, Word, Google Docs",
  },
  other: {
    mimeTypes: [],
    label: "All file types",
  },
};

function isFileAllowed(mimeType: string, category: string): boolean {
  const allowed = CATEGORY_ALLOWED_TYPES[category];
  if (!allowed || allowed.mimeTypes.length === 0) return true;
  return allowed.mimeTypes.some((t) => mimeType.includes(t) || mimeType === t);
}

function getFileIcon(mimeType: string) {
  if (mimeType === "application/vnd.google-apps.folder") return <Folder className="w-4 h-4 text-blue-500" />;
  if (mimeType.startsWith("audio/")) return <FileAudio className="w-4 h-4 text-purple-500" />;
  if (mimeType.startsWith("image/")) return <FileImage className="w-4 h-4 text-green-500" />;
  if (mimeType.includes("pdf") || mimeType.includes("document")) return <FileText className="w-4 h-4 text-red-500" />;
  return <File className="w-4 h-4 text-muted-foreground" />;
}

function formatFileSize(bytes?: string) {
  if (!bytes) return "";
  const size = parseInt(bytes, 10);
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

interface DriveImportDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId: string;
}

export function DriveImportDialog({ open, onOpenChange, projectId }: DriveImportDialogProps) {
  const [folderStack, setFolderStack] = useState<{ id: string; name: string }[]>([
    { id: "root", name: "My Drive" },
  ]);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<DriveFile[]>([]);
  const [importing, setImporting] = useState(false);
  const [category, setCategory] = useState("contract");

  const currentFolder = folderStack[folderStack.length - 1];
  const { data: files, isLoading } = useDriveBrowse(currentFolder.id, open, searchQuery);
  const importMutation = useDriveImport();

  const navigateToFolder = (file: DriveFile) => {
    setFolderStack((prev) => [...prev, { id: file.id, name: file.name }]);
  };

  const navigateBack = () => {
    if (folderStack.length > 1) {
      setFolderStack((prev) => prev.slice(0, -1));
    }
  };

  const toggleFile = (file: DriveFile) => {
    setSelectedFiles((prev) => {
      const exists = prev.find((f) => f.id === file.id);
      if (exists) return prev.filter((f) => f.id !== file.id);
      return [...prev, file];
    });
  };

  const isSelected = (fileId: string) =>
    selectedFiles.some((f) => f.id === fileId);

  const handleImportSelected = async () => {
    if (selectedFiles.length === 0) return;
    setImporting(true);
    let successCount = 0;
    let duplicateCount = 0;
    for (const file of selectedFiles) {
      try {
        await importMutation.mutateAsync({
          drive_file_id: file.id,
          project_id: projectId,
          file_type: category,
        });
        successCount++;
      } catch (err) {
        const msg = (err as Error).message || "";
        if (msg.includes("already been imported")) {
          duplicateCount++;
        } else {
          toast.error(`Failed to import ${file.name}`);
        }
      }
    }
    setImporting(false);
    if (successCount > 0) {
      toast.success(`Imported ${successCount} file${successCount > 1 ? "s" : ""}`);
    }
    if (duplicateCount > 0) {
      toast.info(`${duplicateCount} file${duplicateCount > 1 ? "s" : ""} already in this project — skipped`);
    }
    setSelectedFiles([]);
    if (successCount > 0) {
      onOpenChange(false);
    }
  };

  const isFolder = (mimeType: string) =>
    mimeType === "application/vnd.google-apps.folder";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Import from Google Drive
            {selectedFiles.length > 0 && (
              <Badge variant="secondary" className="text-xs">
                {selectedFiles.length} selected
              </Badge>
            )}
          </DialogTitle>
          <div className="flex items-center gap-1 text-sm text-muted-foreground">
            {folderStack.map((folder, idx) => (
              <span key={folder.id} className="flex items-center">
                {idx > 0 && <ChevronRight className="w-3 h-3 mx-1" />}
                <button
                  className="hover:text-foreground hover:underline"
                  onClick={() => setFolderStack(folderStack.slice(0, idx + 1))}
                >
                  {folder.name}
                </button>
              </span>
            ))}
          </div>
        </DialogHeader>

        <div className="flex items-center gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search all of Drive..."
              className="pl-9 h-9"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <Select value={category} onValueChange={(val) => { setCategory(val); setSelectedFiles([]); }}>
            <SelectTrigger className="w-44 h-9">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="contract">Contracts</SelectItem>
              <SelectItem value="royalty_statement">Royalty Statements</SelectItem>
              <SelectItem value="split_sheet">Split Sheets</SelectItem>
              <SelectItem value="other">Other</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {CATEGORY_ALLOWED_TYPES[category]?.mimeTypes.length > 0 && (
          <p className="text-xs text-muted-foreground">
            Allowed: {CATEGORY_ALLOWED_TYPES[category].label}
          </p>
        )}

        <div className="flex-1 overflow-y-auto">
          {folderStack.length > 1 && (
            <Button variant="ghost" size="sm" className="mb-2" onClick={navigateBack}>
              <ArrowLeft className="w-4 h-4 mr-1" /> Back
            </Button>
          )}

          {isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          ) : (
            <div className="divide-y divide-border rounded-md border">
              {files?.length === 0 && (
                <p className="p-4 text-sm text-muted-foreground text-center">
                  {searchQuery ? "No files found" : "This folder is empty"}
                </p>
              )}
              {files?.map((file) => {
                const folder = isFolder(file.mimeType);
                const allowed = folder || isFileAllowed(file.mimeType, category);
                return (
                <div
                  key={file.id}
                  className={`flex items-center gap-3 px-3 py-2 text-sm ${
                    folder
                      ? "cursor-pointer hover:bg-muted/50"
                      : !allowed
                        ? "opacity-40 cursor-not-allowed"
                        : isSelected(file.id)
                          ? "bg-primary/5 cursor-pointer"
                          : "hover:bg-muted/30 cursor-pointer"
                  }`}
                  onClick={() => {
                    if (folder) {
                      navigateToFolder(file);
                    } else if (allowed) {
                      toggleFile(file);
                    }
                  }}
                >
                  {!folder && (
                    <Checkbox
                      checked={isSelected(file.id)}
                      disabled={!allowed}
                      onCheckedChange={() => allowed && toggleFile(file)}
                      onClick={(e) => e.stopPropagation()}
                    />
                  )}
                  <div className="flex items-center gap-2 min-w-0 flex-1">
                    {getFileIcon(file.mimeType)}
                    <span className="truncate">{file.name}</span>
                  </div>
                  <div className="flex items-center gap-3 shrink-0 text-xs text-muted-foreground">
                    {formatFileSize(file.size)}
                    {file.modifiedTime && (
                      <span>{new Date(file.modifiedTime).toLocaleDateString()}</span>
                    )}
                  </div>
                </div>
                );
              })}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" size="sm" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            size="sm"
            disabled={selectedFiles.length === 0 || importing}
            onClick={handleImportSelected}
          >
            {importing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Importing...
              </>
            ) : (
              <>
                <Download className="w-4 h-4 mr-2" />
                Import {selectedFiles.length > 0 ? `(${selectedFiles.length})` : ""}
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
