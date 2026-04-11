import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
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
} from "lucide-react";
import { useDriveBrowse, useDriveImport } from "@/hooks/useGoogleDrive";
import { useProjectsList } from "@/hooks/useProjectsList";
import type { DriveFile } from "@/types/integrations";

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

interface DrivePanelProps {
  onClose?: () => void;
}

export function DrivePanel({ onClose }: DrivePanelProps) {
  const [folderStack, setFolderStack] = useState<{ id: string; name: string }[]>([
    { id: "root", name: "My Drive" },
  ]);
  const [selectedProjectId, setSelectedProjectId] = useState<string>("");

  const currentFolder = folderStack[folderStack.length - 1];
  const { data: files, isLoading } = useDriveBrowse(currentFolder.id);
  const importMutation = useDriveImport();
  const { projects } = useProjectsList();

  const navigateToFolder = (file: DriveFile) => {
    setFolderStack((prev) => [...prev, { id: file.id, name: file.name }]);
  };

  const navigateBack = () => {
    if (folderStack.length > 1) {
      setFolderStack((prev) => prev.slice(0, -1));
    }
  };

  const handleImport = (file: DriveFile) => {
    if (!selectedProjectId) return;
    importMutation.mutate(
      { drive_file_id: file.id, project_id: selectedProjectId },
    );
  };

  const isFolder = (mimeType: string) =>
    mimeType === "application/vnd.google-apps.folder";

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-3">
        <div className="flex items-center gap-2">
          <CardTitle className="text-base">Google Drive</CardTitle>
          <div className="flex items-center text-sm text-muted-foreground">
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
        </div>
        <div className="flex items-center gap-2">
          {folderStack.length > 1 && (
            <Button variant="ghost" size="sm" onClick={navigateBack}>
              <ArrowLeft className="w-4 h-4 mr-1" /> Back
            </Button>
          )}
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose}>
              Close
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {/* Project selector for imports */}
        <div className="mb-4">
          <Select value={selectedProjectId} onValueChange={setSelectedProjectId}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a project to import into..." />
            </SelectTrigger>
            <SelectContent>
              {(projects || []).map((p) => (
                <SelectItem key={p.id} value={p.id}>
                  {p.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* File list */}
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
                This folder is empty
              </p>
            )}
            {files?.map((file) => (
              <div
                key={file.id}
                className={`flex items-center justify-between px-3 py-2 text-sm ${
                  isFolder(file.mimeType)
                    ? "cursor-pointer hover:bg-muted/50"
                    : ""
                }`}
                onClick={() => isFolder(file.mimeType) && navigateToFolder(file)}
              >
                <div className="flex items-center gap-3 min-w-0">
                  {getFileIcon(file.mimeType)}
                  <span className="truncate">{file.name}</span>
                </div>
                <div className="flex items-center gap-4 shrink-0">
                  <span className="text-xs text-muted-foreground">
                    {formatFileSize(file.size)}
                  </span>
                  {file.modifiedTime && (
                    <span className="text-xs text-muted-foreground">
                      {new Date(file.modifiedTime).toLocaleDateString()}
                    </span>
                  )}
                  {!isFolder(file.mimeType) && (
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={!selectedProjectId || importMutation.isPending}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleImport(file);
                      }}
                    >
                      <Download className="w-4 h-4 mr-1" />
                      Import
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
