import { useState, useRef } from "react";
import { useToast } from "@/hooks/use-toast";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Separator } from "@/components/ui/separator";
import { Checkbox } from "@/components/ui/checkbox";
import { FileShareDialog } from "@/components/FileShareDialog";
import {
  FolderPlus,
  Folder,
  Music,
  Upload,
  MoreVertical,
  Trash2,
  Pencil,
  Link,
  Unlink,
  X,
  Check,
  Send,
} from "lucide-react";
import type { AudioFolder, AudioFile } from "@/types/audio";

interface FileToShare {
  file_name: string;
  file_path: string;
  file_source: "project_file" | "audio_file";
  file_id: string;
}

interface AudioSectionProps {
  artistId: string;
  folders: AudioFolder[];
  filesByFolder: Map<string, AudioFile[]>;
  projectsByAudioFile: Map<string, string[]>;
  artistProjects: { id: string; name: string }[];
  onCreateFolder: (artistId: string, name: string) => Promise<void>;
  onRenameFolder: (folderId: string, newName: string) => Promise<void>;
  onDeleteFolder: (folderId: string) => Promise<void>;
  onUploadFile: (folderId: string, artistId: string, file: File) => Promise<void>;
  onDeleteFile: (fileId: string, filePath: string) => Promise<void>;
  onLinkAudio: (audioFileId: string, projectId: string) => Promise<void>;
  onUnlinkAudio: (audioFileId: string, projectId: string) => Promise<void>;
}

export const AudioSection = ({
  artistId,
  folders,
  filesByFolder,
  projectsByAudioFile,
  artistProjects,
  onCreateFolder,
  onRenameFolder,
  onDeleteFolder,
  onUploadFile,
  onDeleteFile,
  onLinkAudio,
  onUnlinkAudio,
}: AudioSectionProps) => {
  const { toast } = useToast();
  const [creatingFolder, setCreatingFolder] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");
  const [renamingFolderId, setRenamingFolderId] = useState<string | null>(null);
  const [renameName, setRenameName] = useState("");
  const [folderToDelete, setFolderToDelete] = useState<AudioFolder | null>(null);
  const [fileToDelete, setFileToDelete] = useState<AudioFile | null>(null);
  const [linkingFile, setLinkingFile] = useState<AudioFile | null>(null);
  const [uploading, setUploading] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  const [filesToShare, setFilesToShare] = useState<FileToShare[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadFolderIdRef = useRef<string>("");

  const handleCreateFolder = async () => {
    if (!newFolderName.trim()) return;
    setSaving(true);
    try {
      await onCreateFolder(artistId, newFolderName);
      setNewFolderName("");
      setCreatingFolder(false);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "";
      if (msg.startsWith("DUPLICATE:")) {
        toast({ title: "Duplicate folder name", description: msg.replace("DUPLICATE:", ""), className: "bg-white text-black border border-border" });
      }
    } finally {
      setSaving(false);
    }
  };

  const handleRenameFolder = async (folderId: string) => {
    if (!renameName.trim()) return;
    setSaving(true);
    try {
      await onRenameFolder(folderId, renameName);
      setRenamingFolderId(null);
    } catch {
      // toast handled by parent
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteFolder = async () => {
    if (!folderToDelete) return;
    try {
      await onDeleteFolder(folderToDelete.id);
    } catch {
      // handled by parent
    } finally {
      setFolderToDelete(null);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const folderId = uploadFolderIdRef.current;
    setUploading(folderId);
    try {
      await onUploadFile(folderId, artistId, file);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "";
      if (msg.startsWith("DUPLICATE:")) {
        toast({ title: "Duplicate file name", description: msg.replace("DUPLICATE:", ""), className: "bg-white text-black border border-border" });
      }
    } finally {
      setUploading(null);
      e.target.value = "";
    }
  };

  const handleDeleteFile = async () => {
    if (!fileToDelete) return;
    try {
      await onDeleteFile(fileToDelete.id, fileToDelete.file_path);
    } catch {
      // handled by parent
    } finally {
      setFileToDelete(null);
    }
  };

  const handleToggleLink = async (audioFileId: string, projectId: string, linked: boolean) => {
    try {
      if (linked) {
        await onUnlinkAudio(audioFileId, projectId);
      } else {
        await onLinkAudio(audioFileId, projectId);
      }
    } catch {
      // handled by parent
    }
  };

  const handleShareFile = (file: AudioFile) => {
    setFilesToShare([
      {
        file_name: file.file_name,
        file_path: file.file_path,
        file_source: "audio_file",
        file_id: file.id,
      },
    ]);
    setShareDialogOpen(true);
  };

  const handleShareFolder = (folder: AudioFolder) => {
    const files = filesByFolder.get(folder.id) || [];
    if (files.length === 0) return;
    setFilesToShare(
      files.map((f) => ({
        file_name: f.file_name,
        file_path: f.file_path,
        file_source: "audio_file" as const,
        file_id: f.id,
      }))
    );
    setShareDialogOpen(true);
  };

  const formatFileSize = (bytes: number | null) => {
    if (!bytes) return "";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div>
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={handleFileUpload}
      />

      {/* Section header */}
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold flex items-center gap-2">
          <Music className="w-4 h-4 text-muted-foreground" />
          Audio Folders
          {folders.length > 0 && (
            <Badge variant="secondary" className="text-xs">
              {folders.length}
            </Badge>
          )}
        </h4>
        <Button
          variant="outline"
          size="sm"
          className="h-7 text-xs gap-1"
          onClick={() => {
            setCreatingFolder(true);
            setNewFolderName("");
          }}
        >
          <FolderPlus className="w-3.5 h-3.5" />
          New Folder
        </Button>
      </div>

      {/* New folder inline input */}
      {creatingFolder && (
        <div className="flex items-center gap-2 mb-3">
          <Folder className="w-4 h-4 text-muted-foreground flex-shrink-0" />
          <Input
            autoFocus
            placeholder="Folder name..."
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleCreateFolder();
              if (e.key === "Escape") setCreatingFolder(false);
            }}
            className="h-8 text-sm flex-1"
            disabled={saving}
          />
          <Button size="icon" variant="ghost" className="h-8 w-8" onClick={handleCreateFolder} disabled={saving || !newFolderName.trim()}>
            <Check className="w-4 h-4" />
          </Button>
          <Button size="icon" variant="ghost" className="h-8 w-8" onClick={() => setCreatingFolder(false)}>
            <X className="w-4 h-4" />
          </Button>
        </div>
      )}

      {/* Folder tiles — card layout */}
      {folders.length === 0 && !creatingFolder ? (
        <div className="text-center py-6 text-muted-foreground">
          <Folder className="w-10 h-10 mx-auto mb-2 opacity-40" />
          <p className="text-sm">No audio folders yet</p>
          <p className="text-xs mt-1">Create a folder to start uploading audio files</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {folders.map((folder) => {
            const files = filesByFolder.get(folder.id) || [];
            const isRenaming = renamingFolderId === folder.id;
            const isUploading = uploading === folder.id;
            const totalSize = files.reduce((sum, f) => sum + (f.file_size || 0), 0);

            return (
              <Card key={folder.id} className="border border-border bg-card shadow-md hover:shadow-lg transition-shadow">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <Folder className="w-5 h-5 text-cyan-500 flex-shrink-0" />
                      {isRenaming ? (
                        <div className="flex items-center gap-1 flex-1">
                          <Input
                            autoFocus
                            value={renameName}
                            onChange={(e) => setRenameName(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") handleRenameFolder(folder.id);
                              if (e.key === "Escape") setRenamingFolderId(null);
                            }}
                            className="h-7 text-sm"
                            disabled={saving}
                          />
                          <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => handleRenameFolder(folder.id)} disabled={saving}>
                            <Check className="w-3.5 h-3.5" />
                          </Button>
                          <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => setRenamingFolderId(null)}>
                            <X className="w-3.5 h-3.5" />
                          </Button>
                        </div>
                      ) : (
                        <CardTitle className="text-sm truncate">{folder.name}</CardTitle>
                      )}
                    </div>
                    {!isRenaming && (
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-7 w-7">
                            <MoreVertical className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem
                            onClick={() => {
                              uploadFolderIdRef.current = folder.id;
                              fileInputRef.current?.click();
                            }}
                            disabled={isUploading}
                          >
                            <Upload className="w-4 h-4 mr-2" />
                            Upload File
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={() => {
                            setRenamingFolderId(folder.id);
                            setRenameName(folder.name);
                          }}>
                            <Pencil className="w-4 h-4 mr-2" />
                            Rename
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={() => handleShareFolder(folder)}
                            disabled={files.length === 0}
                          >
                            <Send className="w-4 h-4 mr-2" />
                            Share Folder
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            className="text-destructive focus:text-destructive"
                            onClick={() => setFolderToDelete(folder)}
                          >
                            <Trash2 className="w-4 h-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-xs text-muted-foreground mt-1">
                    <span>{files.length} file{files.length !== 1 ? "s" : ""}</span>
                    {totalSize > 0 && <span>{formatFileSize(totalSize)}</span>}
                    {isUploading && (
                      <div className="flex items-center gap-1">
                        <div className="w-3 h-3 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                        <span>Uploading...</span>
                      </div>
                    )}
                  </div>
                </CardHeader>

                <CardContent className="pt-0">
                  {files.length > 0 && (
                    <Accordion type="multiple">
                      <AccordionItem value="files" className="border-none">
                        <AccordionTrigger className="hover:no-underline py-1.5 text-xs font-medium text-muted-foreground">
                          View Files
                        </AccordionTrigger>
                        <AccordionContent>
                    <div className="grid grid-cols-2 gap-2 pt-1">
                      {files.map((file) => {
                        const linkedProjectIds = projectsByAudioFile.get(file.id) || [];
                        const linkedProjects = artistProjects.filter((p) =>
                          linkedProjectIds.includes(p.id)
                        );

                        return (
                          <div
                            key={file.id}
                            className="p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors group/file relative"
                          >
                            <div className="flex items-center justify-between mb-1.5">
                              <Music className="w-4 h-4 text-cyan-500" />
                              <div className="flex items-center gap-0.5 opacity-0 group-hover/file:opacity-100 transition-opacity">
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-5 w-5"
                                  title="Share file"
                                  onClick={() => handleShareFile(file)}
                                >
                                  <Send className="w-3 h-3" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-5 w-5"
                                  title="Link to project"
                                  onClick={() => setLinkingFile(file)}
                                >
                                  <Link className="w-3 h-3" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-5 w-5 text-destructive hover:text-destructive"
                                  title="Delete file"
                                  onClick={() => setFileToDelete(file)}
                                >
                                  <Trash2 className="w-3 h-3" />
                                </Button>
                              </div>
                            </div>
                            <p className="text-xs font-medium truncate" title={file.file_name}>{file.file_name}</p>
                            <p className="text-[11px] text-muted-foreground mt-0.5">
                              {formatFileSize(file.file_size)}
                            </p>
                            {linkedProjects.length > 0 && (
                              <div className="flex items-center gap-1 flex-wrap mt-1.5">
                                {linkedProjects.map((p) => (
                                  <Badge key={p.id} variant="outline" className="text-[10px] h-4 gap-0.5 px-1">
                                    <Link className="w-2 h-2" />
                                    {p.name}
                                  </Badge>
                                ))}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                        </AccordionContent>
                      </AccordionItem>
                    </Accordion>
                  )}

                  {files.length === 0 && (
                    <div className="text-center py-3">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-xs gap-1"
                        onClick={() => {
                          uploadFolderIdRef.current = folder.id;
                          fileInputRef.current?.click();
                        }}
                        disabled={isUploading}
                      >
                        <Upload className="w-3.5 h-3.5" />
                        Upload audio file
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {/* === Share Dialog === */}
      <FileShareDialog
        open={shareDialogOpen}
        onOpenChange={setShareDialogOpen}
        files={filesToShare}
      />

      {/* === Link to Project Dialog === */}
      <Dialog open={linkingFile !== null} onOpenChange={(open) => { if (!open) setLinkingFile(null); }}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle className="text-base">Link to Projects</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground truncate">
            {linkingFile?.file_name}
          </p>
          {artistProjects.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4">No projects for this artist</p>
          ) : (
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {artistProjects.map((project) => {
                const linked = (projectsByAudioFile.get(linkingFile?.id || "") || []).includes(project.id);
                return (
                  <label
                    key={project.id}
                    className="flex items-center gap-3 p-2 rounded-md hover:bg-muted/50 cursor-pointer"
                  >
                    <Checkbox
                      checked={linked}
                      onCheckedChange={() => {
                        if (linkingFile) handleToggleLink(linkingFile.id, project.id, linked);
                      }}
                    />
                    <span className="text-sm">{project.name}</span>
                    {linked && <Unlink className="w-3 h-3 ml-auto text-muted-foreground" />}
                  </label>
                );
              })}
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" size="sm" onClick={() => setLinkingFile(null)}>
              Done
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* === Delete Folder Confirmation === */}
      <AlertDialog open={folderToDelete !== null} onOpenChange={(open) => { if (!open) setFolderToDelete(null); }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Folder?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete <strong>{folderToDelete?.name}</strong> and all audio files inside it.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteFolder}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete Folder
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* === Delete File Confirmation === */}
      <AlertDialog open={fileToDelete !== null} onOpenChange={(open) => { if (!open) setFileToDelete(null); }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Audio File?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete <strong>{fileToDelete?.file_name}</strong>.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteFile}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete File
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};
