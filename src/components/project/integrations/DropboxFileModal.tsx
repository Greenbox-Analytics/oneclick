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
import { Skeleton } from "@/components/ui/skeleton";
import { ArrowLeft, Check, ChevronRight, Copy, Folder, Loader2, RefreshCw, Upload } from "lucide-react";
import { useDriveBrowse } from "@/hooks/useGoogleDrive";
import { useDropboxExport, useDropboxExportStatus } from "@/hooks/useDropbox";
import { toast } from "sonner";

// Shared folder sentinel: the Dropbox backend normalizes its folders to this
// same mimeType so one browse UI serves both providers.
const FOLDER_MIME = "application/vnd.google-apps.folder";

interface DropboxFileModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectFileId: string | null;
  fileName: string;
}

export function DropboxFileModal({ open, onOpenChange, projectFileId, fileName }: DropboxFileModalProps) {
  const [folderStack, setFolderStack] = useState<{ id: string; name: string }[]>([
    { id: "root", name: "Dropbox" },
  ]);
  const [copied, setCopied] = useState(false);

  const status = useDropboxExportStatus(projectFileId, open);
  const exportMutation = useDropboxExport();

  const currentFolder = folderStack[folderStack.length - 1];
  const shareUrl = exportMutation.data?.share_url ?? status.data?.share_url ?? null;
  // `saved` and `shareUrl` are DISTINCT signals. A file can be saved in Dropbox
  // with no link yet (upload succeeded, link creation failed) — that must not
  // show a folder picker, because the retry only repairs the link and ignores
  // any newly-picked folder. Picker shows only when nothing was ever saved.
  const saved = exportMutation.isSuccess || !!status.data?.saved;
  // An errored status query leaves `saved` unknown, not false — falling
  // through to the picker here would let a user "save" an already-saved
  // file, silently ignoring their folder choice via backend idempotency.
  const showPicker = !saved && !status.isLoading && !status.isError;
  const { data: files, isLoading } = useDriveBrowse(currentFolder.id, open && showPicker, "", "dropbox");
  const folders = files?.filter((f) => f.mimeType === FOLDER_MIME) ?? [];

  const handleSave = () => {
    if (!projectFileId) return;
    exportMutation.mutate({
      project_file_id: projectFileId,
      dropbox_folder_id: currentFolder.id === "root" ? undefined : currentFolder.id,
    });
  };

  const handleCopy = async () => {
    if (!shareUrl) return;
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      toast.success("Link copied");
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error("Couldn't copy automatically — click the link above to select it, then copy manually.");
    }
  };

  const handleOpenChange = (next: boolean) => {
    if (!next) {
      setFolderStack([{ id: "root", name: "Dropbox" }]);
      exportMutation.reset();
      setCopied(false);
    }
    onOpenChange(next);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="truncate">Save "{fileName}" to Dropbox</DialogTitle>
        </DialogHeader>

        {status.isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-9 w-full" />
            <Skeleton className="h-9 w-full" />
          </div>
        ) : status.isError && !status.data ? (
          <p className="text-sm text-muted-foreground">
            We couldn't check whether this file is already in your Dropbox. Please try again.
          </p>
        ) : saved ? (
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground">
              {shareUrl
                ? "This file is saved in your Dropbox. Anyone with this link can view it."
                : "This file is saved in your Dropbox, but we couldn't finish creating the share link. Try again to get the link — your file won't be uploaded twice."}
            </p>
            {shareUrl && (
              <div className="flex items-center gap-2">
                <Input readOnly value={shareUrl} className="text-xs" onFocus={(e) => e.target.select()} />
                <Button size="sm" variant="outline" onClick={handleCopy} title="Copy link">
                  {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                </Button>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Choose a Dropbox folder to save a copy of this file into.
            </p>
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
            <div className="max-h-56 overflow-y-auto rounded-md border divide-y divide-border">
              {folderStack.length > 1 && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-start"
                  onClick={() => setFolderStack((prev) => prev.slice(0, -1))}
                >
                  <ArrowLeft className="w-4 h-4 mr-1" /> Back
                </Button>
              )}
              {isLoading ? (
                <div className="p-3 space-y-2">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                </div>
              ) : folders.length === 0 ? (
                <p className="p-3 text-sm text-muted-foreground text-center">No folders here</p>
              ) : (
                folders.map((folder) => (
                  <button
                    key={folder.id}
                    className="flex w-full items-center gap-2 px-3 py-2 text-sm hover:bg-muted/50"
                    onClick={() => setFolderStack((prev) => [...prev, { id: folder.id, name: folder.name }])}
                  >
                    <Folder className="w-4 h-4 text-blue-500" />
                    <span className="truncate">{folder.name}</span>
                  </button>
                ))
              )}
            </div>
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" size="sm" onClick={() => handleOpenChange(false)}>
            {shareUrl ? "Done" : "Cancel"}
          </Button>
          {status.isError && !status.data ? (
            // Saved state is unknown here — offer a re-check, never a Save,
            // so nobody can trigger a save from an unknown state.
            <Button size="sm" onClick={() => status.refetch()}>
              <RefreshCw className="w-4 h-4 mr-2" /> Try again
            </Button>
          ) : (
            !status.isLoading && !shareUrl && (
              <Button size="sm" disabled={exportMutation.isPending} onClick={handleSave}>
                {exportMutation.isPending ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Saving...
                  </>
                ) : saved ? (
                  // Saved but no link: the backend repairs the link without re-uploading.
                  <>
                    <Copy className="w-4 h-4 mr-2" /> Get share link
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4 mr-2" /> Save to {currentFolder.name}
                  </>
                )}
              </Button>
            )
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
