import { useEffect, useState } from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useDeleteWork } from "@/hooks/useRegistry";

interface DeleteWorkConfirmModalProps {
  workId: string;
  workTitle: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called after a successful delete — caller is responsible for navigation. */
  onDeleted?: () => void;
}

/**
 * Type-to-confirm modal for permanently deleting a work. The "Delete
 * permanently" button stays disabled until the typed title exactly matches the
 * work title (and while the delete is in flight). The delete hook already
 * toasts success/error and invalidates the work lists, so we don't double-toast.
 */
export function DeleteWorkConfirmModal({
  workId,
  workTitle,
  open,
  onOpenChange,
  onDeleted,
}: DeleteWorkConfirmModalProps) {
  const deleteWork = useDeleteWork();
  const [confirmText, setConfirmText] = useState("");

  // Reset the typed input whenever the dialog closes.
  useEffect(() => {
    if (!open) setConfirmText("");
  }, [open]);

  const titleMatches = confirmText.trim() === workTitle;
  const canDelete = titleMatches && !deleteWork.isPending;

  const handleDelete = async () => {
    if (!canDelete) return;
    try {
      // useDeleteWork already toasts success + invalidates the work lists.
      await deleteWork.mutateAsync(workId);
      onDeleted?.();
      onOpenChange(false);
    } catch {
      // useDeleteWork already toasts the error — keep the dialog open so the
      // user can retry.
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="w-5 h-5 shrink-0" />
            <span>Delete “{workTitle}”?</span>
          </DialogTitle>
          <DialogDescription>
            Deleting this work permanently:
          </DialogDescription>
        </DialogHeader>

        <ul className="space-y-1.5 text-sm text-muted-foreground list-disc list-inside">
          <li>removes all ownership stakes &amp; the cap table</li>
          <li>revokes access for all collaborators</li>
          <li>
            unlinks documents &amp; audio (the files themselves stay in the
            project)
          </li>
        </ul>

        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm font-semibold text-destructive">
          This can&apos;t be undone.
        </div>

        <div className="space-y-1.5">
          <label
            htmlFor="delete-work-confirm"
            className="text-xs font-semibold text-foreground"
          >
            Type the work title to confirm
          </label>
          <Input
            id="delete-work-confirm"
            value={confirmText}
            placeholder={workTitle}
            autoComplete="off"
            disabled={deleteWork.isPending}
            onChange={(e) => setConfirmText(e.target.value)}
          />
        </div>

        <DialogFooter className="gap-2">
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={deleteWork.isPending}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            disabled={!canDelete}
            onClick={handleDelete}
          >
            {deleteWork.isPending && (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            )}
            Delete permanently
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
