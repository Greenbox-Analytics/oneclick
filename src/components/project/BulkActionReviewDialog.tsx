import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";

export type BulkAction = "download" | "delete";

interface BulkActionReviewDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  action: BulkAction;
  files: { id: string; name: string }[];
  onConfirm: () => void | Promise<void>;
  isWorking?: boolean;
}

const COPY: Record<BulkAction, { title: (n: number) => string; description: string; confirmLabel: string; destructive: boolean }> = {
  download: {
    title: (n) => `Download ${n} file${n === 1 ? "" : "s"}?`,
    description: "Each file will be downloaded to your browser's default location.",
    confirmLabel: "Download",
    destructive: false,
  },
  delete: {
    title: (n) => `Delete ${n} file${n === 1 ? "" : "s"}?`,
    description: "This permanently removes the selected files and any related links. This cannot be undone.",
    confirmLabel: "Delete",
    destructive: true,
  },
};

export default function BulkActionReviewDialog({
  open,
  onOpenChange,
  action,
  files,
  onConfirm,
  isWorking,
}: BulkActionReviewDialogProps) {
  const copy = COPY[action];
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{copy.title(files.length)}</DialogTitle>
          <DialogDescription>{copy.description}</DialogDescription>
        </DialogHeader>
        <ul className="max-h-60 overflow-y-auto space-y-1 rounded-md border border-border bg-secondary/30 p-2">
          {files.map((f) => (
            <li key={f.id} className="text-sm text-foreground truncate">• {f.name}</li>
          ))}
        </ul>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={isWorking}>
            Cancel
          </Button>
          <Button
            variant={copy.destructive ? "destructive" : "default"}
            onClick={onConfirm}
            disabled={isWorking || files.length === 0}
          >
            {isWorking && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            {copy.confirmLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
