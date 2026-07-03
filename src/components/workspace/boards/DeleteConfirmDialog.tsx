import { useState } from "react";

import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const norm = (s: string) => s.trim().normalize("NFC");

interface DeleteConfirmDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  name: string;
  /** e.g. "board" or "team" — used in copy */
  resourceType: string;
  /** e.g. "3 tasks" or "2 boards, 5 tasks, 3 members" */
  impact?: string;
  isPending?: boolean;
  onConfirm: () => void;
}

export function DeleteConfirmDialog({
  open,
  onOpenChange,
  name,
  resourceType,
  impact,
  isPending,
  onConfirm,
}: DeleteConfirmDialogProps) {
  const [value, setValue] = useState("");
  const expected = `delete-${norm(name)}`;
  const matches = norm(value) === expected;

  const handleOpenChange = (next: boolean) => {
    if (!next) setValue("");
    onOpenChange(next);
  };

  return (
    <AlertDialog open={open} onOpenChange={handleOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="text-destructive">Delete {resourceType} permanently?</AlertDialogTitle>
        </AlertDialogHeader>
        <div className="space-y-3 text-sm">
          <p>
            This permanently deletes <span className="font-semibold">{name}</span> and everything in it
            {impact ? ` (${impact})` : ""}. <span className="font-semibold">This cannot be undone.</span>
          </p>
          <p className="text-muted-foreground">
            Type <code className="rounded bg-muted px-1 py-0.5">delete-{name}</code> to confirm.
          </p>
          <Input
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={`delete-${name}`}
            aria-label={`Type delete-${name} to confirm`}
            autoFocus
          />
        </div>
        <AlertDialogFooter>
          <Button variant="outline" onClick={() => handleOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="destructive" disabled={!matches || isPending} onClick={onConfirm}>
            {isPending ? "Deleting…" : `Delete ${resourceType}`}
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
