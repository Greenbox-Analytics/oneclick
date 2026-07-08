import { useState } from "react";
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2 } from "lucide-react";
import { useDeleteAccount, DeleteAccountError } from "@/hooks/useDeleteAccount";

interface DeleteAccountDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  userEmail: string;
}

export function DeleteAccountDialog({ open, onOpenChange, userEmail }: DeleteAccountDialogProps) {
  const [confirmInput, setConfirmInput] = useState("");
  const mutation = useDeleteAccount();
  const matches = confirmInput.trim().toLowerCase() === (userEmail || "").trim().toLowerCase();

  const errorMessage = (() => {
    if (!mutation.error) return null;
    if (mutation.error instanceof DeleteAccountError) {
      switch (mutation.error.code) {
        case "last_admin":
          return "You're the only admin. Promote another admin before deleting your account.";
        case "subscription_cancel_failed":
          return "We couldn't cancel your Stripe subscription. Please contact support before deleting.";
        case "email_mismatch":
          return "Email did not match. Please check the input.";
        default:
          return "Something went wrong. Please try again or contact support.";
      }
    }
    return "Something went wrong. Please try again or contact support.";
  })();

  const handleConfirm = async () => {
    if (!matches) return;
    try {
      await mutation.mutateAsync({ confirmationEmail: confirmInput.trim() });
    } catch {
      /* error rendered above */
    }
  };

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="text-destructive">Delete your account</AlertDialogTitle>
          <AlertDialogDescription asChild>
            <div className="space-y-3 text-sm">
              <p>This is permanent and cannot be undone. We will immediately:</p>
              <ul className="list-disc pl-6 space-y-1">
                <li>Cancel your Stripe subscription (no further charges).</li>
                <li>Delete all your artist profiles, projects, works, files, and audio.</li>
                <li>Remove your account from any projects you're collaborating on.</li>
                <li>Sign you out of Msanii.</li>
              </ul>
              <p className="text-muted-foreground">
                You may still see Msanii listed in your connected Google / Slack app permissions —
                revoke those from each service's settings if you want a complete removal.
              </p>
              <p>
                To confirm, type your email address (<span className="font-mono">{userEmail}</span>) below.
              </p>
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>

        <div className="space-y-2">
          <Label htmlFor="delete-account-email-confirm">Your email</Label>
          <Input
            id="delete-account-email-confirm"
            value={confirmInput}
            onChange={(e) => setConfirmInput(e.target.value)}
            placeholder={userEmail}
            disabled={mutation.isPending}
            autoComplete="off"
          />
        </div>

        {errorMessage && <p className="text-sm text-destructive">{errorMessage}</p>}

        <AlertDialogFooter>
          <AlertDialogCancel disabled={mutation.isPending}>Cancel</AlertDialogCancel>
          <AlertDialogAction
            disabled={!matches || mutation.isPending}
            onClick={handleConfirm}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
          >
            {mutation.isPending ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
            Delete account
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
