import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import { useRevealCredential } from "@/hooks/useArtistCredentials";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  credentialId: string | null;
  onRevealed: (credentialId: string, password: string) => void;
}

export default function PasswordRevealDialog({ open, onOpenChange, credentialId, onRevealed }: Props) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const reveal = useRevealCredential();

  const reset = () => {
    setPassword("");
    setError("");
  };

  const handleConfirm = async () => {
    if (!credentialId || !password) return;
    setError("");
    try {
      const result = await reveal.mutateAsync({ credentialId, msaniiPassword: password });
      onRevealed(credentialId, result.password);
      reset();
      onOpenChange(false);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Reveal failed";
      setError(msg);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) reset();
        onOpenChange(next);
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Confirm your Msanii password</DialogTitle>
          <DialogDescription>
            Enter your Msanii account password to reveal this credential. The password will hide again after 30 seconds.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-2">
          <Label htmlFor="msanii-password">Msanii password</Label>
          <Input
            id="msanii-password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleConfirm();
            }}
            autoFocus
          />
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleConfirm} disabled={!password || reveal.isPending}>
            {reveal.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            Reveal
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
