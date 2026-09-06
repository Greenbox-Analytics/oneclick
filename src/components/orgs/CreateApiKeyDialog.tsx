// src/components/orgs/CreateApiKeyDialog.tsx
// "New API key": name + optional expiry -> secret shown ONCE. One key type:
// every key is this team's credential. Copy is for the person paying, not a
// developer — "credits", never bearer/hash.
import { useState } from "react";
import { Check, Copy, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useCreatePartnerKey } from "@/hooks/usePartnerKeys";
import { endOfLocalDayIso, tomorrowInputValue } from "@/lib/partnerKeys";

export function CreateApiKeyDialog({
  orgId,
  open,
  onOpenChange,
}: {
  orgId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const create = useCreatePartnerKey();
  const [label, setLabel] = useState("");
  const [expires, setExpires] = useState("");
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);

  const secret = create.data?.secret ?? null;
  const missing = !label.trim() ? "Add a name" : null;
  const canSubmit = !missing && !create.isPending;

  const reset = () => {
    setLabel("");
    setExpires("");
    setCopied(false);
    setCopyFailed(false);
    create.reset();
  };

  const handleOpenChange = (next: boolean) => {
    // While a secret is on screen the ONLY way out is Done — a backdrop click
    // must not throw away the one chance to copy it.
    if (!next && secret) return;
    if (!next) reset();
    onOpenChange(next);
  };

  const handleDone = () => {
    reset();
    onOpenChange(false);
  };

  const handleCreate = () => {
    if (!canSubmit) return;
    create.mutate({
      orgId,
      label: label.trim(),
      ...(expires ? { expires_at: endOfLocalDayIso(expires) } : {}),
    });
  };

  const handleCopy = async () => {
    if (!secret) return;
    try {
      await navigator.clipboard.writeText(secret);
      setCopied(true);
      setCopyFailed(false);
    } catch {
      // navigator.clipboard is absent on a non-secure origin — say so, the
      // secret is unrecoverable once Done is clicked.
      setCopied(false);
      setCopyFailed(true);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent
        onInteractOutside={(e) => secret && e.preventDefault()}
        onEscapeKeyDown={(e) => secret && e.preventDefault()}
      >
        {secret ? (
          <>
            <DialogHeader>
              <DialogTitle>Your new API key</DialogTitle>
              <DialogDescription>Copy this now. For your security, we can't show it again.</DialogDescription>
            </DialogHeader>
            <div className="flex items-center gap-2">
              <code className="flex-1 rounded-md border bg-muted px-3 py-2 text-[13px] font-mono break-all select-all">
                {secret}
              </code>
              <Button type="button" variant="outline" size="sm" onClick={handleCopy}>
                {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                <span className="ml-1.5">{copied ? "Copied" : "Copy"}</span>
              </Button>
            </div>
            {copyFailed && (
              <p role="alert" className="text-[12px] text-destructive">
                Couldn&apos;t copy automatically — select the key above and copy it yourself.
              </p>
            )}
            <DialogFooter>
              <Button type="button" onClick={handleDone}>
                Done
              </Button>
            </DialogFooter>
          </>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>New API key</DialogTitle>
              <DialogDescription>
                Treat it like a password: anyone holding it can run calculations on this team&apos;s credits.
              </DialogDescription>
            </DialogHeader>
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="api-key-name">Name</Label>
                <Input
                  id="api-key-name"
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                  placeholder="Your key name"
                  maxLength={120}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="api-key-expires">Expires (optional)</Label>
                <Input
                  id="api-key-expires"
                  type="date"
                  min={tomorrowInputValue()}
                  value={expires}
                  onChange={(e) => setExpires(e.target.value)}
                />
                <p className="text-[12px] text-muted-foreground">The key stops working at the end of this day.</p>
              </div>
              {create.error && (
                <p role="alert" className="text-[13px] text-destructive">
                  {create.error.message}
                </p>
              )}
              {missing && (
                <p id="api-key-create-hint" className="text-[12px] text-muted-foreground">
                  {missing} to create this key.
                </p>
              )}
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => handleOpenChange(false)}>
                Cancel
              </Button>
              {/* aria-disabled, not disabled: keyboard and screen-reader users can reach it and hear why. */}
              <Button
                type="button"
                onClick={handleCreate}
                aria-disabled={!canSubmit}
                aria-describedby={missing ? "api-key-create-hint" : undefined}
                className={canSubmit ? undefined : "opacity-50"}
              >
                {create.isPending && <Loader2 className="w-4 h-4 mr-1.5 animate-spin" />}
                Create
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
