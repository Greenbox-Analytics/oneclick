// src/components/orgs/DissolveOrgDialog.tsx
// Terminal self-serve lifecycle op — type-the-name confirm over exactly what
// will happen (spec §3, §8). A plain Dialog rather than AlertDialog (mirrors
// OrgSeatsTable's CapDialog): we need full control of when it closes — only
// on a successful dissolve, never on click — plus the preview fetch gated on
// `open`. The confirm button borrows AlertDialogAction's destructive styling.
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useDissolvePreview, useDissolveOrg } from "@/hooks/useOrgs";

export function DissolveOrgDialog({
  orgId,
  orgName,
  open,
  onOpenChange,
}: {
  orgId: string;
  orgName: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const navigate = useNavigate();
  const [value, setValue] = useState("");
  const { data: preview, isLoading } = useDissolvePreview(orgId, open);
  const dissolve = useDissolveOrg();

  const confirmed = value.trim() === orgName;

  const handleOpenChange = (next: boolean) => {
    if (!next) setValue("");
    onOpenChange(next);
  };

  const handleConfirm = () => {
    if (!confirmed) return;
    dissolve.mutate(
      { orgId, confirmName: value.trim() },
      {
        onSuccess: () => {
          handleOpenChange(false);
          navigate("/organization");
        },
      },
    );
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Dissolve {orgName}?</DialogTitle>
          <DialogDescription>
            This permanently closes the team. Artists return to their creators, purchased credits in
            the pool are forfeited, and this can&apos;t be undone.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          {isLoading ? (
            <div className="flex items-center justify-center py-6">
              <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
            </div>
          ) : (
            preview && (
              <>
                {preview.recipients.length > 0 && (
                  <div className="bg-background border border-border rounded-xl px-4 py-3.5">
                    <div className="text-[11px] font-semibold tracking-[0.11em] uppercase text-muted-foreground/70 mb-2">
                      Artists return to
                    </div>
                    <div className="space-y-1.5 max-h-40 overflow-y-auto">
                      {preview.recipients.map((r) => (
                        <div key={r.artistId} className="flex items-center justify-between gap-3 text-sm">
                          <span className="min-w-0 truncate">{r.artistName || "Untitled artist"}</span>
                          <span className="text-muted-foreground flex-none text-right">
                            {r.fallback ? "returned to you" : r.email || "—"}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {(preview.forfeitReserve > 0 || preview.inertBundle > 0) && (
                  <div className="bg-background border border-border rounded-xl px-4 py-3.5 space-y-1.5">
                    {preview.forfeitReserve > 0 && (
                      <div className="flex items-center justify-between gap-3 text-sm">
                        <span className="text-muted-foreground">Purchased credits forfeited</span>
                        <span className="tabular-nums">{preview.forfeitReserve.toLocaleString()}</span>
                      </div>
                    )}
                    {preview.inertBundle > 0 && (
                      <div className="text-sm text-muted-foreground">
                        {preview.inertBundle.toLocaleString()} monthly credits expire with the team
                      </div>
                    )}
                  </div>
                )}
              </>
            )
          )}

          <div className="space-y-1.5 pt-1">
            <Label htmlFor="dissolve-confirm">
              Type <span className="font-semibold text-foreground">{orgName}</span> to confirm
            </Label>
            <Input id="dissolve-confirm" value={value} onChange={(e) => setValue(e.target.value)} autoComplete="off" />
          </div>
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={() => handleOpenChange(false)} disabled={dissolve.isPending}>
            Cancel
          </Button>
          <Button variant="destructive" onClick={handleConfirm} disabled={!confirmed || dissolve.isPending}>
            {dissolve.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            Dissolve team
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
