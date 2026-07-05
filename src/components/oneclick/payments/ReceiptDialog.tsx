// src/components/oneclick/payments/ReceiptDialog.tsx
import { useEffect, useState } from "react";
import { Download, FolderPlus, Loader2, CheckCheck } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import { useSaveReceiptToProject } from "@/hooks/useRoyalties";
import type { PayoutOut } from "@/hooks/useRoyalties";
import { PartyAvatar, fmtMoney, fmtDate, downloadPdf } from "./shared";
import { useToast } from "@/hooks/use-toast";

interface Artist {
  id: string;
  name: string;
}

interface Project {
  id: string;
  name: string;
}

interface ReceiptDialogProps {
  payout: PayoutOut;
  onClose: () => void;
}

function getPayeeName(payout: PayoutOut): string {
  const snap = payout.breakdown_snapshot as Record<string, unknown>;
  const payee = snap?.payee as Record<string, unknown> | undefined;
  return (payee?.display_name as string | undefined) ?? "Payee";
}

export function ReceiptDialog({ payout, onClose }: ReceiptDialogProps) {
  const { toast } = useToast();
  const saveReceipt = useSaveReceiptToProject();

  const [downloading, setDownloading] = useState(false);
  const [saved, setSaved] = useState(false);

  const [artists, setArtists] = useState<Artist[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [artistId, setArtistId] = useState("");
  const [projectId, setProjectId] = useState("");
  const [loadingProjects, setLoadingProjects] = useState(false);

  const payeeName = getPayeeName(payout);

  useEffect(() => {
    apiFetch<Artist[]>(`${API_URL}/artists`)
      .then((data) => setArtists(data || []))
      .catch(() => setArtists([]));
  }, []);

  useEffect(() => {
    setProjectId("");
    if (!artistId) {
      setProjects([]);
      return;
    }
    setLoadingProjects(true);
    apiFetch<Project[]>(`${API_URL}/projects/${artistId}`)
      .then((data) => setProjects(data || []))
      .catch(() => setProjects([]))
      .finally(() => setLoadingProjects(false));
  }, [artistId]);

  const handleDownload = async () => {
    setDownloading(true);
    try {
      await downloadPdf(
        `${API_URL}/oneclick/royalties/payouts/${payout.id}/receipt.pdf`,
        `Receipt_${payeeName.replace(/\s+/g, "_")}.pdf`,
      );
    } catch {
      toast({ variant: "destructive", title: "Couldn't download the receipt. Please try again." });
    } finally {
      setDownloading(false);
    }
  };

  const handleSave = () => {
    setSaved(false);
    saveReceipt.mutate(
      { payoutId: payout.id, artist_id: artistId, project_id: projectId },
      {
        onSuccess: () => {
          setSaved(true);
          toast({ title: "Receipt saved to the project's files" });
        },
        onError: (err) =>
          toast({
            variant: "destructive",
            title: "Couldn't save the receipt",
            description: err instanceof Error ? err.message : "An error occurred.",
          }),
      },
    );
  };

  return (
    <Dialog open onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent className="max-w-md gap-0 overflow-y-auto p-0 max-h-[90vh]">
        <DialogHeader className="border-b border-border px-5 py-4">
          <DialogTitle>Payment receipt</DialogTitle>
        </DialogHeader>

        <div className="flex flex-col gap-4 px-5 py-5">
          {/* Payment summary */}
          <div className="flex items-center justify-between rounded-xl border border-border bg-card p-3.5">
            <span className="flex min-w-0 items-center gap-2.5">
              <PartyAvatar id={payout.payee_id} name={payeeName} size={34} />
              <span className="min-w-0">
                <div className="text-[14px] font-semibold">{payeeName}</div>
                <div className="text-[12px] text-muted-foreground">
                  Paid {payout.paid_at ? fmtDate(payout.paid_at) : "—"}
                  {payout.payment_method === "paypal" ? " · PayPal" : ""}
                </div>
              </span>
            </span>
            <span className="text-right">
              <div className="font-mono text-[17px] font-bold tabular-nums tracking-tight">
                {fmtMoney(payout.total_amount, payout.pay_currency)}
              </div>
              <div className="text-[11.5px] text-muted-foreground">{payout.pay_currency}</div>
            </span>
          </div>

          {/* Download */}
          <Button disabled={downloading} onClick={handleDownload}>
            {downloading ? (
              <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
            ) : (
              <Download className="mr-1.5 h-4 w-4" />
            )}
            Download receipt
          </Button>

          {/* Save to project */}
          <div className="rounded-xl border border-border p-3.5">
            <div className="mb-2.5 flex items-center gap-1.5 text-[13px] font-semibold">
              <FolderPlus className="h-4 w-4 text-primary" /> Save to a project
            </div>
            <p className="mb-3 text-[12px] leading-relaxed text-muted-foreground">
              Keep a copy of this receipt in a project's files so it's easy to find later.
            </p>
            <div className="flex flex-col gap-2">
              <Select value={artistId} onValueChange={setArtistId}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose an artist" />
                </SelectTrigger>
                <SelectContent>
                  {artists.map((a) => (
                    <SelectItem key={a.id} value={a.id}>
                      {a.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={projectId} onValueChange={setProjectId} disabled={!artistId || loadingProjects}>
                <SelectTrigger>
                  <SelectValue placeholder={loadingProjects ? "Loading projects…" : "Choose a project"} />
                </SelectTrigger>
                <SelectContent>
                  {projects.map((p) => (
                    <SelectItem key={p.id} value={p.id}>
                      {p.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                variant="outline"
                disabled={!artistId || !projectId || saveReceipt.isPending}
                onClick={handleSave}
              >
                {saveReceipt.isPending ? (
                  <>
                    <Loader2 className="mr-1.5 h-4 w-4 animate-spin" /> Saving…
                  </>
                ) : saved ? (
                  <>
                    <CheckCheck className="mr-1.5 h-4 w-4" /> Saved
                  </>
                ) : (
                  "Save receipt to project"
                )}
              </Button>
            </div>
          </div>

          <div className="flex justify-end">
            <Button variant="ghost" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
