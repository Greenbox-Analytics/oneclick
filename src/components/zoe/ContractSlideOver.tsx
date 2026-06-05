import { useState, useEffect } from "react";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Loader2 } from "lucide-react";
import { apiFetch, API_URL } from "@/lib/apiFetch";

interface ContractSlideOverProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  contractId: string | null;
  contractName: string | null;
  /** Optional 1-based page to jump to in the PDF viewer. */
  page?: number | null;
}

/**
 * Opens the contract's original PDF in a right-side panel (browser-native PDF viewer),
 * jumping to `page` when provided. The PDF is loaded via a short-lived signed URL.
 */
export function ContractSlideOver({ open, onOpenChange, contractId, contractName, page }: ContractSlideOverProps) {
  const [url, setUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !contractId) {
      setUrl(null);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    apiFetch<{ url: string }>(`${API_URL}/contracts/${contractId}/pdf-url`)
      .then((data) => setUrl(data.url || ""))
      .catch(() => setError("Could not open this contract."))
      .finally(() => setLoading(false));
  }, [open, contractId]);

  // PDF fragment params (browser native viewers honor #page= and view=). Query string + fragment coexist.
  const src = url ? `${url}#page=${page && page > 0 ? page : 1}&view=FitH` : "";

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-full sm:max-w-2xl lg:max-w-3xl p-0 flex flex-col">
        {/* pr-14 keeps the title clear of the Sheet's built-in close X (absolute right-4). */}
        <SheetHeader className="pl-5 pr-14 py-3 border-b border-border text-left flex-row items-center space-y-0">
          <SheetTitle className="text-sm font-semibold truncate">{contractName ?? "Contract"}</SheetTitle>
        </SheetHeader>

        <div className="flex-1 min-h-0 bg-muted/30">
          {loading ? (
            <div className="flex h-full items-center justify-center gap-2 text-muted-foreground">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">Loading contract…</span>
            </div>
          ) : error ? (
            <p className="py-8 text-center text-sm text-destructive">{error}</p>
          ) : src ? (
            <iframe src={src} title={contractName ?? "Contract"} className="h-full w-full border-0" />
          ) : null}
        </div>
      </SheetContent>
    </Sheet>
  );
}
