// src/components/profile/MyApiUsage.tsx
// Account & Billing → "My API usage": what the person's OWN partner API keys
// spent, per team. A team's pool numbers stay admin-only (GET /orgs/{id}/usage);
// this is the caller's own keys and nothing else, so a plain member can see it.
import { useState } from "react";
import { Download, KeyRound, Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useCreditUsage, useMyApiUsage, type MyOrgApiUsage } from "@/hooks/useCreditUsage";
import type { UsageRange } from "@/hooks/useOrgs";
import { groupKeysByFolder, USAGE_RANGES, type UsageSubject } from "@/lib/orgUsage";
import { KeysByFolderTable } from "@/components/orgs/usageTableBits";
import { UsageDetailDialog } from "@/components/orgs/UsageDetailDialog";
import { useToast } from "@/hooks/use-toast";
import { API_URL } from "@/lib/apiFetch";
import { downloadPdf } from "@/lib/downloadPdf";

export function MyApiUsage() {
  const [range, setRange] = useState<UsageRange>("mtd");
  // One dialog for the whole card — the org it belongs to rides along so its
  // window and totals frame the numbers correctly.
  const [selected, setSelected] = useState<{ s: UsageSubject; org: MyOrgApiUsage } | null>(null);
  const [downloading, setDownloading] = useState(false);
  const { toast } = useToast();
  const { data, isSuccess, isError } = useMyApiUsage(range);
  // Same gate as the Credits & usage card above it: flag off -> no credit surfaces.
  const { data: creditUsage } = useCreditUsage();
  const orgs = data?.orgs ?? [];
  const credits = orgs.reduce((n, o) => n + o.credits, 0);
  const runs = orgs.reduce((n, o) => n + o.runs, 0);

  const download = async () => {
    setDownloading(true);
    try {
      await downloadPdf(`${API_URL}/me/api-usage/report.pdf?range=${range}`, `my-api-usage-${range}.pdf`);
    } catch {
      toast({ title: "Couldn't download the report", description: "Please try again.", variant: "destructive" });
    } finally {
      setDownloading(false);
    }
  };

  if (!creditUsage?.enabled) return null;

  return (
    <Card className="flex flex-col gap-4 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex flex-col gap-1">
          <h3 className="flex items-center gap-2 text-[15px] font-semibold">
            <KeyRound className="h-4 w-4" /> My API usage
          </h3>
          <p className="text-[13px] text-muted-foreground">
            {isError
              ? "Couldn't load your API usage."
              : isSuccess
                ? `${credits.toLocaleString()} credits · ${runs.toLocaleString()} runs in this window`
                : "— credits · — runs in this window"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div
            role="radiogroup"
            aria-label="API usage range"
            className="inline-flex rounded-lg border border-border bg-muted/40 p-0.5"
          >
            {USAGE_RANGES.map((r) => (
              <button
                key={r.id}
                type="button"
                role="radio"
                aria-checked={range === r.id}
                title={r.title}
                onClick={() => setRange(r.id)}
                className={`rounded-md px-2.5 py-1 text-[12px] font-semibold transition-colors ${range === r.id ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
              >
                {r.label}
              </button>
            ))}
          </div>
          <Button size="sm" variant="outline" onClick={download} disabled={!isSuccess || downloading}>
            {downloading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
            Download PDF
          </Button>
        </div>
      </div>

      {isError ? null : !isSuccess ? (
        <div className="flex justify-center py-8">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      ) : orgs.length === 0 ? (
        <p className="py-6 text-center text-[13px] text-muted-foreground">
          You haven&apos;t created any API keys yet.
        </p>
      ) : (
        orgs.map((org) => (
          <section key={org.orgId} className="flex flex-col gap-2">
            <h4 className="text-[13px] font-semibold">{org.orgName ?? "Team"}</h4>
            <KeysByFolderTable
              groups={groupKeysByFolder(org.byKey, org.byFolder)}
              loaded
              onSelect={(s) => setSelected({ s, org })}
            />
          </section>
        ))
      )}

      <UsageDetailDialog
        subject={selected?.s ?? null}
        range={range}
        since={selected?.org.since ?? null}
        windowTotal={selected?.org.credits ?? 0}
        onClose={() => setSelected(null)}
      />
    </Card>
  );
}
