// Account & Billing → "My API usage": what the caller's OWN partner keys spent,
// per team. Pool numbers stay admin-only; this is their keys and nothing else,
// so a plain member can see it.
import { useState } from "react";
import { KeyRound, Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { useCreditUsage, useMyApiUsage, type MyOrgApiUsage } from "@/hooks/useCreditUsage";
import type { UsageRange } from "@/hooks/useOrgs";
import { groupKeysByFolder, type UsageSubject } from "@/lib/orgUsage";
import { KeysByFolderTable, RangePicker, ReportButton } from "@/components/orgs/usageTableBits";
import { UsageDetailDialog } from "@/components/orgs/UsageDetailDialog";
import { API_URL } from "@/lib/apiFetch";

export function MyApiUsage() {
  const [range, setRange] = useState<UsageRange>("mtd");
  // One dialog for the card; the org rides along so its window and totals
  // frame the numbers.
  const [selected, setSelected] = useState<{ s: UsageSubject; org: MyOrgApiUsage } | null>(null);
  const { data, isSuccess, isError } = useMyApiUsage(range);
  // Same gate as the card above: flag off -> no credit surfaces.
  const { data: creditUsage } = useCreditUsage();
  const orgs = data?.orgs ?? [];
  const credits = orgs.reduce((n, o) => n + o.credits, 0);
  const runs = orgs.reduce((n, o) => n + o.runs, 0);

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
          <RangePicker value={range} onChange={setRange} label="API usage range" />
          <ReportButton
            url={`${API_URL}/me/api-usage/report.pdf?range=${range}`}
            filename={`my-api-usage-${range}.pdf`}
            disabled={!isSuccess}
          />
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
