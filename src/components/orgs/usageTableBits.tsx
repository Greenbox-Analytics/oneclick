// src/components/orgs/usageTableBits.tsx
// Card furniture shared by the team Usage card and the profile's My API usage
// card: range picker, report download, tiles, tables. Rows come in as props.
// No per-tool COLUMNS — they don't scale as tools are added, so each row
// carries one MixBar and the breakdown lives in the detail dialog.
import { useState } from "react";
import { Download, Loader2 } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { downloadPdf } from "@/lib/downloadPdf";
import type { UsageRange } from "@/hooks/useOrgs";
import { TOOLS } from "@/lib/usageTools";
import {
  folderSubject,
  keySubject,
  memberSubject,
  USAGE_RANGES,
  type FolderRow,
  type KeyUsageRow,
  type MemberRow,
  type ToolTotals,
  type UsageSubject,
} from "@/lib/orgUsage";
import { fmtDate } from "@/lib/utils";

/** The window switcher both cards carry. `label` names the group, since a
 * page can show more than one. */
export function RangePicker({
  value,
  onChange,
  label,
}: {
  value: UsageRange;
  onChange: (range: UsageRange) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border border-border bg-muted/40 p-0.5">
      {USAGE_RANGES.map((r) => (
        <button
          key={r.id}
          type="button"
          role="radio"
          aria-checked={value === r.id}
          title={r.title}
          onClick={() => onChange(r.id)}
          className={`rounded-md px-2.5 py-1 text-[12px] font-semibold transition-colors ${value === r.id ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
        >
          {r.label}
        </button>
      ))}
    </div>
  );
}

/** Downloads the report at `url`, owning its own in-flight state. */
export function ReportButton({ url, filename, disabled }: { url: string; filename: string; disabled?: boolean }) {
  const { toast } = useToast();
  const [downloading, setDownloading] = useState(false);

  const download = async () => {
    setDownloading(true);
    try {
      await downloadPdf(url, filename);
    } catch {
      toast({ title: "Couldn't download the report", description: "Please try again.", variant: "destructive" });
    } finally {
      setDownloading(false);
    }
  };

  return (
    <Button size="sm" variant="outline" onClick={download} disabled={disabled || downloading}>
      {downloading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
      Download PDF
    </Button>
  );
}

export const Swatch = ({ color }: { color: string }) => (
  <span className="mr-1.5 inline-block h-2 w-2 rounded-sm align-middle" style={{ background: color }} />
);

export function Tile({
  label,
  value,
  sub,
  change,
  vs,
}: {
  label: string;
  value: string;
  sub?: string;
  change?: number | null;
  vs?: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-muted/30 px-4 py-3">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-1 text-[22px] font-bold tabular-nums text-foreground">{value}</div>
      {sub && <div className="text-[12px] text-muted-foreground">{sub}</div>}
      {change != null && (
        <div className="text-[12px] text-muted-foreground">
          {change > 0 ? "+" : ""}
          {change}% vs {vs}
        </div>
      )}
    </div>
  );
}

/** One row's tool split as a stacked bar; the numbers live in its label and
 * the detail dialog. */
export function MixBar({ tools, total }: { tools: ToolTotals; total: number }) {
  const spent = TOOLS.filter((t) => tools[t.id] > 0);
  const label = spent.length
    ? spent.map((t) => `${t.label} ${tools[t.id].toLocaleString()}`).join(" · ")
    : "No credits used";
  return (
    <span role="img" aria-label={label} title={label} className="flex h-2 w-full overflow-hidden rounded-full bg-muted">
      {spent.map((t) => (
        <span
          key={t.id}
          className="h-2"
          style={{ flexBasis: `${total ? (tools[t.id] / total) * 100 : 0}%`, background: t.color }}
        />
      ))}
    </span>
  );
}

/** The "By tool" bar list — always every tool, so a zero reads as a zero. */
export function ToolMix({ tools, total }: { tools: ToolTotals; total: number }) {
  return (
    <ul className="flex flex-col gap-1.5">
      {TOOLS.map((t) => {
        const share = total ? tools[t.id] / total : 0;
        return (
          <li key={t.id} className="grid grid-cols-[100px_1fr_auto] items-center gap-3 text-[12.5px]">
            <span>
              <Swatch color={t.color} />
              {t.label}
            </span>
            <span className="h-2 rounded-full bg-muted">
              <span
                className="block h-2 rounded-full"
                style={{ width: `${Math.round(share * 100)}%`, background: t.color }}
              />
            </span>
            <span className="tabular-nums text-muted-foreground">
              {tools[t.id].toLocaleString()} · {Math.round(share * 100)}%
            </span>
          </li>
        );
      })}
    </ul>
  );
}

type Selectable = { onSelect?: (s: UsageSubject) => void };

/** Row chrome for a clickable row; the name cell is a real button, so the row
 * is keyboard-reachable. */
const rowProps = (open: (() => void) | undefined) =>
  open ? { className: "cursor-pointer hover:bg-muted/40", onClick: open } : {};

const NameButton = ({ open, children }: { open?: () => void; children: React.ReactNode }) =>
  open ? (
    <button type="button" className="text-left font-medium hover:underline" onClick={open}>
      {children}
    </button>
  ) : (
    <span className="font-medium">{children}</span>
  );

/** Total / Runs / Mix — the three cells every usage row ends on. `loaded` is
 * false until the query resolves; a 0 there would read as "spent nothing". */
function SpendCells({
  total,
  runs,
  tools,
  loaded,
}: {
  total: number;
  runs: number;
  tools: ToolTotals;
  loaded: boolean;
}) {
  return (
    <>
      <TableCell className="text-right font-semibold tabular-nums">{loaded ? total.toLocaleString() : "—"}</TableCell>
      <TableCell className="text-right tabular-nums">{loaded ? runs.toLocaleString() : "—"}</TableCell>
      <TableCell className="w-[160px]">
        <MixBar tools={tools} total={total} />
      </TableCell>
    </>
  );
}

/** A key's name cell: label over prefix · status. Indented when grouped. */
function KeyNameCell({ r, open, indent }: { r: KeyUsageRow; open?: () => void; indent?: boolean }) {
  return (
    <TableCell className={indent ? "pl-8" : undefined}>
      <div className="text-sm">
        <NameButton open={open}>{r.row.label}</NameButton>
      </div>
      <div className="font-mono text-[11px] text-muted-foreground">
        {r.row.keyPrefix}…{r.row.status !== "active" ? ` · ${r.row.status}` : ""}
      </div>
    </TableCell>
  );
}

export function MembersTable({
  rows,
  loaded,
  showKeys,
  onSelect,
}: { rows: MemberRow[]; loaded: boolean; showKeys: boolean } & Selectable) {
  if (rows.length === 0) return <p className="py-6 text-center text-[13px] text-muted-foreground">No members yet.</p>;
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Member</TableHead>
            <TableHead className="text-right">Total</TableHead>
            <TableHead className="text-right">Runs</TableHead>
            <TableHead className="w-[160px]">Mix</TableHead>
            {showKeys && <TableHead>Keys created</TableHead>}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r) => {
            const open = onSelect && (() => onSelect(memberSubject(r)));
            return (
              <TableRow key={r.seat.orgMemberId} {...rowProps(open)}>
                <TableCell>
                  <div className="truncate text-sm">
                    <NameButton open={open}>{r.seat.email ?? "Unknown"}</NameButton>
                  </div>
                  <div className="text-[11px] capitalize text-muted-foreground">
                    {r.seat.role}
                    {r.seat.status !== "active" ? ` · ${r.seat.status}` : ""}
                  </div>
                </TableCell>
                <SpendCells total={r.total} runs={r.runs} tools={r.tools} loaded={loaded} />
                {showKeys && (
                  <TableCell>
                    {r.keys.length === 0 ? (
                      <span className="text-muted-foreground">—</span>
                    ) : (
                      <div className="flex flex-wrap gap-1">
                        {r.keys.map((k) => (
                          <Badge
                            key={k.id}
                            variant={k.status === "active" ? "secondary" : "outline"}
                            className="font-normal"
                          >
                            {k.label}
                            <span className="ml-1 font-mono text-[10px] text-muted-foreground">{k.key_prefix}…</span>
                          </Badge>
                        ))}
                      </div>
                    )}
                  </TableCell>
                )}
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

export function KeysTable({ rows, loaded, onSelect }: { rows: KeyUsageRow[]; loaded: boolean } & Selectable) {
  if (rows.length === 0) return <p className="py-6 text-center text-[13px] text-muted-foreground">No API keys yet.</p>;
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Key</TableHead>
            <TableHead>Folder</TableHead>
            <TableHead className="text-right">Total</TableHead>
            <TableHead className="text-right">Runs</TableHead>
            <TableHead className="w-[160px]">Mix</TableHead>
            <TableHead>Last used</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r) => {
            const { row, total, tools } = r;
            const open = onSelect && (() => onSelect(keySubject(r)));
            return (
              <TableRow key={row.keyId} {...rowProps(open)}>
                <KeyNameCell r={r} open={open} />
                <TableCell className="text-muted-foreground">{row.folderName ?? "—"}</TableCell>
                <SpendCells total={total} runs={r.runs} tools={tools} loaded={loaded} />
                <TableCell className="text-muted-foreground">{fmtDate(row.lastUsedAt)}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

export function FoldersTable({ rows, loaded, onSelect }: { rows: FolderRow[]; loaded: boolean } & Selectable) {
  if (rows.length === 0) return <p className="py-6 text-center text-[13px] text-muted-foreground">No folders yet.</p>;
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Folder</TableHead>
            <TableHead className="text-right">Keys</TableHead>
            <TableHead className="text-right">Total</TableHead>
            <TableHead className="text-right">Runs</TableHead>
            <TableHead className="w-[160px]">Mix</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r) => {
            const { row, total, tools } = r;
            const open = onSelect && (() => onSelect(folderSubject(r)));
            return (
              <TableRow key={row.folderId ?? "unfiled"} {...rowProps(open)}>
                <TableCell className="text-sm">
                  <NameButton open={open}>{row.name}</NameButton>
                </TableCell>
                <TableCell className="text-right tabular-nums">{row.keys.toLocaleString()}</TableCell>
                <SpendCells total={total} runs={r.runs} tools={tools} loaded={loaded} />
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

/** The profile card's table: a folder heading row with its subtotal, then its
 * keys. Both levels open the detail dialog. */
export function KeysByFolderTable({
  groups,
  loaded,
  onSelect,
}: { groups: { folder: FolderRow; keys: KeyUsageRow[] }[]; loaded: boolean } & Selectable) {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Key</TableHead>
            <TableHead className="text-right">Total</TableHead>
            <TableHead className="text-right">Runs</TableHead>
            <TableHead className="w-[160px]">Mix</TableHead>
            <TableHead>Last used</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {groups.map(({ folder, keys }) => {
            const openFolder = onSelect && (() => onSelect(folderSubject(folder)));
            return [
              <TableRow key={`f-${folder.row.folderId ?? "unfiled"}`} className="bg-muted/30" {...rowProps(openFolder)}>
                <TableCell>
                  <div className="text-sm">
                    <NameButton open={openFolder}>{folder.row.name}</NameButton>
                  </div>
                  <div className="text-[11px] text-muted-foreground">
                    {keys.length} {keys.length === 1 ? "key" : "keys"}
                  </div>
                </TableCell>
                <SpendCells total={folder.total} runs={folder.runs} tools={folder.tools} loaded={loaded} />
                <TableCell />
              </TableRow>,
              ...(keys.length === 0
                ? [
                    <TableRow key={`f-${folder.row.folderId ?? "unfiled"}-empty`}>
                      <TableCell colSpan={5} className="text-[12.5px] text-muted-foreground">
                        No keys yet
                      </TableCell>
                    </TableRow>,
                  ]
                : keys.map((r) => {
                    const open = onSelect && (() => onSelect(keySubject(r)));
                    return (
                      <TableRow key={r.row.keyId} {...rowProps(open)}>
                        <KeyNameCell r={r} open={open} indent />
                        <SpendCells total={r.total} runs={r.runs} tools={r.tools} loaded={loaded} />
                        <TableCell className="text-muted-foreground">{fmtDate(r.row.lastUsedAt)}</TableCell>
                      </TableRow>
                    );
                  })),
            ];
          })}
        </TableBody>
      </Table>
    </div>
  );
}
