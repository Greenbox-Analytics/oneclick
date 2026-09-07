// Admin console: a partner-enabled org's API keys, with this period's per-key
// spend from useOrgUsage().byKey. Mounted only when org.partner_api_enabled.
import { useState } from "react";
import { Link } from "react-router-dom";
import { BookOpen, KeyRound, Loader2, Plus } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useOrgUsage } from "@/hooks/useOrgs";
import {
  usePartnerKeys,
  useRevokePartnerKey,
  useSetPartnerKeyFolder,
  type PartnerKeyFolder,
} from "@/hooks/usePartnerKeys";
import { keyRows, type KeyRow as KeyRowData, type KeyStatus } from "@/lib/partnerKeys";
import { fmtDate } from "@/lib/utils";
import { CreateApiKeyDialog } from "./CreateApiKeyDialog";
import { FolderSelect, useFolderChoice } from "./FolderSelect";

const STATUS_LABEL: Record<KeyStatus, string> = { active: "Active", revoked: "Revoked", expired: "Expired" };
const STATUS_VARIANT: Record<KeyStatus, "default" | "secondary" | "outline"> = {
  active: "default",
  revoked: "secondary",
  expired: "outline",
};

function KeyRow({
  row,
  folderName,
  usageLoaded,
  onRevoke,
  onMove,
}: {
  row: KeyRowData;
  folderName: string | null;
  /** False until the usage query resolves; a 0 would read as "spent nothing". */
  usageLoaded: boolean;
  onRevoke: (r: KeyRowData) => void;
  onMove: (r: KeyRowData) => void;
}) {
  const { key, status, usage } = row;
  return (
    <TableRow>
      <TableCell className="font-medium">{key.label}</TableCell>
      <TableCell className="text-muted-foreground">{folderName ?? "—"}</TableCell>
      <TableCell className="font-mono text-[12px] text-muted-foreground">{key.key_prefix}…</TableCell>
      <TableCell className="text-muted-foreground">{key.created_by_label ?? "—"}</TableCell>
      <TableCell>
        <Badge variant={STATUS_VARIANT[status]}>{STATUS_LABEL[status]}</Badge>
      </TableCell>
      <TableCell className="text-right tabular-nums">{usageLoaded ? (usage?.credits ?? 0) : "—"}</TableCell>
      <TableCell className="text-right tabular-nums">{usageLoaded ? (usage?.runs ?? 0) : "—"}</TableCell>
      {/* last_used_at is stamped on every resolve; usage.lastUsedAt only on a
          debited run this period. */}
      <TableCell className="text-muted-foreground">{fmtDate(key.last_used_at ?? usage?.lastUsedAt)}</TableCell>
      <TableCell className="text-right whitespace-nowrap">
        <Button variant="ghost" size="sm" aria-label={`Move ${key.label} to a folder`} onClick={() => onMove(row)}>
          Move to folder
        </Button>
        {status === "active" && (
          <Button variant="ghost" size="sm" aria-label={`Revoke key for ${key.label}`} onClick={() => onRevoke(row)}>
            Revoke
          </Button>
        )}
      </TableCell>
    </TableRow>
  );
}

export function OrgApiKeysPanel({ orgId }: { orgId: string }) {
  const { data, isLoading, isError } = usePartnerKeys(orgId);
  const { data: usage, isSuccess: usageLoaded } = useOrgUsage(orgId);
  const revoke = useRevokePartnerKey();
  const [createOpen, setCreateOpen] = useState(false);
  const [showInactive, setShowInactive] = useState(false);
  const [pending, setPending] = useState<KeyRowData | null>(null);
  const [moving, setMoving] = useState<KeyRowData | null>(null);
  const folders = data?.folders ?? [];

  // Never fall through to the empty state on a failed load — it invites
  // minting a duplicate of a key they already have.
  if (isError) {
    return (
      <Card className="p-6 text-sm text-muted-foreground text-center py-10">
        Couldn&apos;t load API keys. Please try refreshing.
      </Card>
    );
  }

  const rows = keyRows(data?.keys ?? [], usage?.byKey);
  const folderName = new Map(folders.map((f) => [f.id, f.name]));
  const visible = showInactive ? rows : rows.filter((r) => r.status === "active");
  const inactiveCount = rows.filter((r) => r.status !== "active").length;

  const confirmRevoke = () => {
    if (!pending) return;
    revoke.mutate({ orgId, keyId: pending.key.id }, { onSettled: () => setPending(null) });
  };

  return (
    <Card className="p-5 flex flex-col gap-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h3 className="text-[15px] font-semibold flex items-center gap-2">
            <KeyRound className="w-4 h-4" /> API keys
          </h3>
          <p className="text-[13px] text-muted-foreground">
            Keys your systems use to run royalty calculations through Msanii. Each one is this team&apos;s credential and
            spends from its credits — keep them secret.
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button asChild size="sm" variant="ghost">
            <Link to="/docs?section=api">
              <BookOpen className="w-4 h-4 mr-1.5" /> API docs
            </Link>
          </Button>
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <Plus className="w-4 h-4 mr-1.5" /> New API key
          </Button>
        </div>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-8">
          <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
        </div>
      ) : rows.length === 0 ? (
        <p className="text-[13px] text-muted-foreground py-6 text-center">No API keys yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Folder</TableHead>
                <TableHead>Key</TableHead>
                <TableHead>Created by</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Credits this period</TableHead>
                <TableHead className="text-right">Runs</TableHead>
                <TableHead>Last used</TableHead>
                <TableHead />
              </TableRow>
            </TableHeader>
            <TableBody>
              {visible.map((r) => (
                <KeyRow
                  key={r.key.id}
                  row={r}
                  folderName={r.key.folder_id ? (folderName.get(r.key.folder_id) ?? null) : null}
                  usageLoaded={usageLoaded}
                  onRevoke={setPending}
                  onMove={setMoving}
                />
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {inactiveCount > 0 && (
        <div>
          <Button variant="ghost" size="sm" onClick={() => setShowInactive((v) => !v)}>
            {showInactive ? "Hide inactive" : `Show inactive (${inactiveCount})`}
          </Button>
          <p className="text-[12px] text-muted-foreground">
            Revoked and expired keys stay listed for 30 days, then disappear. Their spend still counts.
          </p>
        </div>
      )}

      <CreateApiKeyDialog orgId={orgId} folders={folders} open={createOpen} onOpenChange={setCreateOpen} />

      {/* Keyed so the picker re-seeds from the row's folder each time. */}
      <MoveToFolderDialog
        key={moving?.key.id ?? "none"}
        orgId={orgId}
        folders={folders}
        row={moving}
        onClose={() => setMoving(null)}
      />

      <Dialog open={!!pending} onOpenChange={(o) => !o && setPending(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Revoke "{pending?.key.label}"?</DialogTitle>
            <DialogDescription>Anything using this key stops working immediately.</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPending(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmRevoke} disabled={revoke.isPending}>
              {revoke.isPending && <Loader2 className="w-4 h-4 mr-1.5 animate-spin" />}
              Revoke key
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

/** Reassigns a key's folder. Spend follows the CURRENT folder, so this moves
 * its history too — the copy says as much. */
function MoveToFolderDialog({
  orgId,
  folders,
  row,
  onClose,
}: {
  orgId: string;
  folders: PartnerKeyFolder[];
  row: KeyRowData | null;
  onClose: () => void;
}) {
  const setFolder = useSetPartnerKeyFolder();
  const choice = useFolderChoice(orgId, row?.key.folder_id ?? null);

  const submit = () => {
    if (!row || choice.missing) return;
    choice.resolve((folderId) => setFolder.mutate({ orgId, keyId: row.key.id, folderId }, { onSettled: onClose }));
  };

  return (
    <Dialog open={!!row} onOpenChange={(o) => !o && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Move &quot;{row?.key.label}&quot;</DialogTitle>
          <DialogDescription>
            Folders group keys by use case. This key&apos;s spend is reported under whichever folder it&apos;s in.
          </DialogDescription>
        </DialogHeader>
        <FolderSelect id="move-key-folder" folders={folders} choice={choice} />
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={submit} disabled={!!choice.missing || setFolder.isPending || choice.isPending}>
            {(setFolder.isPending || choice.isPending) && <Loader2 className="w-4 h-4 mr-1.5 animate-spin" />}
            Move key
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
