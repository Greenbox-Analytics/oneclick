import { useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  Eye,
  File as FileIcon,
  FileText,
  Folder,
  Image as ImageIcon,
  Lock,
  Music,
  Plus,
  Search,
  Shield,
  Sparkles,
  Trash2,
  TrendingUp,
  Users,
  X,
} from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import {
  useWorkFull,
  useWorkGrants,
  useAddGrants,
  useRemoveGrants,
  useSetAccessLevel,
  useSetWorkRole,
  useRevokeCollaborator,
  type GrantItem,
} from "@/hooks/useRegistry";
import { useWorkFiles } from "@/hooks/useWorkFiles";
import { useWorkAudio } from "@/hooks/useWorkAudio";

import { RegistryAvatar } from "./RegistryAvatar";
import DeriveCollaboratorSplitDialog from "./DeriveCollaboratorSplitDialog";
import "./permissions-panel.css";

// ============================================================
// Constants
// ============================================================

const WORK_ROLES = [
  "Producer",
  "Co-Producer",
  "Songwriter",
  "Featured Artist",
  "Mix Engineer",
  "Manager",
  "Legal / Business",
];

type AccessLevel = "viewer" | "admin";

const ACCESS_DESC: Record<AccessLevel, string> = {
  viewer: "Can view only the items you share below — nothing else.",
  admin: "Full access — sees and edits everything on this work.",
};

// Resource types that map to a per-resource grant in the draft Set.
type ResourceType = "project_file" | "audio_file" | "license" | "agreement";

// Folder categories treated as contracts in the registry.
const CONTRACT_CATEGORIES = new Set(["contract", "split_sheet", "royalty_statement"]);

const FILE_KIND_LABEL: Record<string, string> = {
  contract: "Contract",
  split_sheet: "Split Sheet",
  royalty_statement: "Statement",
};

// ============================================================
// Types
// ============================================================

interface DocRowData {
  id: string;
  name: string;
  meta: string;
  /** doc kind used for the type icon */
  kind: "contract" | "audio" | "image" | "license" | "agreement" | "file";
  /** composite grant key `${resource_type}:${resource_id}` */
  key: string;
}

interface DocGroup {
  key: string;
  label: string;
  iconNode: JSX.Element;
  items: DocRowData[];
}

interface CollaboratorLite {
  id: string;
  name: string;
  email: string;
  role: string;
  access_level: string; // "viewer" | "admin"
  status: string;
}

export interface CollaboratorAccessPanelProps {
  mode: "invite" | "edit";
  workId: string;
  /** Required in edit mode — the person whose access is being managed. */
  collaborator?: CollaboratorLite;
  /** Optional project name for the subtitle; falls back to work title alone. */
  projectName?: string;
  /** Roster artists for prefill (invite mode). */
  artists?: Array<{ id: string; name: string; email: string }>;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

// ============================================================
// Helpers
// ============================================================

function fmtDate(value: string | null | undefined): string {
  if (!value) return "";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function kindIcon(kind: DocRowData["kind"]): JSX.Element {
  switch (kind) {
    case "contract":
    case "agreement":
      return <FileText />;
    case "audio":
      return <Music />;
    case "image":
      return <ImageIcon />;
    case "license":
      return <Shield />;
    default:
      return <FileIcon />;
  }
}

function imageish(fileType: string | null | undefined, name: string): boolean {
  const ft = (fileType || "").toLowerCase();
  if (ft.startsWith("image/")) return true;
  return /\.(png|jpe?g|gif|webp|svg|heic)$/i.test(name || "");
}

// ============================================================
// Sub-components (all at module top level — no nested definitions)
// ============================================================

function SegmentedAccess({
  value,
  onChange,
}: {
  value: AccessLevel;
  onChange: (v: AccessLevel) => void;
}) {
  const options: Array<{ value: AccessLevel; label: string; icon: JSX.Element }> = [
    { value: "viewer", label: "Viewer", icon: <Eye /> },
    { value: "admin", label: "Admin", icon: <Shield /> },
  ];
  return (
    <div className="perm-segmented">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          data-active={value === o.value ? "true" : "false"}
          onClick={() => onChange(o.value)}
        >
          {o.icon}
          {o.label}
        </button>
      ))}
    </div>
  );
}

function AccessDescriptor({ level }: { level: AccessLevel }) {
  return (
    <div className="perm-access-desc">
      {level === "admin" ? <Shield /> : <Eye />}
      <span>{ACCESS_DESC[level]}</span>
    </div>
  );
}

function AdminCallout() {
  return (
    <div className="perm-callout">
      <div className="perm-callout-ic">
        <Shield />
      </div>
      <div>
        <p className="perm-callout-title">Admins see &amp; edit everything</p>
        <p className="perm-callout-desc">
          Item-by-item sharing is off for admins — they get full access to every document, the
          complete ownership breakdown, and all the details on this work. Drop to{" "}
          <b style={{ color: "hsl(var(--foreground))" }}>Viewer</b> to share only specific items.
        </p>
      </div>
    </div>
  );
}

function InfoRow({
  icon,
  title,
  desc,
  on,
  locked,
  onToggle,
}: {
  icon: JSX.Element;
  title: string;
  desc: string;
  on: boolean;
  locked?: boolean;
  onToggle?: () => void;
}) {
  const active = on || !!locked;
  return (
    <div className="perm-row" data-on={active ? "true" : "false"}>
      <div className="perm-row-ic" data-on={active ? "true" : "false"}>
        {icon}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div className="perm-row-title">
          {title}
          {locked && (
            <span className="perm-always">
              <Lock /> Always
            </span>
          )}
        </div>
        <div className="perm-row-desc">{desc}</div>
      </div>
      <Switch
        checked={active}
        disabled={locked}
        onCheckedChange={() => onToggle?.()}
        className="data-[state=checked]:bg-primary"
      />
    </div>
  );
}

function DocRow({
  doc,
  on,
  onToggle,
}: {
  doc: DocRowData;
  on: boolean;
  onToggle: (key: string) => void;
}) {
  return (
    <button
      type="button"
      className="perm-doc"
      data-on={on ? "true" : "false"}
      onClick={() => onToggle(doc.key)}
    >
      <span className="file-icon">{kindIcon(doc.kind)}</span>
      <span className="perm-doc-body">
        <span className="perm-doc-name">{doc.name}</span>
        <span className="perm-doc-meta">{doc.meta}</span>
      </span>
      <span className="perm-check" data-on={on ? "true" : "false"}>
        {on && <Check />}
      </span>
    </button>
  );
}

function DocSection({
  group,
  draft,
  onToggle,
  onSetAll,
}: {
  group: DocGroup;
  draft: Set<string>;
  onToggle: (key: string) => void;
  onSetAll: (keys: string[], on: boolean) => void;
}) {
  const [q, setQ] = useState("");
  const keys = group.items.map((d) => d.key);
  const onCount = keys.filter((k) => draft.has(k)).length;
  const allOn = keys.length > 0 && keys.every((k) => draft.has(k));
  const query = q.trim().toLowerCase();
  const shown = query
    ? group.items.filter((d) => (d.name + " " + d.meta).toLowerCase().includes(query))
    : group.items;

  return (
    <div className="perm-docsec">
      <div className="perm-group-label">
        <span className="perm-secicon">{group.iconNode}</span>
        {group.label}
        <span className="perm-count">
          · {onCount}/{keys.length} shared
        </span>
        <button
          type="button"
          className="perm-group-allbtn"
          onClick={() => onSetAll(keys, !allOn)}
        >
          {allOn ? "Clear all" : "Share all"}
        </button>
      </div>

      <div className="perm-search">
        <Search />
        <input
          className="perm-search-input"
          placeholder={`Search ${group.label.toLowerCase()}…`}
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        {q && (
          <button type="button" className="perm-search-clear" onClick={() => setQ("")}>
            <X />
          </button>
        )}
      </div>

      <div className="perm-doclist">
        {shown.length === 0 ? (
          <div className="perm-doclist-empty">No matches for &ldquo;{q}&rdquo;</div>
        ) : (
          shown.map((d) => (
            <DocRow key={d.key} doc={d} on={draft.has(d.key)} onToggle={onToggle} />
          ))
        )}
      </div>
    </div>
  );
}

function ManifestDoc({ doc }: { doc: DocRowData }) {
  return (
    <div className="mf-doc">
      <span className="file-icon">{kindIcon(doc.kind)}</span>
      <span className="mf-doc-body">
        <span className="mf-doc-name">{doc.name}</span>
        <span className="mf-doc-meta">{doc.meta}</span>
      </span>
      <Check />
    </div>
  );
}

interface ManifestSplit {
  master: number;
  publishing: number;
}

function Manifest({
  displayName,
  accessLevel,
  workRole,
  split,
  ownershipBreakdown,
  groups,
  draft,
  totalDocs,
}: {
  displayName: string;
  accessLevel: AccessLevel;
  workRole: string;
  split: ManifestSplit | null;
  ownershipBreakdown: boolean;
  groups: DocGroup[];
  draft: Set<string>;
  totalDocs: number;
}) {
  const isAdmin = accessLevel === "admin";

  // Information shared: split is always; ownership breakdown if on or admin.
  const infoLines: Array<{ key: string; label: string; locked?: boolean }> = [
    { key: "split", label: "Their own split", locked: true },
  ];
  if (isAdmin || ownershipBreakdown) {
    infoLines.push({ key: "ownership", label: "Full ownership breakdown" });
  }

  const sharedDocs = isAdmin ? totalDocs : draft.size;
  const hiddenInfo = isAdmin ? 0 : ownershipBreakdown ? 0 : 1; // ownership breakdown row
  const hiddenDocs = isAdmin ? 0 : totalDocs - draft.size;
  const hidden = hiddenInfo + hiddenDocs;

  return (
    <div className="perm-manifest">
      <div className="mf-eyebrow-top">Sharing with</div>
      <div className="mf-head">
        <RegistryAvatar name={displayName} size={36} />
        <div className="mf-head-body">
          <div className="mf-head-title">{displayName}</div>
          <div className="mf-head-sub" style={{ textTransform: "capitalize" }}>
            {accessLevel} · {workRole}
          </div>
        </div>
      </div>

      <div className="mf-scroll">
        {/* their split — always */}
        <div className="mf-split">
          <div className="mf-split-label">
            <Lock /> Their split · always
          </div>
          {split ? (
            <div className="mf-split-vals">
              <span>
                <span className="mono">{split.master}%</span>
                <small>Master</small>
              </span>
              <span>
                <span className="mono">{split.publishing}%</span>
                <small>Publishing</small>
              </span>
            </div>
          ) : (
            <div className="mf-split-pending">Set when you assign their split</div>
          )}
        </div>

        {isAdmin && (
          <div className="mf-admin">
            <Shield /> Full access — every document &amp; detail on this work.
          </div>
        )}

        {/* information shared */}
        <div className="mf-eyebrow">Information · {infoLines.length}</div>
        {infoLines.map((line) => (
          <div className="mf-line" key={line.key}>
            <Check /> <span className="mf-line-name">{line.label}</span>
            {line.locked && (
              <span className="mf-lockmini">
                <Lock />
              </span>
            )}
          </div>
        ))}

        {/* documents shared, grouped */}
        <div className="mf-eyebrow">
          Documents · {sharedDocs} of {totalDocs}
        </div>
        {sharedDocs === 0 ? (
          <div className="mf-empty">No documents shared yet</div>
        ) : (
          groups.map((g) => {
            const picks = isAdmin ? g.items : g.items.filter((d) => draft.has(d.key));
            if (!picks.length) return null;
            return (
              <div className="mf-group" key={g.key}>
                <div className="mf-group-label">
                  {g.label} <span>{picks.length}</span>
                </div>
                {picks.map((d) => (
                  <ManifestDoc key={d.key} doc={d} />
                ))}
              </div>
            );
          })
        )}
      </div>

      {hidden > 0 && (
        <div className="mf-hidden">
          <Lock />
          <span>
            {hidden} item{hidden > 1 ? "s" : ""} kept private
          </span>
        </div>
      )}
    </div>
  );
}

function ShareSummary({
  accessLevel,
  ownershipBreakdown,
  sharedDocs,
  totalDocs,
}: {
  accessLevel: AccessLevel;
  ownershipBreakdown: boolean;
  sharedDocs: number;
  totalDocs: number;
}) {
  if (accessLevel === "admin") {
    return (
      <div className="perm-summary">
        <Shield />
        <span>
          Sharing <b>everything</b> on this work
        </span>
      </div>
    );
  }
  const detailCount = ownershipBreakdown ? 1 : 0;
  return (
    <div className="perm-summary">
      <Eye />
      <span>
        Split <b>always shown</b>
        {detailCount > 0 && (
          <>
            {" "}
            + <b>{detailCount}</b> detail
          </>
        )}
        {" · "}
        <b>{sharedDocs}</b> of {totalDocs} document{totalDocs === 1 ? "" : "s"}
      </span>
    </div>
  );
}

// ============================================================
// Invite payload type (mirrors InviteCollaboratorModal)
// ============================================================

interface InvitePayload {
  work_id: string;
  email: string;
  name: string;
  role: string;
  stakes: Array<{ stake_type: string; percentage: number }>;
  access_level: AccessLevel;
  initial_grants?: Array<{ resource_type: string; resource_id: string }>;
  ownership_breakdown: boolean;
  terms?: Array<{ label: string; value: string }>;
}

// ============================================================
// Main component
// ============================================================

export default function CollaboratorAccessPanel({
  mode,
  workId,
  collaborator,
  projectName,
  artists,
  open,
  onOpenChange,
}: CollaboratorAccessPanelProps) {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  // ---- data ----
  const workFullQuery = useWorkFull(workId);
  const filesQuery = useWorkFiles(workId);
  const audioQuery = useWorkAudio(workId);
  const grantsQuery = useWorkGrants(mode === "edit" ? workId : undefined);

  // ---- mutations ----
  const addGrants = useAddGrants();
  const removeGrants = useRemoveGrants();
  const setAccessLevelMut = useSetAccessLevel();
  const setWorkRoleMut = useSetWorkRole();
  const revokeMut = useRevokeCollaborator();

  const inviteWithStakes = useMutation({
    mutationFn: async (body: InvitePayload) =>
      apiFetch(`${API_URL}/registry/collaborators/invite-with-stakes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("Invitation sent");
      onOpenChange(false);
    },
    onError: (error: Error) => toast.error(error.message),
  });

  // ---- derive the document groups + composite key list during render ----
  const groups: DocGroup[] = useMemo(() => {
    const files = (filesQuery.data || []).filter((wf) => wf.project_files);
    const contracts: DocRowData[] = [];
    const otherFiles: DocRowData[] = [];
    for (const wf of files) {
      const f = wf.project_files!;
      const isContract = CONTRACT_CATEGORIES.has(f.folder_category);
      const row: DocRowData = {
        id: f.id,
        name: f.file_name,
        meta: [FILE_KIND_LABEL[f.folder_category] || "File", fmtDate(f.created_at)]
          .filter(Boolean)
          .join(" · "),
        kind: isContract ? "contract" : imageish(f.file_type, f.file_name) ? "image" : "file",
        key: `project_file:${f.id}`,
      };
      (isContract ? contracts : otherFiles).push(row);
    }

    const audio: DocRowData[] = (audioQuery.data || [])
      .filter((wa) => wa.audio_files)
      .map((wa) => {
        const a = wa.audio_files!;
        return {
          id: a.id,
          name: a.file_name,
          meta: ["Audio", fmtDate(a.created_at)].filter(Boolean).join(" · "),
          kind: "audio" as const,
          key: `audio_file:${a.id}`,
        };
      });

    const licenses: DocRowData[] = (workFullQuery.data?.licenses || []).map((l) => ({
      id: l.id,
      name: l.licensee_name,
      meta: [l.license_type || "License", [l.territory, l.status].filter(Boolean).join(" · ")]
        .filter(Boolean)
        .join(" · "),
      kind: "license" as const,
      key: `license:${l.id}`,
    }));

    const agreements: DocRowData[] = (workFullQuery.data?.agreements || []).map((a) => ({
      id: a.id,
      name: a.title,
      meta: [a.agreement_type || "Agreement", fmtDate(a.effective_date)]
        .filter(Boolean)
        .join(" · "),
      kind: "agreement" as const,
      key: `agreement:${a.id}`,
    }));

    const all: DocGroup[] = [
      { key: "contract", label: "Contracts", iconNode: <FileText />, items: contracts },
      { key: "audio", label: "Audio files", iconNode: <Music />, items: audio },
      { key: "other", label: "Other files", iconNode: <Folder />, items: otherFiles },
      { key: "license", label: "Licenses", iconNode: <Shield />, items: licenses },
      { key: "agreement", label: "Agreements", iconNode: <FileText />, items: agreements },
    ];
    return all.filter((g) => g.items.length > 0);
  }, [filesQuery.data, audioQuery.data, workFullQuery.data?.licenses, workFullQuery.data?.agreements]);

  const totalDocs = useMemo(
    () => groups.reduce((sum, g) => sum + g.items.length, 0),
    [groups]
  );

  // ---- seed values for edit mode (computed, used for lazy init + diff) ----
  const seedGrantRows = useMemo(() => {
    if (mode !== "edit" || !collaborator) return [];
    return grantsQuery.data?.grants_by_collaborator?.[collaborator.id] || [];
  }, [mode, collaborator, grantsQuery.data?.grants_by_collaborator]);

  // Persisted draft keys (composite) for the seeded collaborator — also the
  // baseline the Save diff compares against.
  const seededKeys = useMemo(() => {
    const s = new Set<string>();
    for (const g of seedGrantRows) {
      if (g.resource_id && g.resource_type !== "ownership_breakdown") {
        s.add(`${g.resource_type}:${g.resource_id}`);
      }
    }
    return s;
  }, [seedGrantRows]);

  const seededOwnership = useMemo(
    () => seedGrantRows.some((g) => g.resource_type === "ownership_breakdown"),
    [seedGrantRows]
  );

  // The collaborator's own split from the work's stakes (edit mode).
  const collaboratorSplit: ManifestSplit | null = useMemo(() => {
    if (mode !== "edit" || !collaborator) return null;
    const stakes = (workFullQuery.data?.stakes || []).filter(
      (s) => (s as { collaborator_id?: string }).collaborator_id === collaborator.id
    );
    if (stakes.length === 0) return null;
    const master = stakes.find((s) => s.stake_type === "master")?.percentage ?? 0;
    const publishing = stakes.find((s) => s.stake_type === "publishing")?.percentage ?? 0;
    return { master, publishing };
  }, [mode, collaborator, workFullQuery.data?.stakes]);

  // ---- editable state ----
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [selectedArtistId, setSelectedArtistId] = useState<string>("");
  const [workRole, setWorkRole] = useState<string>(
    () => (mode === "edit" ? collaborator?.role : "") || "Producer"
  );
  const [accessLevel, setAccessLevel] = useState<AccessLevel>(
    () => (mode === "edit" && collaborator?.access_level === "admin" ? "admin" : "viewer")
  );
  const [ownershipBreakdown, setOwnershipBreakdown] = useState<boolean>(() => false);
  // Lazy init the draft Set from the seeded persisted keys.
  const [draft, setDraft] = useState<Set<string>>(() => new Set<string>());

  // Invite-only split inputs.
  const [masterPct, setMasterPct] = useState("");
  const [publishingPct, setPublishingPct] = useState("");

  // Derive-from-contracts (invite only).
  const [useContracts, setUseContracts] = useState(false);
  const [deriveDialogOpen, setDeriveDialogOpen] = useState(false);
  const [terms, setTerms] = useState<Array<{ label: string; value: string }>>([]);

  // Track whether the editable state has been seeded for the current open/collaborator,
  // so we reseed once when grant data arrives. We compare against the data identity
  // rather than running an effect for derived state — this only touches setters that
  // initialize user-editable fields, which is permitted.
  const [seedSig, setSeedSig] = useState<string>("");
  const currentSig = useMemo(
    () =>
      mode === "edit" && open && collaborator
        ? `${collaborator.id}:${grantsQuery.data ? "loaded" : "pending"}`
        : `${mode}:${open}`,
    [mode, open, collaborator, grantsQuery.data]
  );
  if (currentSig !== seedSig) {
    // One-time (per signature) seed of user-editable state. Calling setState during
    // render with a guard is the React-recommended pattern for "reset state when a
    // prop changes" — cheaper and more correct than a useEffect.
    setSeedSig(currentSig);
    if (mode === "edit" && collaborator) {
      setWorkRole(collaborator.role || "Producer");
      setAccessLevel(collaborator.access_level === "admin" ? "admin" : "viewer");
      setOwnershipBreakdown(seededOwnership);
      setDraft(new Set(seededKeys));
      setEmail(collaborator.email || "");
      setName(collaborator.name || "");
    } else if (mode === "invite" && open) {
      // fresh invite form
      setEmail("");
      setName("");
      setSelectedArtistId("");
      setWorkRole("Producer");
      setAccessLevel("viewer");
      setOwnershipBreakdown(false);
      setDraft(new Set());
      setMasterPct("");
      setPublishingPct("");
      setUseContracts(false);
      setTerms([]);
    }
  }

  const isAdmin = accessLevel === "admin";

  // ---- draft mutators (functional setState) ----
  const toggleDoc = (key: string) =>
    setDraft((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const setAllDocs = (keys: string[], on: boolean) =>
    setDraft((prev) => {
      const next = new Set(prev);
      for (const k of keys) {
        if (on) next.add(k);
        else next.delete(k);
      }
      return next;
    });

  // ---- manifest split (derived during render) ----
  const manifestSplit: ManifestSplit | null = useMemo(() => {
    if (mode === "invite") {
      const m = parseFloat(masterPct);
      const p = parseFloat(publishingPct);
      if ((Number.isNaN(m) || m === 0) && (Number.isNaN(p) || p === 0)) return null;
      return { master: Number.isNaN(m) ? 0 : m, publishing: Number.isNaN(p) ? 0 : p };
    }
    return collaboratorSplit;
  }, [mode, masterPct, publishingPct, collaboratorSplit]);

  const displayName = (mode === "edit" ? collaborator?.name : name) || "New collaborator";

  // ---- roster prefill (invite) ----
  const handleArtistSelect = (artistId: string) => {
    setSelectedArtistId(artistId);
    const artist = artists?.find((a) => a.id === artistId);
    if (artist) {
      setEmail(artist.email);
      setName(artist.name);
    }
  };

  // ---- derive-from-contracts apply (invite) ----
  const handleDeriveApply = ({
    masterPct: mPct,
    publishingPct: pPct,
    terms: derivedTerms,
    matchedFileIds,
  }: {
    masterPct: number;
    publishingPct: number;
    terms: Array<{ label: string; value: string }>;
    matchedFileIds: string[];
  }) => {
    setMasterPct(mPct > 0 ? String(mPct) : "");
    setPublishingPct(pPct > 0 ? String(pPct) : "");
    setTerms(derivedTerms);
    if (matchedFileIds.length > 0) {
      setDraft((prev) => {
        const next = new Set(prev);
        for (const id of matchedFileIds) next.add(`project_file:${id}`);
        return next;
      });
    }
  };

  // ---- save (edit) ----
  const handleSave = async () => {
    if (!collaborator) return;
    const collaboratorId = collaborator.id;
    try {
      const tasks: Promise<unknown>[] = [];

      // access level
      if (
        accessLevel !== (collaborator.access_level === "admin" ? "admin" : "viewer")
      ) {
        tasks.push(
          setAccessLevelMut.mutateAsync({ collaboratorId, accessLevel })
        );
      }
      // work role
      if (workRole !== (collaborator.role || "Producer")) {
        tasks.push(setWorkRoleMut.mutateAsync({ collaboratorId, role: workRole }));
      }
      // ownership breakdown
      if (ownershipBreakdown !== seededOwnership) {
        tasks.push(
          addGrants.mutateAsync({ collaboratorId, ownershipBreakdown })
        );
      }
      // doc keys: added / removed
      const added: GrantItem[] = [];
      const removed: GrantItem[] = [];
      for (const key of draft) {
        if (!seededKeys.has(key)) {
          const [rt, id] = splitKey(key);
          if (rt && id) added.push({ resource_type: rt, resource_id: id });
        }
      }
      for (const key of seededKeys) {
        if (!draft.has(key)) {
          const [rt, id] = splitKey(key);
          if (rt && id) removed.push({ resource_type: rt, resource_id: id });
        }
      }
      if (added.length) tasks.push(addGrants.mutateAsync({ collaboratorId, grants: added }));
      if (removed.length)
        tasks.push(removeGrants.mutateAsync({ collaboratorId, grants: removed }));

      await Promise.all(tasks);
      toast.success("Access updated");
      onOpenChange(false);
    } catch {
      // Individual mutations toast their own errors.
    }
  };

  // ---- revoke (edit) ----
  const handleRevoke = async () => {
    if (!collaborator) return;
    await revokeMut.mutateAsync(collaborator.id);
    onOpenChange(false);
  };

  // ---- send (invite) ----
  const resolvedName = name.trim();
  const canSend = !!email.trim() && !!resolvedName && !!workRole;

  const handleSend = async () => {
    if (!canSend) return;
    const stakes: Array<{ stake_type: string; percentage: number }> = [];
    const m = parseFloat(masterPct);
    if (!Number.isNaN(m) && m > 0 && m <= 100) stakes.push({ stake_type: "master", percentage: m });
    const p = parseFloat(publishingPct);
    if (!Number.isNaN(p) && p > 0 && p <= 100)
      stakes.push({ stake_type: "publishing", percentage: p });

    const payload: InvitePayload = {
      work_id: workId,
      email: email.trim(),
      name: resolvedName,
      role: workRole,
      stakes,
      access_level: accessLevel,
      ownership_breakdown: isAdmin ? false : ownershipBreakdown,
      terms,
    };
    if (!isAdmin) {
      payload.initial_grants = Array.from(draft)
        .map((key) => {
          const [rt, id] = splitKey(key);
          return rt && id ? { resource_type: rt, resource_id: id } : null;
        })
        .filter(Boolean) as Array<{ resource_type: string; resource_id: string }>;
    }
    await inviteWithStakes.mutateAsync(payload);
  };

  const sending = inviteWithStakes.isPending;
  const saving =
    setAccessLevelMut.isPending ||
    setWorkRoleMut.isPending ||
    addGrants.isPending ||
    removeGrants.isPending;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="perm-modal p-0 gap-0 max-w-[880px] overflow-hidden [&>button]:hidden">
        {/* Header */}
        <div className="perm-head">
          <div className="perm-head-top">
            <h3 className="perm-title">
              <span className="perm-mark">{mode === "edit" ? <Users /> : <Plus />}</span>
              {mode === "edit" ? "Edit access" : "Invite collaborator"}
            </h3>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() => onOpenChange(false)}
              aria-label="Close"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <p className="perm-sub">
            {mode === "edit"
              ? "Manage what this person can see on "
              : "Choose what to share on "}
            <b>{workFullQuery.data?.title || "this work"}</b>
            {projectName ? <> · {projectName}</> : null}
          </p>
        </div>

        {/* Two-pane body */}
        <div className="perm-two">
          {/* Left controls */}
          <div className="perm-two-controls perm-scope">
            {/* Identity */}
            {mode === "edit" && collaborator ? (
              <div className="perm-chip">
                <RegistryAvatar name={collaborator.name || collaborator.email} size={40} />
                <div className="perm-chip-body">
                  <div className="perm-chip-name">
                    {collaborator.name || collaborator.email}
                    <span
                      className="inline-flex items-center gap-1 rounded border border-emerald-500/40 bg-emerald-500/5 px-2 py-0.5 text-[10px] font-medium capitalize text-emerald-600"
                    >
                      <Check className="h-3 w-3" /> {collaborator.status}
                    </span>
                  </div>
                  <div className="perm-chip-mail">{collaborator.email}</div>
                </div>
              </div>
            ) : (
              <>
                {artists && artists.length > 0 && (
                  <div>
                    <Label className="text-sm font-medium">Select from roster</Label>
                    <Select value={selectedArtistId} onValueChange={handleArtistSelect}>
                      <SelectTrigger className="mt-1">
                        <SelectValue placeholder="Choose an artist" />
                      </SelectTrigger>
                      <SelectContent>
                        {artists.map((a) => (
                          <SelectItem key={a.id} value={a.id}>
                            {a.name} ({a.email})
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label className="text-sm font-medium">
                      Email <span className="text-destructive">*</span>
                    </Label>
                    <Input
                      type="email"
                      value={email}
                      placeholder="name@example.com"
                      onChange={(e) => setEmail(e.target.value)}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label className="text-sm font-medium">
                      Name <span className="text-destructive">*</span>
                    </Label>
                    <Input
                      value={name}
                      placeholder="Display name"
                      onChange={(e) => setName(e.target.value)}
                      className="mt-1"
                    />
                  </div>
                </div>
              </>
            )}

            {/* Access fields */}
            <div>
              <div className="mb-3.5">
                <Label className="text-sm font-medium">Role on this work</Label>
                <Select value={workRole} onValueChange={setWorkRole}>
                  <SelectTrigger className="mt-1">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {WORK_ROLES.map((r) => (
                      <SelectItem key={r} value={r}>
                        {r}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label className="mb-1 block text-sm font-medium">Access level</Label>
                <SegmentedAccess value={accessLevel} onChange={setAccessLevel} />
              </div>
              <AccessDescriptor level={accessLevel} />
            </div>

            <hr className="hr" />

            {isAdmin ? (
              <AdminCallout />
            ) : (
              <>
                {/* Information */}
                <div>
                  <div className="perm-group-label">
                    Information
                    <span className="perm-count">· what they can read</span>
                  </div>
                  <div>
                    <InfoRow
                      icon={<TrendingUp />}
                      title="Their own split"
                      desc="The master & publishing % they personally hold"
                      on
                      locked
                    />
                    <InfoRow
                      icon={<Users />}
                      title="Full ownership breakdown"
                      desc="Everyone's master & publishing split on this work"
                      on={ownershipBreakdown}
                      onToggle={() => setOwnershipBreakdown((v) => !v)}
                    />
                  </div>
                </div>

                {/* Document sections */}
                <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
                  {groups.map((g) => (
                    <DocSection
                      key={g.key}
                      group={g}
                      draft={draft}
                      onToggle={toggleDoc}
                      onSetAll={setAllDocs}
                    />
                  ))}
                </div>

                {/* Invite-only: their split + derive */}
                {mode === "invite" && (
                  <div>
                    <div className="perm-group-label">
                      Their split
                      <span className="perm-count">· master &amp; publishing %</span>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <Label className="text-xs font-medium text-muted-foreground">
                          Master %
                        </Label>
                        <Input
                          type="number"
                          min="0"
                          max="100"
                          step="0.01"
                          value={masterPct}
                          onChange={(e) => setMasterPct(e.target.value)}
                          placeholder="e.g. 15"
                          className="mt-1"
                        />
                      </div>
                      <div>
                        <Label className="text-xs font-medium text-muted-foreground">
                          Publishing %
                        </Label>
                        <Input
                          type="number"
                          min="0"
                          max="100"
                          step="0.01"
                          value={publishingPct}
                          onChange={(e) => setPublishingPct(e.target.value)}
                          placeholder="e.g. 10"
                          className="mt-1"
                        />
                      </div>
                    </div>

                    <div className="mt-3 space-y-2 rounded-lg border border-border bg-card px-3 py-2.5">
                      <label className="flex cursor-pointer items-start gap-2.5">
                        <Checkbox
                          checked={useContracts}
                          onCheckedChange={(c) => setUseContracts(!!c)}
                          className="mt-0.5"
                        />
                        <span className="flex flex-col">
                          <span className="text-sm font-medium text-foreground">
                            Use my contracts to fill in the details
                          </span>
                          <span className="text-xs text-muted-foreground">
                            Scan this work's documents for {resolvedName || "this person"}'s split.
                          </span>
                        </span>
                      </label>
                      {useContracts && (
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          disabled={!resolvedName}
                          onClick={() => setDeriveDialogOpen(true)}
                          className="w-full"
                        >
                          <Sparkles className="mr-2 h-4 w-4" />
                          Derive from contracts
                        </Button>
                      )}
                      {useContracts && !resolvedName && (
                        <p className="text-xs text-muted-foreground">Enter a name first.</p>
                      )}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Right manifest */}
          <Manifest
            displayName={displayName}
            accessLevel={accessLevel}
            workRole={workRole}
            split={manifestSplit}
            ownershipBreakdown={ownershipBreakdown}
            groups={groups}
            draft={draft}
            totalDocs={totalDocs}
          />
        </div>

        {/* Footer */}
        <div className="perm-foot">
          <ShareSummary
            accessLevel={accessLevel}
            ownershipBreakdown={ownershipBreakdown}
            sharedDocs={isAdmin ? totalDocs : draft.size}
            totalDocs={totalDocs}
          />
          <div className="perm-foot-actions">
            {mode === "edit" && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={handleRevoke}
                disabled={revokeMut.isPending}
                className="text-destructive hover:bg-destructive/10 hover:text-destructive"
              >
                <Trash2 className="mr-1.5 h-4 w-4" />
                Revoke
              </Button>
            )}
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </Button>
            {mode === "edit" ? (
              <Button type="button" size="sm" onClick={handleSave} disabled={saving}>
                <Check className="mr-1.5 h-4 w-4" />
                Save changes
              </Button>
            ) : (
              <Button type="button" size="sm" onClick={handleSend} disabled={sending || !canSend}>
                <Users className="mr-1.5 h-4 w-4" />
                Send invitation
              </Button>
            )}
          </div>
        </div>
      </DialogContent>

      {mode === "invite" && (
        <DeriveCollaboratorSplitDialog
          workId={workId}
          collaboratorName={name}
          open={deriveDialogOpen}
          onOpenChange={setDeriveDialogOpen}
          onApply={handleDeriveApply}
        />
      )}
    </Dialog>
  );
}

// Split a composite grant key `${resource_type}:${resource_id}` back into parts.
function splitKey(key: string): [ResourceType | null, string | null] {
  const idx = key.indexOf(":");
  if (idx === -1) return [null, null];
  const rt = key.slice(0, idx) as ResourceType;
  const id = key.slice(idx + 1);
  return [rt, id || null];
}
