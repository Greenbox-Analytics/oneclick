import { useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { type OwnershipStake, useWorkFull } from "@/hooks/useRegistry";
import { useWorkFiles } from "@/hooks/useWorkFiles";
import { useWorkAudio } from "@/hooks/useWorkAudio";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Textarea } from "@/components/ui/textarea";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Loader2, Send, Shield, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import {
  ResourceGrantPicker,
  type ResourceItem,
  type GrantSelection,
} from "./ResourceGrantPicker";
import DeriveCollaboratorSplitDialog from "./DeriveCollaboratorSplitDialog";

const ROLES = ["Artist", "Producer", "Songwriter", "Composer", "Publisher", "Label", "Other"];
const STAKE_TYPES = ["none", "master", "publishing", "both"] as const;
type StakeType = (typeof STAKE_TYPES)[number];

const FOLDER_BADGE: Record<string, string> = {
  contract: "CONTRACT",
  split_sheet: "SPLIT SHEET",
  royalty_statement: "STATEMENT",
};

const EMPTY_SELECTION: GrantSelection = {
  project_file: [],
  audio_file: [],
  license: [],
  agreement: [],
  ownership_breakdown: false,
};

interface Props {
  workId: string;
  stakes: OwnershipStake[];
  artists?: Array<{ id: string; name: string; email: string }>;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export default function InviteCollaboratorModal({ workId, stakes, artists, open, onOpenChange }: Props) {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [role, setRole] = useState("");
  const [customRole, setCustomRole] = useState("");
  const [selectedArtistId, setSelectedArtistId] = useState<string>("");
  const [stakeType, setStakeType] = useState<StakeType>("none");
  const [masterPct, setMasterPct] = useState("");
  const [publishingPct, setPublishingPct] = useState("");
  const [notes, setNotes] = useState("");
  const [isAdmin, setIsAdmin] = useState(false);
  const [selection, setSelection] = useState<GrantSelection>(EMPTY_SELECTION);
  const [useContracts, setUseContracts] = useState(false);
  const [deriveDialogOpen, setDeriveDialogOpen] = useState(false);
  const [terms, setTerms] = useState<Array<{ label: string; value: string }>>([]);

  // Resources linked to this work, for the "Share now" picker. Mapping mirrors
  // PermissionsPanel so the two surfaces stay consistent.
  const filesQuery = useWorkFiles(workId);
  const audioQuery = useWorkAudio(workId);
  const workFullQuery = useWorkFull(workId);

  const documents: ResourceItem[] = useMemo(
    () =>
      (filesQuery.data || [])
        .filter((wf) => wf.project_files)
        .map((wf) => {
          const f = wf.project_files!;
          return {
            id: f.id,
            label: f.file_name,
            badge: FOLDER_BADGE[f.folder_category] || "FILE",
          };
        }),
    [filesQuery.data]
  );

  const audio: ResourceItem[] = useMemo(
    () =>
      (audioQuery.data || [])
        .filter((wa) => wa.audio_files)
        .map((wa) => ({
          id: wa.audio_files!.id,
          label: wa.audio_files!.file_name,
          badge: "AUDIO",
        })),
    [audioQuery.data]
  );

  const licenses: ResourceItem[] = useMemo(
    () =>
      (workFullQuery.data?.licenses || []).map((l) => ({
        id: l.id,
        label: l.licensee_name,
        meta: [l.territory, l.status].filter(Boolean).join(" · "),
        badge: "LICENSE",
      })),
    [workFullQuery.data?.licenses]
  );

  const agreements: ResourceItem[] = useMemo(
    () =>
      (workFullQuery.data?.agreements || []).map((a) => ({
        id: a.id,
        label: a.title,
        badge: "AGREEMENT",
      })),
    [workFullQuery.data?.agreements]
  );

  interface InvitePayload {
    work_id: string;
    email: string;
    name: string;
    role: string;
    stakes: Array<{ stake_type: string; percentage: number }>;
    notes?: string;
    access_level: "viewer" | "admin";
    initial_grants?: Array<{ resource_type: string; resource_id: string }>;
    ownership_breakdown: boolean;
    terms?: Array<{ label: string; value: string }>;
  }

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

  const resetForm = () => {
    setEmail(""); setName(""); setRole(""); setCustomRole("");
    setSelectedArtistId(""); setStakeType("none");
    setMasterPct(""); setPublishingPct(""); setNotes("");
    setIsAdmin(false); setSelection(EMPTY_SELECTION);
    setUseContracts(false); setDeriveDialogOpen(false); setTerms([]);
  };

  // Apply the result of contract-derivation back onto the invite form: set the
  // stake type/percentages, merge matched files into the share-now grants, and
  // store the derived terms to carry on submit.
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
    const hasMaster = mPct > 0;
    const hasPublishing = pPct > 0;
    if (hasMaster && hasPublishing) setStakeType("both");
    else if (hasMaster) setStakeType("master");
    else if (hasPublishing) setStakeType("publishing");
    setMasterPct(hasMaster ? String(mPct) : "");
    setPublishingPct(hasPublishing ? String(pPct) : "");
    setTerms(derivedTerms);
    if (matchedFileIds.length > 0) {
      setSelection((prev) => ({
        ...prev,
        project_file: Array.from(new Set([...prev.project_file, ...matchedFileIds])),
      }));
    }
  };

  const handleArtistSelect = (artistId: string) => {
    setSelectedArtistId(artistId);
    if (artists) {
      const artist = artists.find((a) => a.id === artistId);
      if (artist) {
        setEmail(artist.email);
        setName(artist.name);
      }
    }
  };

  const handleSubmit = async () => {
    const resolvedRole = role === "Other" ? customRole.trim() : role;
    if (!email.trim() || !name.trim() || !resolvedRole) return;

    // Build stakes array
    const stakesArr: Array<{ stake_type: string; percentage: number }> = [];
    if (stakeType === "master" || stakeType === "both") {
      const pct = parseFloat(masterPct);
      if (!isNaN(pct) && pct > 0 && pct <= 100) {
        stakesArr.push({ stake_type: "master", percentage: pct });
      }
    }
    if (stakeType === "publishing" || stakeType === "both") {
      const pct = parseFloat(publishingPct);
      if (!isNaN(pct) && pct > 0 && pct <= 100) {
        stakesArr.push({ stake_type: "publishing", percentage: pct });
      }
    }

    // Admins see & edit everything, so we don't send per-resource grants.
    const payload: InvitePayload = {
      work_id: workId,
      email: email.trim(),
      name: name.trim(),
      role: resolvedRole,
      stakes: stakesArr,
      notes: notes.trim() || undefined,
      access_level: isAdmin ? "admin" : "viewer",
      ownership_breakdown: isAdmin ? false : selection.ownership_breakdown,
      // Terms describe the person's deal regardless of access level, so viewer
      // and admin invites both carry them.
      terms,
    };
    if (!isAdmin) {
      const grantKeys: Array<keyof Pick<GrantSelection, "project_file" | "audio_file" | "license" | "agreement">> = [
        "project_file", "audio_file", "license", "agreement",
      ];
      payload.initial_grants = grantKeys.flatMap((rt) =>
        selection[rt].map((id) => ({ resource_type: rt, resource_id: id }))
      );
    }

    await inviteWithStakes.mutateAsync(payload);
    resetForm();
  };

  const resolvedRole = role === "Other" ? customRole.trim() : role;

  // Running master/publishing totals. Existing stakes come from the work's
  // current cap table; the "new" portion is whatever the user has typed for
  // this invite. We surface both totals inline and block submit on overflow so
  // the user fixes it locally instead of hitting the DB trigger.
  const allocatedMaster = useMemo(
    () => stakes.filter((s) => s.stake_type === "master").reduce((sum, s) => sum + Number(s.percentage || 0), 0),
    [stakes]
  );
  const allocatedPublishing = useMemo(
    () => stakes.filter((s) => s.stake_type === "publishing").reduce((sum, s) => sum + Number(s.percentage || 0), 0),
    [stakes]
  );
  const showMasterInput = stakeType === "master" || stakeType === "both";
  const showPublishingInput = stakeType === "publishing" || stakeType === "both";
  const newMaster = showMasterInput ? parseFloat(masterPct) || 0 : 0;
  const newPublishing = showPublishingInput ? parseFloat(publishingPct) || 0 : 0;
  const totalMaster = allocatedMaster + newMaster;
  const totalPublishing = allocatedPublishing + newPublishing;
  const masterOverflow = showMasterInput && totalMaster > 100;
  const publishingOverflow = showPublishingInput && totalPublishing > 100;

  const canSubmit =
    !!email.trim() &&
    !!name.trim() &&
    !!resolvedRole &&
    !masterOverflow &&
    !publishingOverflow;

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) resetForm(); onOpenChange(o); }}>
      <DialogContent className="max-h-[90vh] overflow-y-auto">
        <DialogHeader><DialogTitle>Invite Collaborator</DialogTitle></DialogHeader>
        <div className="space-y-3 pt-2">
          {artists && artists.length > 0 && (
            <>
              <div>
                <Label className="text-sm font-medium">Select from roster</Label>
                <Select value={selectedArtistId} onValueChange={handleArtistSelect}>
                  <SelectTrigger><SelectValue placeholder="Choose an artist" /></SelectTrigger>
                  <SelectContent>
                    {artists.map((a) => (
                      <SelectItem key={a.id} value={a.id}>
                        {a.name} ({a.email})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="text-xs text-muted-foreground text-center">Or enter details manually</div>
            </>
          )}

          {/* Email */}
          <div>
            <Label className="text-sm font-medium">Email *</Label>
            <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
              placeholder="collaborator@example.com" />
          </div>

          {/* Name */}
          <div>
            <Label className="text-sm font-medium">Name *</Label>
            <Input value={name} onChange={(e) => setName(e.target.value)}
              placeholder="Full name" />
          </div>

          {/* Role */}
          <div>
            <Label className="text-sm font-medium">Role *</Label>
            <Select value={role} onValueChange={setRole}>
              <SelectTrigger><SelectValue placeholder="Select role" /></SelectTrigger>
              <SelectContent>
                {ROLES.map((r) => <SelectItem key={r} value={r}>{r}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          {role === "Other" && (
            <div>
              <Label className="text-sm font-medium">Custom Role *</Label>
              <Input value={customRole} onChange={(e) => setCustomRole(e.target.value)}
                placeholder="Enter custom role" />
            </div>
          )}

          {/* Use my contracts to fill in the details */}
          <div className="space-y-2 rounded-lg border border-border bg-card px-3 py-2.5">
            <label className="flex items-start gap-2.5 cursor-pointer">
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
                  Scan this work's documents for {name.trim() || "this person"}'s split.
                </span>
              </span>
            </label>
            {useContracts && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={!name.trim()}
                onClick={() => setDeriveDialogOpen(true)}
                className="w-full"
              >
                <Sparkles className="mr-2 h-4 w-4" />
                Derive from contracts
              </Button>
            )}
            {useContracts && !name.trim() && (
              <p className="text-xs text-muted-foreground">Enter a name first.</p>
            )}
          </div>

          {/* Stake Type */}
          <div>
            <Label className="text-sm font-medium">Stake Type</Label>
            <Select value={stakeType} onValueChange={(v) => setStakeType(v as StakeType)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                <SelectItem value="master">Master</SelectItem>
                <SelectItem value="publishing">Publishing</SelectItem>
                <SelectItem value="both">Both</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Master percentage */}
          {showMasterInput && (
            <div>
              <Label className="text-sm font-medium">Master Ownership %</Label>
              <Input type="number" min="0" max="100" step="0.01" value={masterPct}
                onChange={(e) => setMasterPct(e.target.value)} placeholder="e.g. 15" />
              <p className={`mt-1 text-xs ${masterOverflow ? "text-destructive" : "text-muted-foreground"}`}>
                Already allocated: {allocatedMaster.toFixed(2)}% · This person: {newMaster.toFixed(2)}% ·{" "}
                <span className="font-medium">Total: {totalMaster.toFixed(2)}% / 100%</span>
                {masterOverflow && ` — exceeds 100% by ${(totalMaster - 100).toFixed(2)}%.`}
              </p>
            </div>
          )}

          {/* Publishing percentage */}
          {showPublishingInput && (
            <div>
              <Label className="text-sm font-medium">Publishing Ownership %</Label>
              <Input type="number" min="0" max="100" step="0.01" value={publishingPct}
                onChange={(e) => setPublishingPct(e.target.value)} placeholder="e.g. 10" />
              <p className={`mt-1 text-xs ${publishingOverflow ? "text-destructive" : "text-muted-foreground"}`}>
                Already allocated: {allocatedPublishing.toFixed(2)}% · This person: {newPublishing.toFixed(2)}% ·{" "}
                <span className="font-medium">Total: {totalPublishing.toFixed(2)}% / 100%</span>
                {publishingOverflow && ` — exceeds 100% by ${(totalPublishing - 100).toFixed(2)}%.`}
              </p>
            </div>
          )}

          {(masterOverflow || publishingOverflow) && (
            <div className="rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs text-destructive">
              Royalty splits can't exceed 100%. Reduce this person's share before sending the invitation.
            </div>
          )}

          {/* Notes */}
          <div>
            <Label className="text-sm font-medium">Notes / Terms (optional)</Label>
            <Textarea value={notes} onChange={(e) => setNotes(e.target.value)}
              placeholder="Any additional terms or notes for this collaborator..." rows={3} />
          </div>

          {/* Admin access level */}
          <div className="flex items-center justify-between gap-3 rounded-lg border border-border bg-card px-3 py-2.5">
            <div className="flex items-start gap-2.5">
              <Shield className="w-4 h-4 text-primary mt-0.5 shrink-0" />
              <div className="flex flex-col">
                <span className="text-sm font-medium text-foreground">Admin</span>
                <span className="text-xs text-muted-foreground">
                  Can see &amp; edit everything and manage others
                </span>
              </div>
            </div>
            <Switch
              checked={isAdmin}
              onCheckedChange={setIsAdmin}
              className="data-[state=checked]:bg-primary"
            />
          </div>

          {/* Share now */}
          <div className="space-y-2">
            <Label className="text-sm font-medium">Share now</Label>
            {isAdmin ? (
              <div className="rounded-lg border border-dashed px-4 py-6 text-center text-sm text-muted-foreground">
                Admins can see &amp; edit everything.
              </div>
            ) : (
              <ResourceGrantPicker
                documents={documents}
                audio={audio}
                licenses={licenses}
                agreements={agreements}
                value={selection}
                onChange={setSelection}
              />
            )}
          </div>

          <Button onClick={handleSubmit} disabled={inviteWithStakes.isPending || !canSubmit}
            className="w-full">
            {inviteWithStakes.isPending ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Send className="w-4 h-4 mr-2" />}
            Send Invitation
          </Button>
        </div>
      </DialogContent>

      <DeriveCollaboratorSplitDialog
        workId={workId}
        projectId={workFullQuery.data?.project_id || undefined}
        collaboratorName={name}
        open={deriveDialogOpen}
        onOpenChange={setDeriveDialogOpen}
        onApply={handleDeriveApply}
      />
    </Dialog>
  );
}
