import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { type OwnershipStake } from "@/hooks/useRegistry";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Loader2, Send } from "lucide-react";
import { toast } from "sonner";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

const ROLES = ["Artist", "Producer", "Songwriter", "Composer", "Publisher", "Label", "Other"];
const STAKE_TYPES = ["none", "master", "publishing", "both"] as const;
type StakeType = (typeof STAKE_TYPES)[number];

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

  const inviteWithStakes = useMutation({
    mutationFn: async (body: any) => {
      const res = await fetch(`${API_URL}/registry/collaborators/invite-with-stakes?user_id=${user!.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to invite");
      }
      return res.json();
    },
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

    await inviteWithStakes.mutateAsync({
      work_id: workId,
      email: email.trim(),
      name: name.trim(),
      role: resolvedRole,
      stakes: stakesArr,
      notes: notes.trim() || undefined,
    });
    resetForm();
  };

  const resolvedRole = role === "Other" ? customRole.trim() : role;
  const canSubmit = !!email.trim() && !!name.trim() && !!resolvedRole;

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
          {(stakeType === "master" || stakeType === "both") && (
            <div>
              <Label className="text-sm font-medium">Master Ownership %</Label>
              <Input type="number" min="0" max="100" step="0.01" value={masterPct}
                onChange={(e) => setMasterPct(e.target.value)} placeholder="e.g. 15" />
            </div>
          )}

          {/* Publishing percentage */}
          {(stakeType === "publishing" || stakeType === "both") && (
            <div>
              <Label className="text-sm font-medium">Publishing Ownership %</Label>
              <Input type="number" min="0" max="100" step="0.01" value={publishingPct}
                onChange={(e) => setPublishingPct(e.target.value)} placeholder="e.g. 10" />
            </div>
          )}

          {/* Notes */}
          <div>
            <Label className="text-sm font-medium">Notes / Terms (optional)</Label>
            <Textarea value={notes} onChange={(e) => setNotes(e.target.value)}
              placeholder="Any additional terms or notes for this collaborator..." rows={3} />
          </div>

          <Button onClick={handleSubmit} disabled={inviteWithStakes.isPending || !canSubmit}
            className="w-full">
            {inviteWithStakes.isPending ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Send className="w-4 h-4 mr-2" />}
            Send Invitation
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
