import { useState } from "react";
import { useInviteCollaborator, type OwnershipStake } from "@/hooks/useRegistry";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Loader2, Send } from "lucide-react";

const ROLES = ["Artist", "Producer", "Songwriter", "Composer", "Publisher", "Label", "Other"];

interface Props {
  workId: string;
  stakes: OwnershipStake[];
  artists?: Array<{ id: string; name: string; email: string }>;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export default function InviteCollaboratorModal({ workId, stakes, artists, open, onOpenChange }: Props) {
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [role, setRole] = useState("");
  const [stakeId, setStakeId] = useState<string>("");
  const [selectedArtistId, setSelectedArtistId] = useState<string>("");

  const invite = useInviteCollaborator();

  const resetForm = () => {
    setEmail(""); setName(""); setRole(""); setStakeId(""); setSelectedArtistId("");
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
    if (!email.trim() || !name.trim() || !role) return;
    await invite.mutateAsync({
      work_id: workId,
      email: email.trim(),
      name: name.trim(),
      role,
      stake_id: stakeId || undefined,
    });
    resetForm();
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) resetForm(); onOpenChange(o); }}>
      <DialogContent>
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
          <div>
            <Label className="text-sm font-medium">Email *</Label>
            <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
              placeholder="collaborator@example.com" />
          </div>
          <div>
            <Label className="text-sm font-medium">Name *</Label>
            <Input value={name} onChange={(e) => setName(e.target.value)}
              placeholder="Full name" />
          </div>
          <div>
            <Label className="text-sm font-medium">Role *</Label>
            <Select value={role} onValueChange={setRole}>
              <SelectTrigger><SelectValue placeholder="Select role" /></SelectTrigger>
              <SelectContent>
                {ROLES.map((r) => <SelectItem key={r} value={r}>{r}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          {stakes.length > 0 && (
            <div>
              <Label className="text-sm font-medium">Link to Stake (optional)</Label>
              <Select value={stakeId} onValueChange={setStakeId}>
                <SelectTrigger><SelectValue placeholder="Select a stake to link" /></SelectTrigger>
                <SelectContent>
                  {stakes.map((s) => (
                    <SelectItem key={s.id} value={s.id}>
                      {s.holder_name} — {s.stake_type} {s.percentage}%
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          <Button onClick={handleSubmit} disabled={invite.isPending || !email.trim() || !name.trim() || !role}
            className="w-full">
            {invite.isPending ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Send className="w-4 h-4 mr-2" />}
            Send Invitation
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
