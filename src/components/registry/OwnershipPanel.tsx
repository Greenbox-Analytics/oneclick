import { useState } from "react";
import {
  useCreateStake, useUpdateStake, useDeleteStake,
  type OwnershipStake, type Collaborator,
} from "@/hooks/useRegistry";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import { Plus, Pencil, Trash2, Loader2 } from "lucide-react";

const ROLES = ["Artist", "Producer", "Songwriter", "Composer", "Publisher", "Label", "Other"];

const APPROVAL_COLORS: Record<string, string> = {
  confirmed: "bg-green-100 text-green-800",
  disputed: "bg-red-100 text-red-800",
  invited: "bg-amber-100 text-amber-800",
};

interface Props {
  workId: string;
  stakes: OwnershipStake[];
  collaborators: Collaborator[];
  isOwner: boolean;
}

function StakeSection({
  label, stakeType, stakes, workId, collaborators, isOwner,
}: {
  label: string; stakeType: string; stakes: OwnershipStake[];
  workId: string; collaborators: Collaborator[]; isOwner: boolean;
}) {
  const [showAdd, setShowAdd] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [role, setRole] = useState("");
  const [pct, setPct] = useState("");
  const [email, setEmail] = useState("");
  const [ipi, setIpi] = useState("");
  const [pubLabel, setPubLabel] = useState("");

  const createStake = useCreateStake();
  const updateStake = useUpdateStake();
  const deleteStake = useDeleteStake();

  const totalPct = stakes.reduce((s, st) => s + st.percentage, 0);
  const barWidth = Math.min(totalPct, 100);

  const resetForm = () => {
    setName(""); setRole(""); setPct(""); setEmail(""); setIpi(""); setPubLabel("");
    setShowAdd(false); setEditingId(null);
  };

  const handleSubmit = async () => {
    const percentage = parseFloat(pct);
    if (!name || !role || isNaN(percentage) || percentage <= 0) return;
    if (editingId) {
      await updateStake.mutateAsync({
        stakeId: editingId, holder_name: name, holder_role: role, percentage,
        holder_email: email || undefined, holder_ipi: ipi || undefined,
        publisher_or_label: pubLabel || undefined,
      });
    } else {
      await createStake.mutateAsync({
        work_id: workId, stake_type: stakeType, holder_name: name,
        holder_role: role, percentage,
        holder_email: email || undefined, holder_ipi: ipi || undefined,
        publisher_or_label: pubLabel || undefined,
      });
    }
    resetForm();
  };

  const startEdit = (stake: OwnershipStake) => {
    setEditingId(stake.id); setName(stake.holder_name); setRole(stake.holder_role);
    setPct(String(stake.percentage)); setEmail(stake.holder_email || "");
    setIpi(stake.holder_ipi || ""); setPubLabel(stake.publisher_or_label || "");
    setShowAdd(true);
  };

  const getApprovalStatus = (stake: OwnershipStake): string | null => {
    const byStake = collaborators.find((c) => c.stake_id === stake.id);
    if (byStake) return byStake.status;
    const byEmail = stake.holder_email
      ? collaborators.find((c) => c.email.toLowerCase() === stake.holder_email!.toLowerCase())
      : null;
    return byEmail?.status || null;
  };

  const isPending = createStake.isPending || updateStake.isPending;

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{label} Ownership</CardTitle>
          {isOwner && (
            <Dialog open={showAdd} onOpenChange={(open) => { if (!open) resetForm(); setShowAdd(open); }}>
              <DialogTrigger asChild>
                <Button size="sm" variant="outline"><Plus className="w-3 h-3 mr-1" /> Add</Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>{editingId ? "Edit" : "Add"} {label} Stake</DialogTitle>
                </DialogHeader>
                <div className="space-y-3 pt-2">
                  <div>
                    <label className="text-sm font-medium">Holder Name *</label>
                    <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Role *</label>
                    <Select value={role} onValueChange={setRole}>
                      <SelectTrigger><SelectValue placeholder="Select role" /></SelectTrigger>
                      <SelectContent>
                        {ROLES.map((r) => <SelectItem key={r} value={r}>{r}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Percentage *</label>
                    <Input type="number" min="0.01" max="100" step="0.01" value={pct}
                      onChange={(e) => setPct(e.target.value)} placeholder="e.g. 50.00" />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-sm font-medium">Email</label>
                      <Input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" />
                    </div>
                    <div>
                      <label className="text-sm font-medium">IPI Number</label>
                      <Input value={ipi} onChange={(e) => setIpi(e.target.value)} placeholder="IPI/CAE" />
                    </div>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Publisher / Label</label>
                    <Input value={pubLabel} onChange={(e) => setPubLabel(e.target.value)}
                      placeholder="Publisher or label name" />
                  </div>
                  <Button onClick={handleSubmit} disabled={isPending} className="w-full">
                    {isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    {editingId ? "Update" : "Add"} Stake
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          )}
        </div>
        <div className="mt-2">
          <div className="flex justify-between text-xs text-muted-foreground mb-1">
            <span>{totalPct.toFixed(2)}% allocated</span>
            <span>{(100 - totalPct).toFixed(2)}% unallocated</span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div className={`h-full rounded-full transition-all ${totalPct > 100 ? "bg-red-500" : "bg-primary"}`}
              style={{ width: `${barWidth}%` }} />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {stakes.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-4">
            No {label.toLowerCase()} ownership stakes recorded
          </p>
        ) : (
          <div className="space-y-2">
            {stakes.map((stake) => {
              const approval = getApprovalStatus(stake);
              return (
                <div key={stake.id} className="flex items-center justify-between p-3 rounded-lg border">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{stake.holder_name}</span>
                      <span className="text-xs text-muted-foreground">({stake.holder_role})</span>
                      {approval && (
                        <Badge className={APPROVAL_COLORS[approval] || "bg-gray-100 text-gray-800"}>
                          {approval}
                        </Badge>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground mt-0.5">
                      {stake.publisher_or_label && <span>{stake.publisher_or_label} · </span>}
                      {stake.holder_ipi && <span>IPI: {stake.holder_ipi}</span>}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-sm">{stake.percentage.toFixed(2)}%</span>
                    {isOwner && (
                      <>
                        <Button size="icon" variant="ghost" className="h-7 w-7"
                          onClick={() => startEdit(stake)}>
                          <Pencil className="w-3 h-3" />
                        </Button>
                        <Button size="icon" variant="ghost" className="h-7 w-7 text-destructive"
                          onClick={() => deleteStake.mutate(stake.id)}>
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function OwnershipPanel({ workId, stakes, collaborators, isOwner }: Props) {
  const masterStakes = stakes.filter((s) => s.stake_type === "master");
  const pubStakes = stakes.filter((s) => s.stake_type === "publishing");
  return (
    <div className="space-y-6">
      <StakeSection label="Master" stakeType="master" stakes={masterStakes}
        workId={workId} collaborators={collaborators} isOwner={isOwner} />
      <StakeSection label="Publishing" stakeType="publishing" stakes={pubStakes}
        workId={workId} collaborators={collaborators} isOwner={isOwner} />
    </div>
  );
}
