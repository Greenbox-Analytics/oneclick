import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import {
  useWorkFull, useUpdateWork, useDeleteWork,
  useConfirmStake, useDisputeStake,
  useResendInvitation, useRevokeCollaborator,
  type Collaborator,
} from "@/hooks/useRegistry";
import CollaborationStatus from "@/components/registry/CollaborationStatus";
import OwnershipPanel from "@/components/registry/OwnershipPanel";
import LicensingPanel from "@/components/registry/LicensingPanel";
import AgreementsPanel from "@/components/registry/AgreementsPanel";
import ProofOfOwnership from "@/components/registry/ProofOfOwnership";
import InviteCollaboratorModal from "@/components/registry/InviteCollaboratorModal";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import {
  Music, ArrowLeft, Loader2, Shield, Pencil, Trash2,
  Users, Scale, FileCheck, CheckCircle, XCircle, UserPlus,
} from "lucide-react";

const STATUS_COLORS: Record<string, string> = {
  draft: "bg-yellow-100 text-yellow-800",
  pending_approval: "bg-amber-100 text-amber-800",
  registered: "bg-green-100 text-green-800",
  disputed: "bg-red-100 text-red-800",
};

const WorkDetail = () => {
  const { workId } = useParams<{ workId: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const workQuery = useWorkFull(workId);
  const updateWork = useUpdateWork();
  const deleteWork = useDeleteWork();
  const confirmStake = useConfirmStake();
  const disputeStake = useDisputeStake();

  const [showEdit, setShowEdit] = useState(false);
  const [editTitle, setEditTitle] = useState("");
  const [editIsrc, setEditIsrc] = useState("");
  const [editIswc, setEditIswc] = useState("");
  const [editUpc, setEditUpc] = useState("");
  const [editStatus, setEditStatus] = useState("");
  const [editReleaseDate, setEditReleaseDate] = useState("");
  const [editNotes, setEditNotes] = useState("");

  const [showDisputeDialog, setShowDisputeDialog] = useState(false);
  const [disputeReason, setDisputeReason] = useState("");
  const [disputeCollabId, setDisputeCollabId] = useState("");

  const [showInvite, setShowInvite] = useState(false);

  const work = workQuery.data;
  const isOwner = work?.user_id === user?.id;

  // Find this user's collaborator record (if they're a collaborator)
  const myCollab: Collaborator | undefined = work?.collaborators?.find(
    (c) => c.collaborator_user_id === user?.id
  );

  const openEdit = () => {
    if (!work) return;
    setEditTitle(work.title); setEditIsrc(work.isrc || ""); setEditIswc(work.iswc || "");
    setEditUpc(work.upc || ""); setEditStatus(work.status);
    setEditReleaseDate(work.release_date || ""); setEditNotes(work.notes || "");
    setShowEdit(true);
  };

  const handleUpdate = async () => {
    if (!workId || !editTitle.trim()) return;
    await updateWork.mutateAsync({
      workId, title: editTitle.trim(),
      isrc: editIsrc.trim() || undefined, iswc: editIswc.trim() || undefined,
      upc: editUpc.trim() || undefined, status: editStatus,
      release_date: editReleaseDate || undefined, notes: editNotes.trim() || undefined,
    });
    setShowEdit(false);
  };

  const handleDelete = async () => {
    if (!workId) return;
    if (!window.confirm("Delete this work and all its ownership, licensing, and agreement records?")) return;
    await deleteWork.mutateAsync(workId);
    navigate("/tools/registry");
  };

  const handleConfirm = () => {
    if (myCollab) confirmStake.mutate(myCollab.id);
  };

  const openDispute = () => {
    if (myCollab) {
      setDisputeCollabId(myCollab.id);
      setDisputeReason("");
      setShowDisputeDialog(true);
    }
  };

  const handleDispute = async () => {
    if (!disputeCollabId || !disputeReason.trim()) return;
    await disputeStake.mutateAsync({ collaboratorId: disputeCollabId, reason: disputeReason.trim() });
    setShowDisputeDialog(false);
  };

  if (workQuery.isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!work) {
    return (
      <div className="min-h-screen bg-background flex flex-col items-center justify-center gap-4">
        <p className="text-muted-foreground">Work not found</p>
        <Button variant="outline" onClick={() => navigate("/tools/registry")}>
          <ArrowLeft className="w-4 h-4 mr-2" /> Back to Registry
        </Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate("/dashboard")}>
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
          </div>
          <Button variant="outline" onClick={() => navigate("/tools/registry")}>
            <ArrowLeft className="w-4 h-4 mr-2" /> Back to Registry
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Collaborator action banner */}
        {myCollab && !isOwner && myCollab.status === "invited" && (
          <div className="mb-6 p-4 rounded-lg border-2 border-amber-300 bg-amber-50">
            <p className="text-sm font-medium text-amber-900 mb-3">
              You've been listed as <strong>{myCollab.role}</strong> on this work.
              Please review the ownership details and confirm or dispute your stake.
            </p>
            <div className="flex gap-2">
              <Button size="sm" onClick={handleConfirm} disabled={confirmStake.isPending}>
                {confirmStake.isPending ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <CheckCircle className="w-3 h-3 mr-1" />}
                Confirm My Stake
              </Button>
              <Button size="sm" variant="destructive" onClick={openDispute}>
                <XCircle className="w-3 h-3 mr-1" /> Dispute
              </Button>
            </div>
          </div>
        )}

        {/* Work Header */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <Shield className="w-6 h-6 text-primary" />
              <h2 className="text-2xl font-bold text-foreground">{work.title}</h2>
              <Badge className={STATUS_COLORS[work.status] || ""}>{work.status.replace("_", " ")}</Badge>
            </div>
            <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
              <span>{work.work_type.replace("_", " ").toUpperCase()}</span>
              {work.isrc && <span>ISRC: {work.isrc}</span>}
              {work.iswc && <span>ISWC: {work.iswc}</span>}
              {work.upc && <span>UPC: {work.upc}</span>}
              {work.release_date && <span>Released: {work.release_date}</span>}
            </div>
          </div>
          {isOwner && (
            <div className="flex items-center gap-2">
              <ProofOfOwnership workId={work.id} />
              <Button variant="outline" size="sm" onClick={() => setShowInvite(true)}>
                <UserPlus className="w-4 h-4 mr-1" /> Invite
              </Button>
              <InviteCollaboratorModal workId={work.id} stakes={work.stakes || []}
                open={showInvite} onOpenChange={setShowInvite} />
              <Dialog open={showEdit} onOpenChange={setShowEdit}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm" onClick={openEdit}>
                    <Pencil className="w-4 h-4 mr-1" /> Edit
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader><DialogTitle>Edit Work</DialogTitle></DialogHeader>
                  <div className="space-y-3 pt-2">
                    <div>
                      <label className="text-sm font-medium">Title *</label>
                      <Input value={editTitle} onChange={(e) => setEditTitle(e.target.value)} />
                    </div>
                    <div>
                      <label className="text-sm font-medium">Status</label>
                      <Select value={editStatus} onValueChange={setEditStatus}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="draft">Draft</SelectItem>
                          <SelectItem value="registered">Registered</SelectItem>
                          <SelectItem value="disputed">Disputed</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                      <div><label className="text-sm font-medium">ISRC</label>
                        <Input value={editIsrc} onChange={(e) => setEditIsrc(e.target.value)} /></div>
                      <div><label className="text-sm font-medium">ISWC</label>
                        <Input value={editIswc} onChange={(e) => setEditIswc(e.target.value)} /></div>
                      <div><label className="text-sm font-medium">UPC</label>
                        <Input value={editUpc} onChange={(e) => setEditUpc(e.target.value)} /></div>
                    </div>
                    <div>
                      <label className="text-sm font-medium">Release Date</label>
                      <Input type="date" value={editReleaseDate}
                        onChange={(e) => setEditReleaseDate(e.target.value)} />
                    </div>
                    <div>
                      <label className="text-sm font-medium">Notes</label>
                      <Input value={editNotes} onChange={(e) => setEditNotes(e.target.value)} />
                    </div>
                    <Button onClick={handleUpdate} disabled={updateWork.isPending} className="w-full">
                      {updateWork.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                      Save Changes
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
              <Button variant="outline" size="sm" className="text-destructive" onClick={handleDelete}>
                <Trash2 className="w-4 h-4 mr-1" /> Delete
              </Button>
            </div>
          )}
        </div>

        {/* Collaboration Status */}
        <CollaborationStatus workId={work.id} workStatus={work.status}
          collaborators={work.collaborators || []} isOwner={isOwner} />

        {/* Tabs */}
        <Tabs defaultValue="ownership">
          <TabsList className="mb-6">
            <TabsTrigger value="ownership" className="gap-1.5">
              <Users className="w-4 h-4" /> Ownership
            </TabsTrigger>
            <TabsTrigger value="licensing" className="gap-1.5">
              <Scale className="w-4 h-4" /> Licensing
            </TabsTrigger>
            <TabsTrigger value="agreements" className="gap-1.5">
              <FileCheck className="w-4 h-4" /> Agreements
            </TabsTrigger>
          </TabsList>
          <TabsContent value="ownership">
            <OwnershipPanel workId={work.id} stakes={work.stakes || []}
              collaborators={work.collaborators || []} isOwner={isOwner} />
          </TabsContent>
          <TabsContent value="licensing">
            <LicensingPanel workId={work.id} licenses={work.licenses || []} isOwner={isOwner} />
          </TabsContent>
          <TabsContent value="agreements">
            <AgreementsPanel workId={work.id} agreements={work.agreements || []} isOwner={isOwner} />
          </TabsContent>
        </Tabs>

        {/* Dispute Dialog */}
        <Dialog open={showDisputeDialog} onOpenChange={setShowDisputeDialog}>
          <DialogContent>
            <DialogHeader><DialogTitle>Dispute Your Stake</DialogTitle></DialogHeader>
            <div className="space-y-3 pt-2">
              <p className="text-sm text-muted-foreground">
                Please explain why you're disputing this ownership stake.
              </p>
              <Textarea value={disputeReason} onChange={(e) => setDisputeReason(e.target.value)}
                placeholder="Reason for dispute..." rows={3} />
              <Button onClick={handleDispute} variant="destructive"
                disabled={!disputeReason.trim() || disputeStake.isPending} className="w-full">
                {disputeStake.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                Submit Dispute
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </main>
    </div>
  );
};

export default WorkDetail;
