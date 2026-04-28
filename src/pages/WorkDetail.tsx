import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import {
  useWorkFull, useUpdateWork, useDeleteWork,
  useConfirmStake, useDeclineInvitation,
  type Collaborator,
} from "@/hooks/useRegistry";
import { useWorkFiles } from "@/hooks/useWorkFiles";
import { useWorkAudio } from "@/hooks/useWorkAudio";
import { InlineEdit } from "@/components/InlineEdit";
import CollaborationStatus from "@/components/registry/CollaborationStatus";
import OwnershipPanel from "@/components/registry/OwnershipPanel";
import LicensingPanel from "@/components/registry/LicensingPanel";
import AgreementsPanel from "@/components/registry/AgreementsPanel";
import ProofOfOwnership from "@/components/registry/ProofOfOwnership";
import InviteCollaboratorModal from "@/components/registry/InviteCollaboratorModal";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
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
  FileText, Headphones, BookOpen,
} from "lucide-react";
import { toast } from "sonner";
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";

const STATUS_COLORS: Record<string, string> = {
  draft: "bg-gray-500/15 text-gray-400",
  pending_approval: "bg-amber-500/15 text-amber-400",
  registered: "bg-emerald-500/15 text-emerald-400",
};

const WorkDetail = () => {
  const { workId } = useParams<{ workId: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const workQuery = useWorkFull(workId);
  const updateWork = useUpdateWork();
  const deleteWork = useDeleteWork();
  const confirmStake = useConfirmStake();
  const declineInvitation = useDeclineInvitation();
  const workFilesQuery = useWorkFiles(workId);
  const workAudioQuery = useWorkAudio(workId);

  const [showEdit, setShowEdit] = useState(false);
  const [editTitle, setEditTitle] = useState("");
  const [editIsrc, setEditIsrc] = useState("");
  const [editIswc, setEditIswc] = useState("");
  const [editUpc, setEditUpc] = useState("");
  const [editStatus, setEditStatus] = useState("");
  const [editReleaseDate, setEditReleaseDate] = useState("");
  const [editNotes, setEditNotes] = useState("");

  const [showInvite, setShowInvite] = useState(false);

  // Tour
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(TOOL_CONFIGS.work_detail, {
    onComplete: () => markToolCompleted("work_detail"),
  });

  const work = workQuery.data;
  const isOwner = work?.user_id === user?.id;

  useEffect(() => {
    if (!onboardingLoading && !statuses.work_detail && walkthrough.phase === "idle" && work) {
      const timer = setTimeout(() => walkthrough.startModal(), 500);
      return () => clearTimeout(timer);
    }
  }, [onboardingLoading, statuses.work_detail, work]);

  // Find this user's collaborator record (if they're a collaborator)
  const myCollab: Collaborator | undefined = work?.collaborators?.find(
    (c) => c.collaborator_user_id === user?.id
  );

  const collaborators = work?.collaborators || [];
  const workFiles = workFilesQuery.data || [];
  const workAudio = workAudioQuery.data || [];

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

  const handleDecline = () => {
    if (myCollab) declineInvitation.mutate(myCollab.id);
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
        <Button variant="outline" onClick={() => navigate(-1)}>
          <ArrowLeft className="w-4 h-4 mr-2" /> Back
        </Button>
      </div>
    );
  }

  // Display label for work type — use custom_work_type when work_type is "other"
  const workTypeLabel =
    work.work_type === "other" && (work as { custom_work_type?: string }).custom_work_type
      ? (work as { custom_work_type?: string }).custom_work_type
      : work.work_type.replace("_", " ").toUpperCase();

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
          <div className="flex items-center gap-2">
            <ToolHelpButton onClick={() => walkthrough.replay()} />
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/docs")}
              title="Documentation"
              className="text-muted-foreground hover:text-foreground"
            >
              <BookOpen className="w-4 h-4" />
            </Button>
            <Button variant="outline" onClick={() => navigate(`/projects/${work.project_id}`)}>
              <ArrowLeft className="w-4 h-4 mr-2" /> Back to Project
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Collaborator action banner */}
        {myCollab && !isOwner && myCollab.status === "invited" && (
          <div className="mb-6 p-4 rounded-lg border-2 border-amber-300 bg-amber-50">
            <p className="text-sm font-medium text-amber-900 mb-3">
              You've been listed as <strong>{myCollab.role}</strong> on this work.
              Please review the ownership details and accept or decline.
            </p>
            <div className="flex gap-2">
              <Button size="sm" onClick={handleConfirm} disabled={confirmStake.isPending}>
                {confirmStake.isPending ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <CheckCircle className="w-3 h-3 mr-1" />}
                Accept
              </Button>
              <Button size="sm" variant="destructive" onClick={handleDecline} disabled={declineInvitation.isPending}>
                {declineInvitation.isPending ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <XCircle className="w-3 h-3 mr-1" />}
                Decline
              </Button>
            </div>
          </div>
        )}

        {/* Work Header */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <Shield className="w-6 h-6 text-primary" />
              <InlineEdit
                value={work.title}
                onSave={async (newTitle) => {
                  await updateWork.mutateAsync({ workId: work.id, title: newTitle });
                }}
                disabled={!isOwner}
                className="text-2xl font-bold"
              />
              <Badge data-walkthrough="work-status" className={STATUS_COLORS[work.status] || ""}>{work.status.replace("_", " ")}</Badge>
            </div>
            <div data-walkthrough="work-header" className="flex flex-wrap gap-4 text-sm text-muted-foreground">
              <span>{workTypeLabel}</span>
              {work.isrc && <span>ISRC: {work.isrc}</span>}
              {work.iswc && <span>ISWC: {work.iswc}</span>}
              {work.upc && <span>UPC: {work.upc}</span>}
              {work.release_date && <span>Released: {work.release_date}</span>}
            </div>
          </div>
          {isOwner && (
            <div data-walkthrough="work-actions" className="flex items-center gap-2">
              {/* Register button for zero-collaborator draft works */}
              {work.status === "draft" && collaborators.length === 0 && (
                <Button size="sm" onClick={async () => {
                  await updateWork.mutateAsync({ workId: work.id, status: "registered" });
                  toast.success("Work registered");
                }} disabled={updateWork.isPending}>
                  {updateWork.isPending ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : <CheckCircle className="w-4 h-4 mr-1" />}
                  Register
                </Button>
              )}
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
                          <SelectItem value="pending_approval">Pending Approval</SelectItem>
                          <SelectItem value="registered">Registered</SelectItem>
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
                      <label className="text-sm font-medium">Description</label>
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
          <TabsList data-walkthrough="work-tabs" className="mb-6">
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

        {/* Linked Files Section */}
        {workFiles.length > 0 && (
          <div className="mt-6 rounded-lg border p-4">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <FileText className="w-4 h-4 text-muted-foreground" /> Linked Files
            </h3>
            <ul className="space-y-1.5">
              {workFiles.map((link) => (
                <li key={link.id} className="flex items-center gap-2 text-sm">
                  <FileText className="w-3.5 h-3.5 text-muted-foreground" />
                  {link.project_files ? (
                    <a href={link.project_files.file_url} target="_blank" rel="noopener noreferrer"
                      className="text-primary hover:underline">
                      {link.project_files.file_name}
                    </a>
                  ) : (
                    <span className="text-muted-foreground">File unavailable</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Linked Audio Section */}
        {workAudio.length > 0 && (
          <div className="mt-4 rounded-lg border p-4">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <Headphones className="w-4 h-4 text-muted-foreground" /> Linked Audio
            </h3>
            <ul className="space-y-1.5">
              {workAudio.map((link) => (
                <li key={link.id} className="flex items-center gap-2 text-sm">
                  <Headphones className="w-3.5 h-3.5 text-muted-foreground" />
                  {link.audio_files ? (
                    <a href={link.audio_files.file_url} target="_blank" rel="noopener noreferrer"
                      className="text-primary hover:underline">
                      {link.audio_files.file_name}
                    </a>
                  ) : (
                    <span className="text-muted-foreground">Audio file unavailable</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
      </main>

      <ToolIntroModal
        config={TOOL_CONFIGS.work_detail}
        isOpen={walkthrough.phase === "modal"}
        onStartTour={walkthrough.startSpotlight}
        onSkip={walkthrough.skip}
      />
      <WalkthroughProvider
        isActive={walkthrough.phase === "spotlight"}
        currentStep={walkthrough.currentStep}
        currentStepIndex={walkthrough.visibleStepIndex}
        totalSteps={walkthrough.totalSteps}
        onNext={walkthrough.next}
        onSkip={walkthrough.skip}
      />
    </div>
  );
};

export default WorkDetail;
