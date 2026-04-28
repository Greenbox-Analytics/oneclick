import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2, Music2, Plus } from "lucide-react";
import { useWorksByProject, type Work } from "@/hooks/useRegistry";
import { InlineEdit } from "@/components/InlineEdit";
import { useUpdateWork } from "@/hooks/useRegistry";
import AddWorkDialog from "./AddWorkDialog";

interface WorksTabProps {
  projectId: string;
  userRole: string | null;
  artistId: string;
}

const STATUS_COLORS: Record<string, string> = {
  draft: "bg-gray-500/15 text-gray-400",
  pending_approval: "bg-amber-500/15 text-amber-400",
  registered: "bg-emerald-500/15 text-emerald-400",
};

const STATUS_LABELS: Record<string, string> = {
  draft: "Draft",
  pending_approval: "Pending",
  registered: "Registered",
};

const canEdit = (role: string | null) => role === "owner" || role === "admin" || role === "editor";

export default function WorksTab({ projectId, userRole, artistId }: WorksTabProps) {
  const navigate = useNavigate();
  const { data: works, isLoading, isError } = useWorksByProject(projectId);
  const updateWork = useUpdateWork();
  const [dialogOpen, setDialogOpen] = useState(false);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <Music2 className="w-10 h-10 text-destructive/40 mb-3" />
        <p className="text-sm text-muted-foreground">Failed to load works</p>
        <p className="text-xs text-muted-foreground/60 mt-1">Please try refreshing the page</p>
      </div>
    );
  }

  const isEmpty = !works || works.length === 0;

  return (
    <>
      {isEmpty ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <Music2 className="w-10 h-10 text-muted-foreground/40 mb-3" />
          <p className="text-sm text-muted-foreground">No works yet</p>
          <p className="text-xs text-muted-foreground/60 mt-1">Add your first work to start tracking compositions and registrations</p>
          {canEdit(userRole) && (
            <div className="mt-4">
              <Button onClick={() => setDialogOpen(true)}>
                <Plus className="w-4 h-4 mr-2" /> Add Work
              </Button>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          {canEdit(userRole) && (
            <div className="flex justify-end">
              <Button size="sm" onClick={() => setDialogOpen(true)}>
                <Plus className="w-4 h-4 mr-2" /> Add Work
              </Button>
            </div>
          )}

          <div className="grid gap-3">
            {works.map((work: Work) => (
              <Card
                key={work.id}
                className="p-4 hover:bg-muted/50 transition-colors cursor-pointer"
                onClick={() => navigate(`/tools/registry/${work.id}`)}
              >
                <div className="flex items-center justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      {canEdit(userRole) ? (
                        <InlineEdit
                          value={work.title}
                          onSave={async (newTitle) => {
                            await updateWork.mutateAsync({ workId: work.id, title: newTitle });
                          }}
                          className="text-sm font-medium text-foreground"
                          inputClassName="text-sm h-7"
                        />
                      ) : (
                        <span className="text-sm font-medium text-foreground">{work.title}</span>
                      )}
                      <Badge variant="outline" className="text-xs shrink-0">
                        {work.work_type || "Single"}
                      </Badge>
                    </div>
                    {work.isrc && (
                      <p className="text-xs text-muted-foreground mt-1">ISRC: {work.isrc}</p>
                    )}
                  </div>
                  <Badge
                    className={`text-xs border-0 shrink-0 ${STATUS_COLORS[work.status] || STATUS_COLORS.draft}`}
                  >
                    {STATUS_LABELS[work.status] || work.status}
                  </Badge>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {canEdit(userRole) && (
        <AddWorkDialog
          open={dialogOpen}
          onOpenChange={setDialogOpen}
          projectId={projectId}
          artistId={artistId}
        />
      )}
    </>
  );
}
