import { ChevronRight, FileText, Folder, AlertTriangle, Eye } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Artwork } from "./Artwork";
import { ReleaseTag } from "./ReleaseTag";
import { RegistryStatusBadge } from "./RegistryStatusBadge";
import { RegistryAvatar } from "./RegistryAvatar";
import type { Work } from "@/hooks/useRegistry";

export interface DashboardWork extends Work {
  /** Derived: !!release_date */
  released: boolean;
  /** Resolved artist info (joined client-side). */
  artist?: { id: string; name: string };
  /** Resolved project info (joined client-side). */
  project?: { id: string; name: string };
  /** Pre-computed issue list (empty when work has no flagged issues). */
  issues?: string[];
  /** Visible collaborator-name list for the avatar stack. */
  collaboratorNames?: string[];
  /** Total related-doc count for the row chip. */
  documentCount?: number;
  /** Whether the user owns this work or only collaborates on it. */
  ownership?: "owner" | "collaborator";
  /** "Can edit" / "View only" — only shown for collaborator works. */
  canEdit?: boolean;
}

interface WorkRowProps {
  work: DashboardWork;
  onOpen: (workId: string) => void;
  onOpenProject?: (projectId: string) => void;
}

export function WorkRow({ work, onOpen, onOpenProject }: WorkRowProps) {
  const issues = work.issues || [];
  const collabs = work.collaboratorNames || [];
  const docCount = work.documentCount ?? 0;

  return (
    <Card
      className="flex items-center gap-3 p-3 hover:bg-muted/50 hover:border-primary/40 transition-colors cursor-pointer"
      onClick={() => onOpen(work.id)}
    >
      <Artwork seed={work.title} hasArtwork={work.released} size={44} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm font-semibold tracking-tight">{work.title}</span>
          <ReleaseTag released={work.released} />
          <RegistryStatusBadge status={work.status} />
          {issues.length > 0 && (
            <span
              className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-full px-2 py-0.5"
              title={issues.join(" · ")}
            >
              <AlertTriangle className="w-3 h-3" />
              {issues.length === 1 ? issues[0] : `${issues.length} issues`}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 flex-wrap mt-1.5 text-[12px] text-muted-foreground">
          {work.artist && <span>{work.artist.name}</span>}
          {work.project && (
            <button
              type="button"
              className="inline-flex items-center gap-1 hover:text-foreground"
              title={`Open ${work.project.name}`}
              onClick={(e) => {
                e.stopPropagation();
                onOpenProject?.(work.project!.id);
              }}
            >
              <Folder className="w-3 h-3" />
              {work.project.name}
            </button>
          )}
          {work.isrc && <span className="font-mono text-[11px]">ISRC {work.isrc}</span>}
          {docCount > 0 && (
            <span className="inline-flex items-center gap-1">
              <FileText className="w-3 h-3" />
              {docCount}
            </span>
          )}
        </div>
      </div>
      <div className="flex items-center gap-3 shrink-0">
        {work.ownership === "collaborator" && (
          <Badge
            variant="outline"
            className="text-[10px] gap-1 border-blue-500/30 text-blue-400 bg-blue-500/10"
          >
            <Eye className="w-3 h-3" /> {work.canEdit ? "Can edit" : "View only"}
          </Badge>
        )}
        {collabs.length > 0 && (
          <div className="flex -space-x-1.5">
            {collabs.slice(0, 3).map((name, i) => (
              <RegistryAvatar key={i} name={name} size={24} className="border-2 border-card" />
            ))}
          </div>
        )}
        <ChevronRight className="w-4 h-4 text-muted-foreground" />
      </div>
    </Card>
  );
}
