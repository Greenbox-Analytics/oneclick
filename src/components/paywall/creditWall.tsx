// src/components/paywall/creditWall.tsx
// Shared helpers for the org-seat credit-wall (402) UX: derive the structured
// fields off an ApiError's `detail`, and render the "unlink this project"
// escape-hatch hint that an owner sees when their own project is dry on an org
// seat. Co-located so the derivation + copy stay in one place across the
// paywall card, the OneClick error alert, and the AddWork parse queue.
import { Link } from "react-router-dom";
import { cn } from "@/lib/utils";

export interface CreditWallInfo {
  /** Denial came from a seat wallet in ORG billing context. */
  managedByOrg: boolean;
  /** Where "Request credits" navigates (member request form) when present. */
  requestUrl?: string;
  /** The dry seat is on a project the CALLER OWNS and can unlink. */
  ownerCanUnlink: boolean;
  projectId?: string;
  projectName?: string;
}

/**
 * Derive the org-seat credit-wall fields from an ApiError's structured
 * `detail`. Mirrors the 402 shape from subscriptions/enforcement.py — every
 * field is presence-checked so a legacy plain-string detail (or any non-object)
 * yields an all-false/undefined result.
 */
export function parseCreditWallDetail(detail: unknown): CreditWallInfo {
  const d = (detail && typeof detail === "object" ? detail : {}) as Record<string, unknown>;
  const managedByOrg = d.managedByOrg === true;
  const ownerCanUnlink = managedByOrg && d.ownerCanUnlink === true;
  return {
    managedByOrg,
    requestUrl: managedByOrg && typeof d.requestUrl === "string" ? d.requestUrl : undefined,
    ownerCanUnlink,
    projectId: ownerCanUnlink && typeof d.projectId === "string" ? d.projectId : undefined,
    projectName: ownerCanUnlink && typeof d.projectName === "string" ? d.projectName : undefined,
  };
}

/** "Or, unlink … in its settings to use your own plan here." — shown alongside
 * the Request-credits CTA when the caller owns the dry project. */
export function UnlinkProjectHint({
  projectId,
  projectName,
  className,
}: {
  projectId?: string;
  projectName?: string;
  className?: string;
}) {
  return (
    <p className={cn("text-xs text-muted-foreground mt-1", className)}>
      Or,{" "}
      {projectId ? (
        <Link to={`/projects/${projectId}?tab=settings`} className="underline underline-offset-2">
          unlink {projectName ? `"${projectName}"` : "this project"} in its settings
        </Link>
      ) : (
        "unlink this project in its settings"
      )}{" "}
      to use your own plan here.
    </p>
  );
}
