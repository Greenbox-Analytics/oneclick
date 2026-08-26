// src/hooks/useWorkspaceScope.ts
// The active workspace: which artists — and therefore which projects, works,
// files, boards and tool inputs — every list should show.
//
// One rule decides membership: an item's workspace is the owner of its artist
// (`artists.team_id`; NULL = personal). Anything reachable only through a
// membership grant (project_members, registry_collaborators) belongs to
// Personal. The server enforces it; this hook exists so the client asks the
// right question and CACHES the answer under the right key.
//
// Why the scope rides on the request rather than being read server-side from
// the stored context: React Query caches by key. An ambient server-side scope
// would keep serving org A's roster from cache after a switch to org B, because
// nothing in the key changed. In the URL it is also in the key, and correctness
// falls out.
import { useCallback, useMemo } from "react";
import { useEntitlements, type WorkspaceScope } from "@/hooks/useEntitlements";

/** Cache-key fragment used when scoping is off — distinct from "personal", so
 *  flipping the flag invalidates rather than colliding with scoped entries. */
export const UNSCOPED_KEY = "unscoped";

export interface WorkspaceScopeState {
  /** The active workspace. Defaults to personal; meaningless until `ready`. */
  scope: WorkspaceScope;
  /** Org id, or null for Personal. */
  scopeId: string | null;
  /** Human label for empty states — "Personal" or the org's name. */
  scopeLabel: string;
  /** Put this in EVERY scoped React Query key. */
  scopeKey: string;
  /** Query-param value, or undefined when scoping is off (send no param). */
  scopeParam: string | undefined;
  /** True when workspace scoping is live for this user. */
  enabled: boolean;
  /**
   * Entitlements have resolved (or failed). Gate scoped queries on this:
   * fetching before the scope is known sends a param-less request, caches it
   * under the wrong key, then refetches — a double fetch and a visible flash
   * of the wrong workspace on every cold load.
   */
  ready: boolean;
  /** Append `?scope=`/`&scope=` to a URL, or return it untouched when off. */
  withScope: (url: string) => string;
}

const PERSONAL: WorkspaceScope = { type: "personal" };

export function useWorkspaceScope(): WorkspaceScopeState {
  // `isLoading` (not `isPending`) so an ERRORED entitlements fetch still counts
  // as ready: with no payload we send no param and the server falls back to the
  // caller's stored context. Degrading to the server's own default beats
  // blocking every list behind a failed request.
  const { data, isLoading } = useEntitlements();

  const scope = data?.workspaceScope ?? null;
  const enabled = scope != null;
  const scopeId = scope?.type === "org" ? scope.orgId : null;

  const withScope = useCallback(
    (url: string) => {
      if (!enabled) return url;
      const value = scopeId ?? "personal";
      return `${url}${url.includes("?") ? "&" : "?"}scope=${encodeURIComponent(value)}`;
    },
    [enabled, scopeId],
  );

  return useMemo(
    () => ({
      scope: scope ?? PERSONAL,
      scopeId,
      scopeLabel: scope?.type === "org" ? scope.orgName : "Personal",
      scopeKey: enabled ? (scopeId ?? "personal") : UNSCOPED_KEY,
      scopeParam: enabled ? (scopeId ?? "personal") : undefined,
      enabled,
      ready: !isLoading,
      withScope,
    }),
    [scope, scopeId, enabled, isLoading, withScope],
  );
}
