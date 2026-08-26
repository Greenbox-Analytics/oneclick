// src/hooks/useArtistTeam.ts
// Artist ownership: which team owns an artist, and moving one into a team.
import { useMemo } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import { useEntitlements } from "@/hooks/useEntitlements";
import { useBillingContextSwitcher } from "@/hooks/useBillingContext";

/**
 * The org whose billing context is active, or null when billing is personal.
 * Artist creation defaults to this team — a roster that ends up half-private
 * because someone forgot to switch context is the failure mode this prevents.
 *
 * Returns null whenever LICENSING_ENABLED is off, because `billingContext` is
 * only present when it is on: every team affordance disappears with the flag.
 */
export function useActiveTeam(): { orgId: string; orgName: string } | null {
  const { data: ent } = useEntitlements();
  const ctx = ent?.billingContext;
  return ctx && ctx.type === "org" ? { orgId: ctx.orgId, orgName: ctx.orgName } : null;
}

/**
 * Scope artists to the ACTIVE billing context: personal shows `team_id === null`,
 * an org shows only that org's. Display only — access is decided server-side by
 * artist_access (the mirror of SQL can_access_artist) and is never softened here.
 */
export function scopeArtistsToContext<T extends { team_id?: string | null }>(
  artists: T[],
  scopeId: string | null,
  canSwitch: boolean,
): T[] {
  // Nothing to switch between => never hide anything. The rollback guard: with
  // LICENSING_ENABLED off the header pill doesn't render, and filtering to
  // `team_id === null` there would strand every already-transferred artist
  // behind a control that no longer exists.
  if (!canSwitch) return artists;
  return artists.filter((a) => (a.team_id ?? null) === scopeId);
}

export function useContextScopedArtists<T extends { team_id?: string | null }>(
  artists: T[] | undefined,
): T[] {
  const activeTeam = useActiveTeam();
  const { canSwitch } = useBillingContextSwitcher();
  const scopeId = activeTeam?.orgId ?? null;

  return useMemo(
    () => scopeArtistsToContext(artists ?? [], scopeId, canSwitch),
    [artists, canSwitch, scopeId],
  );
}

export function useTransferArtistToTeam() {
  const qc = useQueryClient();
  return useMutation<unknown, Error, { orgId: string; artistId: string }>({
    mutationFn: ({ orgId, artistId }) =>
      apiFetch(`${API_URL}/orgs/${orgId}/artists/${artistId}/transfer`, { method: "POST" }),
    onSuccess: () => {
      // Ownership changes who can see the artist and who pays for it, so the
      // portfolio, the artist itself and the personal artist-cap count are all
      // stale afterwards.
      qc.invalidateQueries({ queryKey: ["portfolio-artists"] });
      qc.invalidateQueries({ queryKey: ["artists-count"] });
      qc.invalidateQueries({ queryKey: ["artist"] });
      qc.invalidateQueries({ queryKey: ["entitlements"] });
      toast.success("Artist moved to your team");
    },
    onError: (e) => toast.error(e instanceof Error ? e.message : "Couldn't move this artist."),
  });
}
