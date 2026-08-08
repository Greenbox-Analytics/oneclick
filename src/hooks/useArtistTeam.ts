// src/hooks/useArtistTeam.ts
// Artist ownership: which team owns an artist, and moving one into a team.
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import { useEntitlements } from "@/hooks/useEntitlements";

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
