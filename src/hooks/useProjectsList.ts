import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export interface ProjectOption {
  id: string;
  name: string;
}

export interface ContractOption {
  id: string;
  file_name: string;
}

/**
 * Fetch projects scoped to selected artist IDs, and contracts scoped to selected project IDs.
 * Chain: artist → projects → contracts
 */
export function useProjectsList(artistIds?: string[], projectIds?: string[]) {
  const { user } = useAuth();

  const projectsQuery = useQuery<ProjectOption[]>({
    queryKey: ["projects-list", user?.id, artistIds],
    queryFn: async () => {
      if (!user?.id) return [];

      if (artistIds && artistIds.length > 0) {
        // Fetch projects for each selected artist and deduplicate
        const allProjects: ProjectOption[] = [];
        const seen = new Set<string>();

        for (const artistId of artistIds) {
          let data: any;
          try { data = await apiFetch<any>(`${API_URL}/artists/${artistId}/projects`); }
          catch { continue; }
          const projects = Array.isArray(data) ? data : data.projects || data.data || [];
          for (const p of projects) {
            if (!seen.has(p.id)) {
              seen.add(p.id);
              allProjects.push({ id: p.id, name: p.name });
            }
          }
        }
        return allProjects;
      }

      // No artist selected — fetch all user projects
      const data = await apiFetch<any>(`${API_URL}/projects`);
      const projects = Array.isArray(data) ? data : data.projects || data.data || [];
      return projects.map((p: Record<string, unknown>) => ({
        id: p.id as string,
        name: p.name as string,
      }));
    },
    enabled: !!user?.id,
  });

  const contractsQuery = useQuery<ContractOption[]>({
    queryKey: ["contracts-list", user?.id, projectIds, artistIds],
    queryFn: async () => {
      if (!user?.id) return [];

      // If specific projects selected, fetch contracts from those projects
      if (projectIds && projectIds.length > 0) {
        const allContracts: ContractOption[] = [];
        const seen = new Set<string>();

        for (const projId of projectIds) {
          let data: any;
          try { data = await apiFetch<any>(`${API_URL}/projects/${projId}/contracts`); }
          catch { continue; }
          const files = Array.isArray(data) ? data : data.contracts || data.data || [];
          for (const f of files) {
            if (!seen.has(f.id)) {
              seen.add(f.id);
              allContracts.push({ id: f.id, file_name: f.file_name });
            }
          }
        }
        return allContracts;
      }

      // If artists selected but no projects, fetch contracts from all artist projects
      if (artistIds && artistIds.length > 0) {
        const projects = projectsQuery.data || [];
        const allContracts: ContractOption[] = [];
        const seen = new Set<string>();

        for (const proj of projects) {
          let data: any;
          try { data = await apiFetch<any>(`${API_URL}/projects/${proj.id}/contracts`); }
          catch { continue; }
          const files = Array.isArray(data) ? data : data.contracts || data.data || [];
          for (const f of files) {
            if (!seen.has(f.id)) {
              seen.add(f.id);
              allContracts.push({ id: f.id, file_name: f.file_name });
            }
          }
        }
        return allContracts;
      }

      return [];
    },
    enabled: !!user?.id && (
      (projectIds !== undefined && projectIds.length > 0) ||
      (artistIds !== undefined && artistIds.length > 0 && (projectsQuery.data?.length || 0) > 0)
    ),
  });

  return {
    projects: projectsQuery.data || [],
    contracts: contractsQuery.data || [],
    isLoading: projectsQuery.isLoading || contractsQuery.isLoading,
  };
}
