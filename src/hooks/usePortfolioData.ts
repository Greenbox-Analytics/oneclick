import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export interface ArtistInfo {
  id: string;
  name: string;
  avatar?: string;
}

export interface ProjectCard {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  artist_id: string;
  artist_name: string;
  artist_avatar?: string;
  member_count: number;
  work_count: number;
}

export interface SharedProjectCard extends ProjectCard {
  role: "admin" | "editor" | "viewer";
}

export interface PortfolioFilters {
  selectedArtistIds: string[];
  searchQuery: string;
  sortOrder: "alpha" | "newest" | "oldest";
}

export function usePortfolioData(filters: PortfolioFilters) {
  const { user } = useAuth();

  // Query 1: Fetch artists from backend API
  const artistsQuery = useQuery<ArtistInfo[]>({
    queryKey: ["portfolio-artists", user?.id],
    queryFn: async () => {
      if (!user?.id) return [];
      const data = await apiFetch<unknown>(`${API_URL}/artists`);
      const rows = Array.isArray(data)
        ? data
        : ((data as { artists?: unknown[]; data?: unknown[] })?.artists || (data as { data?: unknown[] })?.data || []);
      return rows.map(
        (a: Record<string, unknown>) => ({
          id: a.id as string,
          name: a.name as string,
          avatar: (a.avatar_url as string) || (a.avatar as string) || undefined,
        })
      );
    },
    enabled: !!user?.id,
  });

  const artistMap = useMemo(() => {
    const map = new Map<string, ArtistInfo>();
    for (const a of artistsQuery.data || []) map.set(a.id, a);
    return map;
  }, [artistsQuery.data]);

  const artistIds = useMemo(() => (artistsQuery.data || []).map(a => a.id), [artistsQuery.data]);

  // Query 2: Fetch projects for user's artists
  const projectsQuery = useQuery({
    queryKey: ["portfolio-projects", artistIds],
    queryFn: async () => {
      if (artistIds.length === 0) return [];
      const { data, error } = await supabase
        .from("projects")
        .select("*")
        .in("artist_id", artistIds)
        .order("created_at", { ascending: false });
      if (error) { console.error("Error fetching projects:", error); return []; }
      return data || [];
    },
    enabled: artistIds.length > 0,
  });

  const projectIds = useMemo(() => (projectsQuery.data || []).map(p => p.id), [projectsQuery.data]);

  // Query 3: Fetch member counts per project
  const memberCountsQuery = useQuery<Map<string, number>>({
    queryKey: ["portfolio-member-counts", projectIds],
    queryFn: async () => {
      if (projectIds.length === 0) return new Map();
      const map = new Map<string, number>();
      const batchSize = 100;
      for (let i = 0; i < projectIds.length; i += batchSize) {
        const batch = projectIds.slice(i, i + batchSize);
        const { data, error } = await supabase
          .from("project_members")
          .select("project_id")
          .in("project_id", batch);
        if (error) { console.error("Error fetching member counts:", error); continue; }
        for (const row of data || []) {
          map.set(row.project_id, (map.get(row.project_id) || 0) + 1);
        }
      }
      return map;
    },
    enabled: projectIds.length > 0,
  });

  // Query 4: Fetch work counts per project
  const workCountsQuery = useQuery<Map<string, number>>({
    queryKey: ["portfolio-work-counts", projectIds],
    queryFn: async () => {
      if (projectIds.length === 0) return new Map();
      const map = new Map<string, number>();
      const batchSize = 100;
      for (let i = 0; i < projectIds.length; i += batchSize) {
        const batch = projectIds.slice(i, i + batchSize);
        const { data, error } = await supabase
          .from("works_registry")
          .select("project_id")
          .in("project_id", batch);
        if (error) { console.error("Error fetching work counts:", error); continue; }
        for (const row of data || []) {
          map.set(row.project_id, (map.get(row.project_id) || 0) + 1);
        }
      }
      return map;
    },
    enabled: projectIds.length > 0,
  });

  // Query 5: Fetch shared projects (where user is a member but NOT the owner)
  const sharedProjectsQuery = useQuery<SharedProjectCard[]>({
    queryKey: ["portfolio-shared-projects", user?.id],
    queryFn: async () => {
      if (!user?.id) return [];
      // Get all project memberships for current user where role is not owner
      const { data: memberships, error: memberError } = await supabase
        .from("project_members")
        .select("project_id, role")
        .eq("user_id", user.id)
        .neq("role", "owner");
      if (memberError) { console.error("Error fetching shared memberships:", memberError); return []; }
      if (!memberships || memberships.length === 0) return [];

      const sharedProjectIds = memberships.map(m => m.project_id);
      const roleMap = new Map(memberships.map(m => [m.project_id, m.role]));

      // Fetch the projects
      const { data: projects, error: projectError } = await supabase
        .from("projects")
        .select("*")
        .in("id", sharedProjectIds);
      if (projectError) { console.error("Error fetching shared projects:", projectError); return []; }
      if (!projects || projects.length === 0) return [];

      // Fetch artists for these projects
      const sharedArtistIds = [...new Set(projects.map(p => p.artist_id))];
      const { data: artists, error: artistError } = await supabase
        .from("artists")
        .select("id, name, avatar_url")
        .in("id", sharedArtistIds);
      if (artistError) console.error("Error fetching shared artists:", artistError);
      const sharedArtistMap = new Map((artists || []).map(a => [a.id, a]));

      // Fetch member counts for shared projects
      const { data: memberRows } = await supabase
        .from("project_members")
        .select("project_id")
        .in("project_id", sharedProjectIds);
      const sharedMemberCounts = new Map<string, number>();
      for (const row of memberRows || []) {
        sharedMemberCounts.set(row.project_id, (sharedMemberCounts.get(row.project_id) || 0) + 1);
      }

      // Fetch work counts for shared projects
      const { data: workRows } = await supabase
        .from("works_registry")
        .select("project_id")
        .in("project_id", sharedProjectIds);
      const sharedWorkCounts = new Map<string, number>();
      for (const row of workRows || []) {
        sharedWorkCounts.set(row.project_id, (sharedWorkCounts.get(row.project_id) || 0) + 1);
      }

      return projects.map(p => {
        const artist = sharedArtistMap.get(p.artist_id);
        return {
          id: p.id,
          name: p.name,
          description: p.description,
          created_at: p.created_at,
          artist_id: p.artist_id,
          artist_name: artist?.name || "Unknown",
          artist_avatar: artist?.avatar_url || undefined,
          member_count: sharedMemberCounts.get(p.id) || 0,
          work_count: sharedWorkCounts.get(p.id) || 0,
          role: roleMap.get(p.id) as "admin" | "editor" | "viewer",
        };
      });
    },
    enabled: !!user?.id,
  });

  // Build filtered + sorted project cards grouped by artist
  const myProjects = useMemo((): ProjectCard[] => {
    const projects = projectsQuery.data || [];
    const search = filters.searchQuery.toLowerCase();
    const memberCounts = memberCountsQuery.data || new Map();
    const workCounts = workCountsQuery.data || new Map();

    const filtered = projects.filter(p => {
      if (filters.selectedArtistIds.length > 0 && !filters.selectedArtistIds.includes(p.artist_id)) {
        return false;
      }
      if (search && !p.name.toLowerCase().includes(search)) {
        return false;
      }
      return true;
    });

    const cards: ProjectCard[] = filtered.map(p => {
      const artist = artistMap.get(p.artist_id);
      return {
        id: p.id,
        name: p.name,
        description: p.description,
        created_at: p.created_at,
        artist_id: p.artist_id,
        artist_name: artist?.name || "Unknown",
        artist_avatar: artist?.avatar,
        member_count: memberCounts.get(p.id) || 0,
        work_count: workCounts.get(p.id) || 0,
      };
    });

    // Sort
    cards.sort((a, b) => {
      if (filters.sortOrder === "alpha") return a.name.localeCompare(b.name);
      if (filters.sortOrder === "oldest") return a.created_at.localeCompare(b.created_at);
      return b.created_at.localeCompare(a.created_at); // newest
    });

    return cards;
  }, [projectsQuery.data, filters, artistMap, memberCountsQuery.data, workCountsQuery.data]);

  // Filter shared projects by search and artist
  const sharedProjects = useMemo((): SharedProjectCard[] => {
    const projects = sharedProjectsQuery.data || [];
    const search = filters.searchQuery.toLowerCase();

    const filtered = projects.filter(p => {
      if (filters.selectedArtistIds.length > 0 && !filters.selectedArtistIds.includes(p.artist_id)) {
        return false;
      }
      if (search && !p.name.toLowerCase().includes(search)) {
        return false;
      }
      return true;
    });

    filtered.sort((a, b) => {
      if (filters.sortOrder === "alpha") return a.name.localeCompare(b.name);
      if (filters.sortOrder === "oldest") return a.created_at.localeCompare(b.created_at);
      return b.created_at.localeCompare(a.created_at);
    });

    return filtered;
  }, [sharedProjectsQuery.data, filters]);

  return {
    myProjects,
    sharedProjects,
    allArtists: artistsQuery.data || [],
    isLoading: artistsQuery.isLoading || projectsQuery.isLoading,
    refetchProjects: projectsQuery.refetch,
  };
}
