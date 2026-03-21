import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import type { Tables } from "@/integrations/supabase/types";
import type { BoardTask } from "@/types/integrations";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

export interface ArtistInfo {
  id: string;
  name: string;
  avatar?: string;
}

export interface ProjectWithFiles {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  artist_id: string;
  files: Record<string, Tables<"project_files">[]>; // keyed by folder_category
  tasks: BoardTask[];
}

export interface ArtistGroup {
  artist: ArtistInfo;
  projects: ProjectWithFiles[];
}

export interface LetterGroup {
  letter: string;
  artists: ArtistGroup[];
}

export interface YearGroup {
  year: number;
  artists: ArtistGroup[];
  letterGroups: LetterGroup[];
  totalProjects: number;
}

export interface PortfolioFilters {
  selectedArtistIds: string[];
  searchQuery: string;       // matches project name + file name
  dateFrom?: string;         // ISO date string
  dateTo?: string;           // ISO date string
  sortOrder: "alpha" | "newest" | "oldest";
}

export function usePortfolioData(filters: PortfolioFilters) {
  const { user } = useAuth();

  // Query 1: Fetch artists from backend API (same pattern as useArtistsList.ts)
  const artistsQuery = useQuery<ArtistInfo[]>({
    queryKey: ["portfolio-artists", user?.id],
    queryFn: async () => {
      if (!user?.id) return [];
      const res = await fetch(`${API_URL}/artists?user_id=${user.id}`);
      if (!res.ok) return [];
      const data = await res.json();
      return (Array.isArray(data) ? data : data.artists || data.data || []).map(
        (a: Record<string, unknown>) => ({
          id: a.id as string,
          name: a.name as string,
          avatar: (a.avatar_url as string) || (a.avatar as string) || undefined,
        })
      );
    },
    enabled: !!user?.id,
  });

  const artistIds = useMemo(() => (artistsQuery.data || []).map(a => a.id), [artistsQuery.data]);

  // Query 2: Fetch ALL projects for user's artists from Supabase
  const projectsQuery = useQuery<Tables<"projects">[]>({
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

  // Query 3: Fetch ALL project files from Supabase
  const filesQuery = useQuery<Tables<"project_files">[]>({
    queryKey: ["portfolio-files", projectIds],
    queryFn: async () => {
      if (projectIds.length === 0) return [];
      // Supabase .in() has a limit, batch if needed
      const batchSize = 100;
      const allFiles: Tables<"project_files">[] = [];
      for (let i = 0; i < projectIds.length; i += batchSize) {
        const batch = projectIds.slice(i, i + batchSize);
        const { data, error } = await supabase
          .from("project_files")
          .select("*")
          .in("project_id", batch);
        if (error) { console.error("Error fetching files:", error); continue; }
        allFiles.push(...(data || []));
      }
      return allFiles;
    },
    enabled: projectIds.length > 0,
  });

  // Query 4: Fetch ALL board tasks from backend API
  const tasksQuery = useQuery<BoardTask[]>({
    queryKey: ["portfolio-tasks", user?.id],
    queryFn: async () => {
      if (!user?.id) return [];
      const res = await fetch(`${API_URL}/boards/tasks?user_id=${user.id}`);
      if (!res.ok) return [];
      const data = await res.json();
      return data.tasks || [];
    },
    enabled: !!user?.id,
  });

  // Build lookup maps
  const filesByProject = useMemo(() => {
    const map = new Map<string, Record<string, Tables<"project_files">[]>>();
    for (const file of filesQuery.data || []) {
      if (!map.has(file.project_id)) {
        map.set(file.project_id, { contract: [], split_sheet: [], royalty_statement: [], other: [] });
      }
      const bucket = map.get(file.project_id)!;
      const cat = file.folder_category || "other";
      if (!bucket[cat]) bucket[cat] = [];
      bucket[cat].push(file);
    }
    return map;
  }, [filesQuery.data]);

  const tasksByProject = useMemo(() => {
    const map = new Map<string, BoardTask[]>();
    for (const task of tasksQuery.data || []) {
      for (const pid of task.project_ids || []) {
        if (!map.has(pid)) map.set(pid, []);
        map.get(pid)!.push(task);
      }
    }
    return map;
  }, [tasksQuery.data]);

  const artistMap = useMemo(() => {
    const map = new Map<string, ArtistInfo>();
    for (const a of artistsQuery.data || []) map.set(a.id, a);
    return map;
  }, [artistsQuery.data]);

  // Compute filtered + grouped hierarchy
  const years = useMemo((): YearGroup[] => {
    const projects = projectsQuery.data || [];
    const search = filters.searchQuery.toLowerCase();

    // Step 1: Filter projects
    const filtered = projects.filter(p => {
      // Artist filter
      if (filters.selectedArtistIds.length > 0 && !filters.selectedArtistIds.includes(p.artist_id)) {
        return false;
      }
      // Date filter
      if (filters.dateFrom && p.created_at < filters.dateFrom) return false;
      if (filters.dateTo && p.created_at > filters.dateTo) return false;
      // Text search: match project name OR any file name in this project
      if (search) {
        const nameMatch = p.name.toLowerCase().includes(search);
        const filesMap = filesByProject.get(p.id);
        const fileMatch = filesMap && Object.values(filesMap).flat().some(
          f => f.file_name.toLowerCase().includes(search)
        );
        if (!nameMatch && !fileMatch) return false;
      }
      return true;
    });

    // Step 2: Group by year
    const yearMap = new Map<number, Map<string, Tables<"projects">[]>>();
    for (const p of filtered) {
      const year = new Date(p.created_at).getFullYear();
      if (!yearMap.has(year)) yearMap.set(year, new Map());
      const artistGroup = yearMap.get(year)!;
      if (!artistGroup.has(p.artist_id)) artistGroup.set(p.artist_id, []);
      artistGroup.get(p.artist_id)!.push(p);
    }

    // Step 3: Build output sorted by year desc, artists alphabetical
    return Array.from(yearMap.entries())
      .sort(([a], [b]) => b - a) // years descending
      .map(([year, artistGroups]) => {
        const artists = Array.from(artistGroups.entries())
          .map(([artistId, projects]) => ({
            artist: artistMap.get(artistId) || { id: artistId, name: "Unknown" },
            projects: projects
              .sort((a, b) => {
                if (filters.sortOrder === "alpha") return a.name.localeCompare(b.name);
                if (filters.sortOrder === "oldest") return a.created_at.localeCompare(b.created_at);
                return b.created_at.localeCompare(a.created_at); // newest
              })
              .map(p => ({
                id: p.id,
                name: p.name,
                description: p.description,
                created_at: p.created_at,
                artist_id: p.artist_id,
                files: filesByProject.get(p.id) || { contract: [], split_sheet: [], royalty_statement: [], other: [] },
                tasks: tasksByProject.get(p.id) || [],
              })),
          }))
          .sort((a, b) => a.artist.name.localeCompare(b.artist.name)); // alphabetical

        // Build letter groups for alphabetical legend
        const letterMap = new Map<string, ArtistGroup[]>();
        for (const ag of artists) {
          const letter = ag.artist.name.charAt(0).toUpperCase();
          if (!letterMap.has(letter)) letterMap.set(letter, []);
          letterMap.get(letter)!.push(ag);
        }
        const letterGroups = Array.from(letterMap.entries())
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([letter, artists]) => ({ letter, artists }));

        return {
          year,
          artists,
          letterGroups,
          totalProjects: artists.reduce((sum, a) => sum + a.projects.length, 0),
        };
      });
  }, [projectsQuery.data, filters, filesByProject, tasksByProject, artistMap]);

  return {
    years,
    allArtists: artistsQuery.data || [],
    allFiles: filesQuery.data || [],
    isLoading: artistsQuery.isLoading || projectsQuery.isLoading || filesQuery.isLoading || tasksQuery.isLoading,
    refetchFiles: filesQuery.refetch,
  };
}
