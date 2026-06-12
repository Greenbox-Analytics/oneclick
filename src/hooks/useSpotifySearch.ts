import { useQuery } from "@tanstack/react-query";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import { useAuth } from "@/contexts/AuthContext";

export interface SpotifyTrack {
  id: string;
  title: string | null;
  artist: string;
  album: string | null;
  release_date: string | null;
  year: number | null;
  duration_ms: number | null;
  explicit: boolean;
  popularity: number;
  isrc: string | null;
  upc: string | null;
  label: string | null;
  cover_url: string | null;
  spotify_url: string | null;
}

interface SearchResponse {
  tracks: SpotifyTrack[];
}

/**
 * Search Spotify for tracks. The query is sent only when non-empty and
 * `enabled` is true (the wizard waits until the user clicks "Search").
 */
export function useSpotifySearch(query: string, enabled: boolean) {
  const { user } = useAuth();
  const q = query.trim();
  return useQuery<SpotifyTrack[]>({
    queryKey: ["spotify-search", q],
    queryFn: async () => {
      const data = await apiFetch<SearchResponse>(
        `${API_URL}/integrations/spotify/search?q=${encodeURIComponent(q)}&limit=10`
      );
      return data.tracks || [];
    },
    enabled: !!user?.id && !!q && enabled,
    staleTime: 60_000,
  });
}

/**
 * Fetch full metadata for a single track id. Used by the work editor's
 * "Pull from Spotify" button when the work has a stored Spotify URL.
 */
export function useSpotifyTrack(trackId: string | null | undefined) {
  const { user } = useAuth();
  return useQuery<SpotifyTrack>({
    queryKey: ["spotify-track", trackId],
    queryFn: async () =>
      apiFetch<SpotifyTrack>(`${API_URL}/integrations/spotify/tracks/${trackId}`),
    enabled: !!user?.id && !!trackId,
    staleTime: 5 * 60_000,
  });
}

/**
 * Extract the track id from any open.spotify.com/track/<id>… URL.
 * Returns null when the input doesn't look like a Spotify track URL.
 */
export function spotifyTrackIdFromUrl(url: string | null | undefined): string | null {
  if (!url) return null;
  const m = url.match(/spotify\.com\/track\/([a-zA-Z0-9]+)/);
  return m ? m[1] : null;
}
