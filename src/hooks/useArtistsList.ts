import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

export interface ArtistOption {
  id: string;
  name: string;
  avatar?: string;
}

export function useArtistsList() {
  const { user } = useAuth();

  const query = useQuery<ArtistOption[]>({
    queryKey: ["artists-list", user?.id],
    queryFn: async () => {
      if (!user?.id) return [];
      const res = await fetch(`${API_URL}/artists?user_id=${user.id}`);
      if (!res.ok) return [];
      const data = await res.json();
      // Backend returns array of artist objects
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

  return {
    artists: query.data || [],
    isLoading: query.isLoading,
  };
}
