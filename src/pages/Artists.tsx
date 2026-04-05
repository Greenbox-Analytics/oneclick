import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Music, Plus, Search, FileText, Trash2, CheckCircle, ArrowLeft } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";

interface Artist {
  id: string;
  name: string;
  genres: string[];
  has_contract: boolean;
  avatar_url: string | null;
  verified?: boolean;
  teamcard?: {
    display_name?: string;
    avatar_url?: string;
    bio?: string;
  };
}

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

const Artists = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { user } = useAuth();
  const [searchQuery, setSearchQuery] = useState("");
  const [artistToDelete, setArtistToDelete] = useState<Artist | null>(null);

  // Fetch artists with TeamCard overlay for verified artists
  const { data: artists = [], isLoading, refetch } = useQuery<Artist[]>({
    queryKey: ["artists-with-teamcards", user?.id],
    queryFn: async () => {
      if (!user?.id) return [];
      // Try batch overlay endpoint first, fall back to direct Supabase query
      try {
        const res = await fetch(`${API_URL}/registry/artists/with-teamcards?user_id=${user.id}`);
        if (res.ok) {
          const data = await res.json();
          return data.artists || [];
        }
      } catch {
        // Fall back to direct Supabase query
      }
      const { data, error } = await supabase
        .from('artists')
        .select('*')
        .order('created_at', { ascending: false });
      if (error) {
        console.error('Error fetching artists:', error);
        return [];
      }
      return data || [];
    },
    enabled: !!user?.id,
  });

  // Tool walkthrough
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(TOOL_CONFIGS.artists, {
    onComplete: () => markToolCompleted("artists"),
  });

  useEffect(() => {
    if (!onboardingLoading && !statuses.artists && walkthrough.phase === "idle") {
      const timer = setTimeout(() => walkthrough.startModal(), 500);
      return () => clearTimeout(timer);
    }
  }, [onboardingLoading, statuses.artists]);

  const handleDeleteArtist = async () => {
    if (!artistToDelete) return;

    const { error } = await supabase
      .from('artists')
      .delete()
      .eq('id', artistToDelete.id);

    if (error) {
      toast({
        title: "Error",
        description: "Failed to delete artist. Please try again.",
        variant: "destructive",
      });
    } else {
      refetch();
      toast({
        title: "Success",
        description: `${artistToDelete.name} has been deleted.`,
      });
    }
    setArtistToDelete(null);
  };

  const filteredArtists = artists.filter(artist =>
    artist.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              className="text-muted-foreground hover:text-foreground"
              onClick={() => navigate(-1)}
            >
              <ArrowLeft className="w-4 h-4 mr-1" /> Back
            </Button>
            <div className="w-px h-6 bg-border" />
            <div
              className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
              onClick={() => navigate("/dashboard")}
            >
              <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center p-1.5">
                <Music className="w-full h-full text-primary-foreground" />
              </div>
              <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <ToolHelpButton onClick={walkthrough.replay} />
            <Button variant="outline" onClick={() => navigate("/dashboard")}>
              Back to Dashboard
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-3xl font-bold text-foreground mb-2">Artist Profiles</h2>
            <p className="text-muted-foreground">Manage your artist roster</p>
          </div>
          <Button data-walkthrough="artists-add" onClick={() => navigate("/artists/new")}>
            <Plus className="w-4 h-4 mr-2" />
            Add Artist
          </Button>
        </div>

        <div className="mb-6">
          <div className="relative" data-walkthrough="artists-search">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
            <Input
              placeholder="Search artists..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        {isLoading ? (
          <div className="text-center py-12">
            <p className="text-muted-foreground">Loading artists...</p>
          </div>
        ) : (
          <>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {filteredArtists.map((artist, index) => {
                const cardDisplayName = artist.teamcard?.display_name || artist.name;
                const cardDisplayAvatar = artist.teamcard?.avatar_url || artist.avatar_url;
                const cardIsVerified = artist.verified === true;

                return (
                <Card key={artist.id} className="hover:shadow-lg transition-shadow" data-walkthrough={index === 0 ? "artists-card" : undefined}>
                  <CardHeader>
                    <div className="flex items-start justify-between mb-4">
                      <Avatar className="w-16 h-16">
                        <AvatarImage src={cardDisplayAvatar || undefined} alt={cardDisplayName} />
                        <AvatarFallback className="bg-primary text-primary-foreground text-xl">
                          {cardDisplayName.charAt(0)}
                        </AvatarFallback>
                      </Avatar>
                      <div className="flex flex-col items-end gap-1">
                        {cardIsVerified && (
                          <Badge className="bg-green-100 text-green-800 text-xs flex items-center gap-1">
                            <CheckCircle className="w-3 h-3" /> Verified
                          </Badge>
                        )}
                        {artist.has_contract && (
                          <div className="flex items-center gap-1 text-xs text-success bg-success/10 px-2 py-1 rounded-full">
                            <FileText className="w-3 h-3" />
                            Contract
                          </div>
                        )}
                      </div>
                    </div>
                    <CardTitle>{cardDisplayName}</CardTitle>
                    <CardDescription>{artist.genres.join(', ')}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <Button
                      variant="outline"
                      className="w-full"
                      onClick={() => navigate(`/artists/${artist.id}`)}
                    >
                      View Profile
                    </Button>
                    <Button
                      variant="outline"
                      className="w-full text-destructive hover:bg-destructive hover:text-destructive-foreground"
                      onClick={(e) => {
                        e.stopPropagation();
                        setArtistToDelete(artist);
                      }}
                    >
                      <Trash2 className="w-4 h-4 mr-2" />
                      Delete Artist
                    </Button>
                  </CardContent>
                </Card>
                );
              })}
            </div>

            {filteredArtists.length === 0 && !isLoading && (
              <div className="text-center py-12">
                <p className="text-muted-foreground">No artists found</p>
              </div>
            )}
          </>
        )}
        <ToolIntroModal
          config={TOOL_CONFIGS.artists}
          isOpen={walkthrough.phase === "modal"}
          onStartTour={walkthrough.startSpotlight}
          onSkip={walkthrough.skip}
        />
        <WalkthroughProvider
          isActive={walkthrough.phase === "spotlight"}
          currentStep={walkthrough.currentStep}
          currentStepIndex={walkthrough.visibleStepIndex}
          totalSteps={walkthrough.totalSteps}
          onNext={walkthrough.next}
          onSkip={walkthrough.skip}
        />
      </main>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={!!artistToDelete} onOpenChange={() => setArtistToDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete <strong>{artistToDelete?.name}</strong> and all associated data. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteArtist}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete Artist
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default Artists;
