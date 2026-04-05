import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Music, Plus, Search, FileText, Trash2, CheckCircle, ArrowLeft, BookOpen, Users } from "lucide-react";
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
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/docs")}
              title="Documentation"
              className="text-muted-foreground hover:text-foreground"
            >
              <BookOpen className="w-4 h-4" />
            </Button>
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
            <h2 className="text-3xl font-bold text-foreground mb-1">Artist Profiles</h2>
            <p className="text-muted-foreground mb-3">Manage your artist roster</p>
            <div className="h-0.5 w-20 bg-gradient-to-r from-primary to-primary/0 rounded-full" />
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
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <Card key={i} className="border border-border">
                <div className="h-0.5 bg-muted" />
                <CardContent className="p-5">
                  <div className="flex items-start gap-4">
                    <div className="w-14 h-14 rounded-full bg-muted animate-pulse shrink-0" />
                    <div className="flex-1 space-y-2">
                      <div className="h-5 bg-muted rounded animate-pulse w-2/3" />
                      <div className="h-4 bg-muted rounded animate-pulse w-1/2" />
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {filteredArtists.map((artist, index) => {
                const cardDisplayName = artist.teamcard?.display_name || artist.name;
                const cardDisplayAvatar = artist.teamcard?.avatar_url || artist.avatar_url;
                const cardIsVerified = artist.verified === true;

                return (
                <Card
                  key={artist.id}
                  className="group cursor-pointer border border-border hover:border-primary/30 hover:shadow-md transition-all overflow-hidden"
                  onClick={() => navigate(`/artists/${artist.id}`)}
                  data-walkthrough={index === 0 ? "artists-card" : undefined}
                >
                  <div className="h-0.5 bg-primary/40 group-hover:bg-primary transition-colors" />
                  <CardContent className="p-5">
                    <div className="flex items-start gap-4">
                      <Avatar className="w-14 h-14 ring-2 ring-primary/10">
                        <AvatarImage src={cardDisplayAvatar || undefined} alt={cardDisplayName} />
                        <AvatarFallback className="bg-primary/10 text-primary text-lg font-semibold">
                          {cardDisplayName.charAt(0)}
                        </AvatarFallback>
                      </Avatar>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="font-semibold text-foreground truncate">{cardDisplayName}</h3>
                          {cardIsVerified && (
                            <CheckCircle className="w-4 h-4 text-emerald-500 shrink-0" />
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground mb-2 truncate">
                          {artist.genres.length > 0 ? artist.genres.join(' \u00b7 ') : 'No genres set'}
                        </p>
                        <div className="flex items-center gap-2">
                          {artist.has_contract && (
                            <Badge variant="outline" className="text-xs gap-1 text-primary border-primary/30">
                              <FileText className="w-3 h-3" /> Contract
                            </Badge>
                          )}
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive shrink-0"
                        onClick={(e) => {
                          e.stopPropagation();
                          setArtistToDelete(artist);
                        }}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
                );
              })}
            </div>

            {filteredArtists.length === 0 && !isLoading && (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <Users className="w-12 h-12 text-muted-foreground/30 mb-3" />
                <p className="text-muted-foreground font-medium">No artists found</p>
                <p className="text-sm text-muted-foreground/60 mt-1">
                  {searchQuery ? 'Try a different search term' : 'Add your first artist to get started'}
                </p>
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
