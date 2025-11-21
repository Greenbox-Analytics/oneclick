import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Music, Plus, Search, FileText, Trash2 } from "lucide-react";
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

interface Artist {
  id: string;
  name: string;
  genres: string[];
  has_contract: boolean;
  avatar_url: string | null;
}

const Artists = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  const [artists, setArtists] = useState<Artist[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [artistToDelete, setArtistToDelete] = useState<Artist | null>(null);

  useEffect(() => {
    const fetchArtists = async () => {
      const { data, error } = await supabase
        .from('artists')
        .select('*')
        .order('created_at', { ascending: false });

      if (!error && data) {
        setArtists(data);
      }
      setIsLoading(false);
    };

    fetchArtists();
  }, []);

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
      setArtists(artists.filter(a => a.id !== artistToDelete.id));
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
          <div 
            className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate("/")}
          >
            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center p-1.5">
              <img src="/iconspear.png" alt="Msanii AI" className="w-full h-full object-contain" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii AI</h1>
          </div>
          <Button variant="outline" onClick={() => navigate("/dashboard")}>
            Back to Dashboard
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-3xl font-bold text-foreground mb-2">Artist Profiles</h2>
            <p className="text-muted-foreground">Manage your artist roster</p>
          </div>
          <Button onClick={() => navigate("/artists/new")}>
            <Plus className="w-4 h-4 mr-2" />
            Add Artist
          </Button>
        </div>

        <div className="mb-6">
          <div className="relative">
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
              {filteredArtists.map((artist) => (
                <Card key={artist.id} className="hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <div className="flex items-start justify-between mb-4">
                      <Avatar className="w-16 h-16">
                        <AvatarImage src={artist.avatar_url || undefined} alt={artist.name} />
                        <AvatarFallback className="bg-primary text-primary-foreground text-xl">
                          {artist.name.charAt(0)}
                        </AvatarFallback>
                      </Avatar>
                      {artist.has_contract && (
                        <div className="flex items-center gap-1 text-xs text-success bg-success/10 px-2 py-1 rounded-full">
                          <FileText className="w-3 h-3" />
                          Contract
                        </div>
                      )}
                    </div>
                    <CardTitle>{artist.name}</CardTitle>
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
              ))}
            </div>

            {filteredArtists.length === 0 && !isLoading && (
              <div className="text-center py-12">
                <p className="text-muted-foreground">No artists found</p>
              </div>
            )}
          </>
        )}
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
