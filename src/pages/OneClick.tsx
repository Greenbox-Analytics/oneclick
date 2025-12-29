import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Music, AlertCircle, ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

// Backend API URL
const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

interface Artist {
  id: string; // UUID in database
  name: string;
  has_contract: boolean;
  // Add other fields as needed
}

const OneClick = () => {
  const navigate = useNavigate();
  
  // State for fetched artists
  const [artists, setArtists] = useState<Artist[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // State tracks artist user has selected and errors
  // Changed to string[] because Supabase IDs are UUIDs (strings)
  const [selectedArtists, setSelectedArtists] = useState<string[]>([]);
  const [error, setError] = useState<string>("");

  // Fetch artists from backend on component mount
  useEffect(() => {
    fetch(`${API_URL}/artists`)
      .then((res) => {
        if (!res.ok) {
          throw new Error("Failed to fetch artists");
        }
        return res.json();
      })
      .then((data) => {
        setArtists(data);
        setIsLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching artists:", err);
        setError("Failed to load artists. Please check your backend connection.");
        setIsLoading(false);
      });
  }, []);

  // This lets us select/deselect a single artist (only one artist can be selected at a time)
  const handleArtistToggle = (artistId: string) => {
    setSelectedArtists((prev) =>
      prev.includes(artistId)
        ? [] // Deselect if already selected
        : [artistId] // Select only this artist (replaces any previous selection)
    );
  };

  // Navigate to document selection page for the selected artist
  const handleContinue = () => {
    setError("");

    if (selectedArtists.length === 0) {
      setError("Please select at least one artist");
      return;
    }

    const selectedArtistId = selectedArtists[0];
    navigate(`/oneclick/${selectedArtistId}/documents`);
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
          </div>
          <Button variant="outline" onClick={() => navigate("/tools")}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Tools
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-foreground mb-2">OneClick</h2>
          <p className="text-muted-foreground">Calculate royalty splits across your artists</p>
        </div>

        <div className="grid gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Select Artists</CardTitle>
              <CardDescription>Choose artists to include in the royalty calculation</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {isLoading ? (
                <div className="text-center py-4 text-muted-foreground">Loading artists...</div>
              ) : artists.length === 0 ? (
                <div className="text-center py-4 text-muted-foreground">
                  No artists found. Please add an artist first.
                </div>
              ) : (
                artists.map((artist) => (
                  <div
                    key={artist.id}
                    className="flex items-center p-4 border border-border rounded-lg hover:bg-secondary/50 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <Checkbox
                        id={`artist-${artist.id}`}
                        checked={selectedArtists.includes(artist.id)}
                        onCheckedChange={() => handleArtistToggle(artist.id)}
                      />
                      <label
                        htmlFor={`artist-${artist.id}`}
                        className="text-foreground font-medium cursor-pointer"
                      >
                        {artist.name}
                      </label>
                    </div>
                  </div>
                ))
              )}

              <Button
                onClick={handleContinue}
                className="w-full"
                disabled={selectedArtists.length === 0}
              >
                Continue
              </Button>
            </CardContent>
          </Card>

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>
      </main>
    </div>
  );
};

export default OneClick;
