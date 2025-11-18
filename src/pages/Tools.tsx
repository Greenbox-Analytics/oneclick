import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Music, AlertCircle } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

//This is temporary placeholder data. TODO: Later to be replaced with a backend API
const mockArtists = [
  { id: 1, name: "Luna Rivers", hasContract: true },
  { id: 2, name: "The Echoes", hasContract: true },
  { id: 3, name: "DJ Neon", hasContract: false },
];

const Tools = () => {
  const navigate = useNavigate();
  //State tracks artist user has selected and errors
  const [selectedArtists, setSelectedArtists] = useState<number[]>([]);
  const [error, setError] = useState<string>("");

  //This lets us select/deselect a single artist (only one artist can be selected at a time)
  const handleArtistToggle = (artistId: number) => {
    setSelectedArtists(prev =>
      prev.includes(artistId)
        ? [] // Deselect if already selected
        : [artistId] // Select only this artist (replaces any previous selection)
    );
  };

  //Navigate to document upload page for the selected artist
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
            <h1 className="text-2xl font-bold text-foreground">Msanii AI</h1>
          </div>
          <Button variant="outline" onClick={() => navigate("/dashboard")}>
            Back to Dashboard
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
              {mockArtists.map((artist) => (
                <div key={artist.id} className="flex items-center p-4 border border-border rounded-lg hover:bg-secondary/50 transition-colors">
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
              ))}

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

export default Tools;
