import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Music, Calculator, AlertCircle } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

const mockArtists = [
  { id: 1, name: "Luna Rivers", hasContract: true },
  { id: 2, name: "The Echoes", hasContract: true },
  { id: 3, name: "DJ Neon", hasContract: false },
];

const Tools = () => {
  const navigate = useNavigate();
  const [selectedArtists, setSelectedArtists] = useState<number[]>([]);
  const [royaltyResults, setRoyaltyResults] = useState<any>(null);
  const [error, setError] = useState<string>("");

  const handleArtistToggle = (artistId: number) => {
    setSelectedArtists(prev =>
      prev.includes(artistId)
        ? prev.filter(id => id !== artistId)
        : [...prev, artistId]
    );
  };

  const handleCalculateRoyalties = () => {
    setError("");
    setRoyaltyResults(null);

    if (selectedArtists.length === 0) {
      setError("Please select at least one artist");
      return;
    }

    const artistsWithoutContract = selectedArtists.filter(id => {
      const artist = mockArtists.find(a => a.id === id);
      return artist && !artist.hasContract;
    });

    if (artistsWithoutContract.length > 0) {
      const artistNames = artistsWithoutContract
        .map(id => mockArtists.find(a => a.id === id)?.name)
        .join(", ");
      setError(`Cannot calculate royalties. Missing contracts for: ${artistNames}`);
      return;
    }

    // TODO: Call backend royalty split API
    // Mock result
    setRoyaltyResults({
      splits: selectedArtists.map(id => ({
        artist: mockArtists.find(a => a.id === id)?.name,
        percentage: (100 / selectedArtists.length).toFixed(2),
        amount: `$${(1000 / selectedArtists.length).toFixed(2)}`,
      })),
      total: "$1,000.00",
    });
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
                <div key={artist.id} className="flex items-center justify-between p-4 border border-border rounded-lg hover:bg-secondary/50 transition-colors">
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
                  <div className="flex items-center gap-2">
                    {artist.hasContract ? (
                      <span className="text-xs text-success bg-success/10 px-2 py-1 rounded-full">
                        Contract ✓
                      </span>
                    ) : (
                      <span className="text-xs text-destructive bg-destructive/10 px-2 py-1 rounded-full">
                        No Contract
                      </span>
                    )}
                  </div>
                </div>
              ))}

              <Button
                onClick={handleCalculateRoyalties}
                className="w-full"
                disabled={selectedArtists.length === 0}
              >
                <Calculator className="w-4 h-4 mr-2" />
                OneClick
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

          {royaltyResults && (
            <Card>
              <CardHeader>
                <CardTitle>Royalty Split Results</CardTitle>
                <CardDescription>Total Amount: {royaltyResults.total}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {royaltyResults.splits.map((split: any, index: number) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-4 bg-secondary/50 rounded-lg"
                    >
                      <span className="font-medium text-foreground">{split.artist}</span>
                      <div className="text-right">
                        <div className="font-bold text-primary">{split.amount}</div>
                        <div className="text-sm text-muted-foreground">{split.percentage}%</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
};

export default Tools;
