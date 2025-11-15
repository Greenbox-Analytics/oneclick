import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Music, Plus, Search, FileText } from "lucide-react";
import { useNavigate } from "react-router-dom";

// Mock data
const mockArtists = [
  { id: 1, name: "Luna Rivers", genre: "Electronic", hasContract: true, avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=Luna" },
  { id: 2, name: "The Echoes", genre: "Indie Rock", hasContract: true, avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=Echoes" },
  { id: 3, name: "DJ Neon", genre: "EDM", hasContract: false, avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=Neon" },
];

const Artists = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");

  const filteredArtists = mockArtists.filter(artist =>
    artist.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Artist Manager</h1>
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

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredArtists.map((artist) => (
            <Card key={artist.id} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between mb-4">
                  <Avatar className="w-16 h-16">
                    <AvatarImage src={artist.avatar} alt={artist.name} />
                    <AvatarFallback className="bg-primary text-primary-foreground text-xl">
                      {artist.name.charAt(0)}
                    </AvatarFallback>
                  </Avatar>
                  {artist.hasContract && (
                    <div className="flex items-center gap-1 text-xs text-success bg-success/10 px-2 py-1 rounded-full">
                      <FileText className="w-3 h-3" />
                      Contract
                    </div>
                  )}
                </div>
                <CardTitle>{artist.name}</CardTitle>
                <CardDescription>{artist.genre}</CardDescription>
              </CardHeader>
              <CardContent>
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={() => navigate(`/artists/${artist.id}`)}
                >
                  View Profile
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>

        {filteredArtists.length === 0 && (
          <div className="text-center py-12">
            <p className="text-muted-foreground">No artists found</p>
          </div>
        )}
      </main>
    </div>
  );
};

export default Artists;
