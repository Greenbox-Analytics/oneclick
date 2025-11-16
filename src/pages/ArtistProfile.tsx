import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Music, Upload, FileText, ArrowLeft, Camera } from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";

const ArtistProfile = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const [hasContract, setHasContract] = useState(id === "1" || id === "2");

  // Mock data
  const artist = {
    name: id === "1" ? "Luna Rivers" : id === "2" ? "The Echoes" : "DJ Neon",
    genre: id === "1" ? "Electronic" : id === "2" ? "Indie Rock" : "EDM",
    email: "artist@example.com",
    avatar: id === "1" 
      ? "https://api.dicebear.com/7.x/avataaars/svg?seed=Luna"
      : id === "2" 
      ? "https://api.dicebear.com/7.x/avataaars/svg?seed=Echoes"
      : "https://api.dicebear.com/7.x/avataaars/svg?seed=Neon",
  };

  const handleContractUpload = () => {
    // TODO: Add backend contract upload logic
    setHasContract(true);
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
          <Button variant="outline" onClick={() => navigate("/artists")}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Artists
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="mb-8 flex items-start gap-6">
          <div className="relative group">
            <Avatar className="w-24 h-24">
              <AvatarImage src={artist.avatar} alt={artist.name} />
              <AvatarFallback className="bg-primary text-primary-foreground text-3xl">
                {artist.name.charAt(0)}
              </AvatarFallback>
            </Avatar>
            <button className="absolute inset-0 bg-background/80 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
              <Camera className="w-6 h-6 text-foreground" />
            </button>
          </div>
          <div>
            <h2 className="text-3xl font-bold text-foreground mb-2">{artist.name}</h2>
            <p className="text-muted-foreground">{artist.genre}</p>
          </div>
        </div>

        <div className="grid gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Artist Information</CardTitle>
              <CardDescription>Manage basic artist details</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">Artist Name</Label>
                <Input id="name" defaultValue={artist.name} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="genre">Genre</Label>
                <Input id="genre" defaultValue={artist.genre} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input id="email" type="email" defaultValue={artist.email} />
              </div>
              <Button>Save Changes</Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Contract Management</CardTitle>
              <CardDescription>Upload and manage artist contracts</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {hasContract ? (
                <div className="space-y-4">
                  <div className="flex items-center gap-3 p-4 border border-border rounded-lg bg-secondary/50">
                    <FileText className="w-8 h-8 text-primary" />
                    <div className="flex-1">
                      <p className="font-medium text-foreground">Artist Contract.pdf</p>
                      <p className="text-sm text-muted-foreground">Uploaded 3 months ago</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <Button variant="outline">View Contract</Button>
                    <Button variant="outline" onClick={() => setHasContract(false)}>
                      Replace Contract
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="border-2 border-dashed border-border rounded-lg p-8 text-center">
                  <Upload className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-foreground font-medium mb-2">No contract uploaded</p>
                  <p className="text-muted-foreground mb-4 text-sm">
                    Upload a contract to enable royalty calculations
                  </p>
                  <Button onClick={handleContractUpload}>
                    <Upload className="w-4 h-4 mr-2" />
                    Upload Contract
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default ArtistProfile;
