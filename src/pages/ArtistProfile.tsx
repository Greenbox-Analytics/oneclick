import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Music, Upload, FileText, ArrowLeft, Camera, Edit, Save, X } from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";

const ArtistProfile = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const [hasContract, setHasContract] = useState(id === "1" || id === "2");
  const [isEditMode, setIsEditMode] = useState(false);

  // Mock data
  const initialArtist = {
    name: id === "1" ? "Luna Rivers" : id === "2" ? "The Echoes" : "DJ Neon",
    email: "artist@example.com",
    bio: id === "1" 
      ? "Electronic artist pushing boundaries with ambient soundscapes and experimental beats."
      : id === "2"
      ? "Indie rock band known for raw energy and authentic storytelling."
      : "EDM producer bringing high-energy performances to festivals worldwide.",
    genres: id === "1" ? ["Electronic", "Ambient", "Experimental"] : id === "2" ? ["Indie Rock", "Alternative"] : ["EDM", "House"],
    avatar: id === "1" 
      ? "https://api.dicebear.com/7.x/avataaars/svg?seed=Luna"
      : id === "2" 
      ? "https://api.dicebear.com/7.x/avataaars/svg?seed=Echoes"
      : "https://api.dicebear.com/7.x/avataaars/svg?seed=Neon",
    social: {
      instagram: "@artist_instagram",
      tiktok: "@artist_tiktok",
      youtube: "@artist_youtube",
    },
    dsp: {
      spotify: "spotify:artist:...",
      appleMusic: "https://music.apple.com/artist/...",
      soundcloud: "soundcloud.com/artist",
    },
    additional: {
      epk: "https://epk.example.com",
      pressKit: "https://press.example.com",
      linktree: "https://linktr.ee/artist",
    },
  };

  const [formData, setFormData] = useState(initialArtist);

  const handleContractUpload = () => {
    // TODO: Add backend contract upload logic
    setHasContract(true);
  };

  const handleSave = () => {
    // TODO: Add backend save logic
    setIsEditMode(false);
  };

  const handleCancel = () => {
    setFormData(initialArtist);
    setIsEditMode(false);
  };

  const updateField = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const updateNestedField = (parent: string, field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [parent]: { ...prev[parent as keyof typeof prev] as any, [field]: value }
    }));
  };

  const addGenre = (genre: string) => {
    if (genre && !formData.genres.includes(genre)) {
      setFormData(prev => ({ ...prev, genres: [...prev.genres, genre] }));
    }
  };

  const removeGenre = (genre: string) => {
    setFormData(prev => ({ ...prev, genres: prev.genres.filter(g => g !== genre) }));
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
          <div className="flex gap-2">
            <Button
              variant={isEditMode ? "outline" : "default"}
              onClick={() => isEditMode ? handleCancel() : setIsEditMode(true)}
            >
              {isEditMode ? <><X className="w-4 h-4 mr-2" />Cancel</> : <><Edit className="w-4 h-4 mr-2" />Edit Profile</>}
            </Button>
            {isEditMode && (
              <Button onClick={handleSave}>
                <Save className="w-4 h-4 mr-2" />
                Save Changes
              </Button>
            )}
            <Button variant="outline" onClick={() => navigate("/artists")}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Artists
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="mb-8 flex items-start gap-6">
          <div className="relative group">
            <Avatar className="w-24 h-24">
              <AvatarImage src={formData.avatar} alt={formData.name} />
              <AvatarFallback className="bg-primary text-primary-foreground text-3xl">
                {formData.name.charAt(0)}
              </AvatarFallback>
            </Avatar>
            {isEditMode && (
              <button className="absolute inset-0 bg-background/80 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                <Camera className="w-6 h-6 text-foreground" />
              </button>
            )}
          </div>
          <div>
            <h2 className="text-3xl font-bold text-foreground mb-2">{formData.name}</h2>
            <div className="flex flex-wrap gap-2">
              {formData.genres.map(genre => (
                <Badge key={genre} variant="secondary">{genre}</Badge>
              ))}
            </div>
          </div>
        </div>

        <div className="grid gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Basic Information</CardTitle>
              <CardDescription>Artist details and biography</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">Artist Name</Label>
                {isEditMode ? (
                  <Input 
                    id="name" 
                    value={formData.name}
                    onChange={(e) => updateField('name', e.target.value)}
                  />
                ) : (
                  <p className="text-foreground p-2">{formData.name}</p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                {isEditMode ? (
                  <Input 
                    id="email" 
                    type="email"
                    value={formData.email}
                    onChange={(e) => updateField('email', e.target.value)}
                  />
                ) : (
                  <p className="text-foreground p-2">{formData.email}</p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="bio">Bio</Label>
                {isEditMode ? (
                  <Textarea 
                    id="bio"
                    value={formData.bio}
                    onChange={(e) => updateField('bio', e.target.value)}
                    rows={4}
                  />
                ) : (
                  <p className="text-foreground p-2">{formData.bio}</p>
                )}
              </div>
              <div className="space-y-2">
                <Label>Genres</Label>
                {isEditMode ? (
                  <div className="space-y-2">
                    <div className="flex flex-wrap gap-2">
                      {formData.genres.map(genre => (
                        <Badge key={genre} variant="secondary" className="cursor-pointer" onClick={() => removeGenre(genre)}>
                          {genre} <X className="w-3 h-3 ml-1" />
                        </Badge>
                      ))}
                    </div>
                    <Input 
                      placeholder="Add genre and press Enter"
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          addGenre(e.currentTarget.value);
                          e.currentTarget.value = '';
                        }
                      }}
                    />
                  </div>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {formData.genres.map(genre => (
                      <Badge key={genre} variant="secondary">{genre}</Badge>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Social Media</CardTitle>
              <CardDescription>Instagram, TikTok, YouTube links</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="instagram">Instagram</Label>
                {isEditMode ? (
                  <Input 
                    id="instagram"
                    value={formData.social.instagram}
                    onChange={(e) => updateNestedField('social', 'instagram', e.target.value)}
                    placeholder="@username"
                  />
                ) : (
                  <p className="text-foreground p-2">{formData.social.instagram}</p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="tiktok">TikTok</Label>
                {isEditMode ? (
                  <Input 
                    id="tiktok"
                    value={formData.social.tiktok}
                    onChange={(e) => updateNestedField('social', 'tiktok', e.target.value)}
                    placeholder="@username"
                  />
                ) : (
                  <p className="text-foreground p-2">{formData.social.tiktok}</p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="youtube">YouTube</Label>
                {isEditMode ? (
                  <Input 
                    id="youtube"
                    value={formData.social.youtube}
                    onChange={(e) => updateNestedField('social', 'youtube', e.target.value)}
                    placeholder="@channel"
                  />
                ) : (
                  <p className="text-foreground p-2">{formData.social.youtube}</p>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Streaming Platforms</CardTitle>
              <CardDescription>Spotify, Apple Music, SoundCloud links</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="spotify">Spotify</Label>
                {isEditMode ? (
                  <Input 
                    id="spotify"
                    value={formData.dsp.spotify}
                    onChange={(e) => updateNestedField('dsp', 'spotify', e.target.value)}
                    placeholder="spotify:artist:..."
                  />
                ) : (
                  <p className="text-foreground p-2">{formData.dsp.spotify}</p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="appleMusic">Apple Music</Label>
                {isEditMode ? (
                  <Input 
                    id="appleMusic"
                    value={formData.dsp.appleMusic}
                    onChange={(e) => updateNestedField('dsp', 'appleMusic', e.target.value)}
                    placeholder="https://music.apple.com/..."
                  />
                ) : (
                  <p className="text-foreground p-2">{formData.dsp.appleMusic}</p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="soundcloud">SoundCloud</Label>
                {isEditMode ? (
                  <Input 
                    id="soundcloud"
                    value={formData.dsp.soundcloud}
                    onChange={(e) => updateNestedField('dsp', 'soundcloud', e.target.value)}
                    placeholder="soundcloud.com/artist"
                  />
                ) : (
                  <p className="text-foreground p-2">{formData.dsp.soundcloud}</p>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Additional Links</CardTitle>
              <CardDescription>EPK, press kit, Linktree, and other links</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="epk">EPK</Label>
                {isEditMode ? (
                  <Input 
                    id="epk"
                    value={formData.additional.epk}
                    onChange={(e) => updateNestedField('additional', 'epk', e.target.value)}
                    placeholder="https://epk.example.com"
                  />
                ) : (
                  <p className="text-foreground p-2">{formData.additional.epk}</p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="pressKit">Press Kit</Label>
                {isEditMode ? (
                  <Input 
                    id="pressKit"
                    value={formData.additional.pressKit}
                    onChange={(e) => updateNestedField('additional', 'pressKit', e.target.value)}
                    placeholder="https://press.example.com"
                  />
                ) : (
                  <p className="text-foreground p-2">{formData.additional.pressKit}</p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="linktree">Linktree</Label>
                {isEditMode ? (
                  <Input 
                    id="linktree"
                    value={formData.additional.linktree}
                    onChange={(e) => updateNestedField('additional', 'linktree', e.target.value)}
                    placeholder="https://linktr.ee/artist"
                  />
                ) : (
                  <p className="text-foreground p-2">{formData.additional.linktree}</p>
                )}
              </div>
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
