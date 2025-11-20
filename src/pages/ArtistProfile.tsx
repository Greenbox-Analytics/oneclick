import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Music, Upload, FileText, ArrowLeft, Camera, Edit, Save, X, Instagram, Youtube, MessageCircle, Mic2, Link as LinkIcon, Users, Music2 } from "lucide-react";
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
    customLinks: id === "1" ? [
      { id: "1", label: "Website", url: "https://{sample_personal_website}" },
      { id: "2", label: "Merch Store", url: "https://{sample_shop}" }
    ] : [],
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

  const addCustomLink = () => {
    const newLink = {
      id: Date.now().toString(),
      label: "",
      url: ""
    };
    setFormData(prev => ({
      ...prev,
      customLinks: [...prev.customLinks, newLink]
    }));
  };

  const updateCustomLink = (linkId: string, field: 'label' | 'url', value: string) => {
    setFormData(prev => ({
      ...prev,
      customLinks: prev.customLinks.map(link =>
        link.id === linkId ? { ...link, [field]: value } : link
      )
    }));
  };

  const removeCustomLink = (linkId: string) => {
    setFormData(prev => ({
      ...prev,
      customLinks: prev.customLinks.filter(link => link.id !== linkId)
    }));
  };

  // Helper component for field containers
  const FieldContainer = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
    <div className={`p-4 rounded-lg border border-border bg-card/50 hover:bg-card transition-colors ${className}`}>
      {children}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/10">
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

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Profile Header */}
        <Card className="mb-6 border-2 shadow-lg">
          <CardContent className="pt-6">
            <div className="flex items-start gap-6">
              <div className="relative group">
                <Avatar className="w-28 h-28 ring-4 ring-primary/10">
                  <AvatarImage src={formData.avatar} alt={formData.name} />
                  <AvatarFallback className="bg-gradient-to-br from-primary to-primary/70 text-primary-foreground text-3xl">
                    {formData.name.charAt(0)}
                  </AvatarFallback>
                </Avatar>
                {isEditMode && (
                  <button className="absolute inset-0 bg-background/90 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all hover:scale-105">
                    <Camera className="w-7 h-7 text-primary" />
                  </button>
                )}
              </div>
              <div className="flex-1">
                <h2 className="text-4xl font-bold text-foreground mb-3 bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text">{formData.name}</h2>
                <div className="flex flex-wrap gap-2 mb-3">
                  {formData.genres.map(genre => (
                    <Badge key={genre} variant="secondary" className="px-3 py-1 text-sm">{genre}</Badge>
                  ))}
                </div>
                <p className="text-muted-foreground text-sm line-clamp-2">{formData.bio}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid gap-6">
          {/* Basic Information */}
          <Card className="border-2 shadow-md hover:shadow-lg transition-shadow">
            <CardHeader className="bg-gradient-to-r from-primary/5 to-transparent">
              <div className="flex items-center gap-2">
                <Users className="w-5 h-5 text-primary" />
                <CardTitle>Artist Details</CardTitle>
              </div>
              <CardDescription>Artist information and biography</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 pt-6">
              <FieldContainer>
                <Label htmlFor="name" className="text-sm font-semibold text-muted-foreground mb-2 block">Artist Name</Label>
                {isEditMode ? (
                  <Input 
                    id="name" 
                    value={formData.name}
                    onChange={(e) => updateField('name', e.target.value)}
                    className="bg-background border-2 focus:border-primary transition-colors"
                  />
                ) : (
                  <p className="text-foreground text-lg font-medium">{formData.name}</p>
                )}
              </FieldContainer>

              <FieldContainer>
                <Label htmlFor="email" className="text-sm font-semibold text-muted-foreground mb-2 block">Email Address</Label>
                {isEditMode ? (
                  <Input 
                    id="email" 
                    type="email"
                    value={formData.email}
                    onChange={(e) => updateField('email', e.target.value)}
                    className="bg-background border-2 focus:border-primary transition-colors"
                  />
                ) : (
                  <p className="text-foreground text-lg">{formData.email}</p>
                )}
              </FieldContainer>

              <FieldContainer>
                <Label htmlFor="bio" className="text-sm font-semibold text-muted-foreground mb-2 block">Biography</Label>
                {isEditMode ? (
                  <Textarea 
                    id="bio"
                    value={formData.bio}
                    onChange={(e) => updateField('bio', e.target.value)}
                    rows={4}
                    className="bg-background border-2 focus:border-primary transition-colors resize-none"
                  />
                ) : (
                  <p className="text-foreground leading-relaxed">{formData.bio}</p>
                )}
              </FieldContainer>

              <FieldContainer>
                <Label className="text-sm font-semibold text-muted-foreground mb-3 block">Genres</Label>
                {isEditMode ? (
                  <div className="space-y-3">
                    <div className="flex flex-wrap gap-2 min-h-[2rem]">
                      {formData.genres.map(genre => (
                        <Badge 
                          key={genre} 
                          variant="secondary" 
                          className="cursor-pointer hover:bg-destructive hover:text-destructive-foreground transition-colors px-3 py-1"
                          onClick={() => removeGenre(genre)}
                        >
                          {genre} <X className="w-3 h-3 ml-1" />
                        </Badge>
                      ))}
                    </div>
                    <Input 
                      placeholder="Type a genre and press Enter to add"
                      className="bg-background border-2 focus:border-primary transition-colors"
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
                      <Badge key={genre} variant="secondary" className="px-3 py-1">{genre}</Badge>
                    ))}
                  </div>
                )}
              </FieldContainer>
            </CardContent>
          </Card>

          {/* Social Media */}
          <Card className="border-2 shadow-md hover:shadow-lg transition-shadow">
            <CardHeader className="bg-gradient-to-r from-pink-500/5 to-transparent">
              <div className="flex items-center gap-2">
                <Instagram className="w-5 h-5 text-pink-500" />
                <CardTitle>Social Media</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-4 pt-6">
              <FieldContainer>
                <div className="flex items-center gap-2 mb-2">
                  <Instagram className="w-4 h-4 text-pink-500" />
                  <Label htmlFor="instagram" className="text-sm font-semibold text-muted-foreground">Instagram</Label>
                </div>
                {isEditMode ? (
                  <Input 
                    id="instagram"
                    value={formData.social.instagram}
                    onChange={(e) => updateNestedField('social', 'instagram', e.target.value)}
                    placeholder="@username"
                    className="bg-background border-2 focus:border-pink-500 transition-colors"
                  />
                ) : (
                  <p className="text-foreground text-lg">{formData.social.instagram}</p>
                )}
              </FieldContainer>

              <FieldContainer>
                <div className="flex items-center gap-2 mb-2">
                  <Music2 className="w-4 h-4 text-blue-500" />
                  <Label htmlFor="tiktok" className="text-sm font-semibold text-muted-foreground">TikTok</Label>
                </div>
                {isEditMode ? (
                  <Input 
                    id="tiktok"
                    value={formData.social.tiktok}
                    onChange={(e) => updateNestedField('social', 'tiktok', e.target.value)}
                    placeholder="@username"
                    className="bg-background border-2 focus:border-blue-500 transition-colors"
                  />
                ) : (
                  <p className="text-foreground text-lg">{formData.social.tiktok}</p>
                )}
              </FieldContainer>

              <FieldContainer>
                <div className="flex items-center gap-2 mb-2">
                  <Youtube className="w-4 h-4 text-red-500" />
                  <Label htmlFor="youtube" className="text-sm font-semibold text-muted-foreground">YouTube</Label>
                </div>
                {isEditMode ? (
                  <Input 
                    id="youtube"
                    value={formData.social.youtube}
                    onChange={(e) => updateNestedField('social', 'youtube', e.target.value)}
                    placeholder="@channel"
                    className="bg-background border-2 focus:border-red-500 transition-colors"
                  />
                ) : (
                  <p className="text-foreground text-lg">{formData.social.youtube}</p>
                )}
              </FieldContainer>
            </CardContent>
          </Card>

          {/* Streaming Platforms */}
          <Card className="border-2 shadow-md hover:shadow-lg transition-shadow">
            <CardHeader className="bg-gradient-to-r from-green-500/5 to-transparent">
              <div className="flex items-center gap-2">
                <Mic2 className="w-5 h-5 text-green-500" />
                <CardTitle>Streaming Platforms</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-4 pt-6">
              <FieldContainer>
                <div className="flex items-center gap-2 mb-2">
                  <img src="/public/spotify.svg" alt="Spotify" className="w-4 h-4" />
                  <Label htmlFor="spotify" className="text-sm font-semibold text-muted-foreground">Spotify</Label>
                </div>
                {isEditMode ? (
                  <Input 
                    id="spotify"
                    value={formData.dsp.spotify}
                    onChange={(e) => updateNestedField('dsp', 'spotify', e.target.value)}
                    placeholder=""
                    className="bg-background border-2 focus:border-green-500 transition-colors"
                  />
                ) : (
                  <p className="text-foreground text-lg font-mono text-sm">{formData.dsp.spotify}</p>
                )}
              </FieldContainer>

              <FieldContainer>
                <div className="flex items-center gap-2 mb-2">
                  <img src="/public/apple_music.png" alt="Apple Music" className="w-4 h-4" />
                  <Label htmlFor="appleMusic" className="text-sm font-semibold text-muted-foreground">Apple Music</Label>
                </div>
                {isEditMode ? (
                  <Input 
                    id="appleMusic"
                    value={formData.dsp.appleMusic}
                    onChange={(e) => updateNestedField('dsp', 'appleMusic', e.target.value)}
                    placeholder="https://music.apple.com/..."
                    className="bg-background border-2 focus:border-red-500 transition-colors"
                  />
                ) : (
                  <p className="text-foreground text-lg break-all">{formData.dsp.appleMusic}</p>
                )}
              </FieldContainer>

              <FieldContainer>
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-4 h-4 rounded-full bg-orange-500" />
                  <Label htmlFor="soundcloud" className="text-sm font-semibold text-muted-foreground">SoundCloud</Label>
                </div>
                {isEditMode ? (
                  <Input 
                    id="soundcloud"
                    value={formData.dsp.soundcloud}
                    onChange={(e) => updateNestedField('dsp', 'soundcloud', e.target.value)}
                    placeholder="soundcloud.com/artist"
                    className="bg-background border-2 focus:border-orange-500 transition-colors"
                  />
                ) : (
                  <p className="text-foreground text-lg">{formData.dsp.soundcloud}</p>
                )}
              </FieldContainer>
            </CardContent>
          </Card>

          {/* Additional Links */}
          <Card className="border-2 shadow-md hover:shadow-lg transition-shadow">
            <CardHeader className="bg-gradient-to-r from-purple-500/5 to-transparent">
              <div className="flex items-center gap-2">
                <LinkIcon className="w-5 h-5 text-purple-500" />
                <CardTitle>Additional Links</CardTitle>
              </div>
              <CardDescription>Any other important links pertaining to the artist</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 pt-6">
              <FieldContainer>
                <div className="flex items-center gap-2 mb-2">
                  <LinkIcon className="w-4 h-4 text-purple-500" />
                  <Label htmlFor="epk" className="text-sm font-semibold text-muted-foreground">Electronic Press Kit (EPK)</Label>
                </div>
                {isEditMode ? (
                  <Input 
                    id="epk"
                    value={formData.additional.epk}
                    onChange={(e) => updateNestedField('additional', 'epk', e.target.value)}
                    placeholder="https://epk.example.com"
                    className="bg-background border-2 focus:border-purple-500 transition-colors"
                  />
                ) : (
                  <p className="text-foreground text-lg break-all">{formData.additional.epk}</p>
                )}
              </FieldContainer>

              <FieldContainer>
                <div className="flex items-center gap-2 mb-2">
                  <LinkIcon className="w-4 h-4 text-blue-500" />
                  <Label htmlFor="pressKit" className="text-sm font-semibold text-muted-foreground">Press Kit</Label>
                </div>
                {isEditMode ? (
                  <Input 
                    id="pressKit"
                    value={formData.additional.pressKit}
                    onChange={(e) => updateNestedField('additional', 'pressKit', e.target.value)}
                    placeholder="https://press.example.com"
                    className="bg-background border-2 focus:border-blue-500 transition-colors"
                  />
                ) : (
                  <p className="text-foreground text-lg break-all">{formData.additional.pressKit}</p>
                )}
              </FieldContainer>

              <FieldContainer>
                <div className="flex items-center gap-2 mb-2">
                  <LinkIcon className="w-4 h-4 text-green-500" />
                  <Label htmlFor="linktree" className="text-sm font-semibold text-muted-foreground">Linktree</Label>
                </div>
                {isEditMode ? (
                  <Input 
                    id="linktree"
                    value={formData.additional.linktree}
                    onChange={(e) => updateNestedField('additional', 'linktree', e.target.value)}
                    placeholder="https://linktr.ee/artist"
                    className="bg-background border-2 focus:border-green-500 transition-colors"
                  />
                ) : (
                  <p className="text-foreground text-lg break-all">{formData.additional.linktree}</p>
                )}
              </FieldContainer>

              {/* Custom Links Section */}
              {(formData.customLinks.length > 0 || isEditMode) && (
                <div className="pt-4 border-t border-border">
                  <div className="flex items-center justify-between mb-4">
                    <Label className="text-sm font-semibold text-muted-foreground">Custom Links</Label>
                    {isEditMode && (
                      <Button 
                        type="button" 
                        variant="outline" 
                        size="sm"
                        onClick={addCustomLink}
                        className="text-xs"
                      >
                        <LinkIcon className="w-3 h-3 mr-1" />
                        Add Link
                      </Button>
                    )}
                  </div>
                  
                  <div className="space-y-3">
                    {formData.customLinks.map((link) => (
                      <FieldContainer key={link.id} className="relative">
                        {isEditMode ? (
                          <div className="space-y-3">
                            <div className="flex items-center gap-2">
                              <Input 
                                value={link.label}
                                onChange={(e) => updateCustomLink(link.id, 'label', e.target.value)}
                                placeholder="Link Label (e.g., Website, Merch)"
                                className="bg-background border-2 focus:border-purple-500 transition-colors flex-1"
                              />
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                onClick={() => removeCustomLink(link.id)}
                                className="text-destructive hover:text-destructive hover:bg-destructive/10"
                              >
                                <X className="w-4 h-4" />
                              </Button>
                            </div>
                            <Input 
                              value={link.url}
                              onChange={(e) => updateCustomLink(link.id, 'url', e.target.value)}
                              placeholder="https://example.com"
                              className="bg-background border-2 focus:border-purple-500 transition-colors"
                            />
                          </div>
                        ) : (
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <LinkIcon className="w-4 h-4 text-purple-500" />
                              <Label className="text-sm font-semibold text-muted-foreground">{link.label}</Label>
                            </div>
                            <p className="text-foreground text-lg break-all">{link.url}</p>
                          </div>
                        )}
                      </FieldContainer>
                    ))}
                    
                    {formData.customLinks.length === 0 && isEditMode && (
                      <p className="text-sm text-muted-foreground text-center py-4">
                        No custom links added. Click "Add Link" to create one.
                      </p>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Contract Management */}
          <Card className="border-2 shadow-md hover:shadow-lg transition-shadow">
            <CardHeader className="bg-gradient-to-r from-amber-500/5 to-transparent">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-amber-500" />
                <CardTitle>Contract Management</CardTitle>
              </div>
              <CardDescription>Upload and manage artist contracts</CardDescription>
            </CardHeader>
            <CardContent className="pt-6">
              {hasContract ? (
                <div className="space-y-4">
                  <FieldContainer className="bg-gradient-to-br from-primary/5 to-transparent border-2 border-primary/20">
                    <div className="flex items-center gap-4">
                      <div className="p-3 rounded-lg bg-primary/10">
                        <FileText className="w-8 h-8 text-primary" />
                      </div>
                      <div className="flex-1">
                        <p className="font-semibold text-foreground text-lg">Artist Contract.pdf</p>
                        <p className="text-sm text-muted-foreground">Uploaded 3 months ago</p>
                      </div>
                    </div>
                  </FieldContainer>
                  <div className="flex gap-3">
                    <Button variant="outline" className="flex-1">
                      <FileText className="w-4 h-4 mr-2" />
                      View Contract
                    </Button>
                    <Button variant="outline" className="flex-1" onClick={() => setHasContract(false)}>
                      <Upload className="w-4 h-4 mr-2" />
                      Replace Contract
                    </Button>
                  </div>
                </div>
              ) : (
                <FieldContainer className="border-2 border-dashed text-center py-8">
                  <div className="flex flex-col items-center">
                    <div className="p-4 rounded-full bg-muted/50 mb-4">
                      <Upload className="w-12 h-12 text-muted-foreground" />
                    </div>
                    <p className="text-foreground font-semibold text-lg mb-2">No contract uploaded</p>
                    <p className="text-muted-foreground mb-6 text-sm max-w-sm">
                      Upload a contract to enable royalty calculations and track artist agreements
                    </p>
                    <Button onClick={handleContractUpload} size="lg">
                      <Upload className="w-4 h-4 mr-2" />
                      Upload Contract
                    </Button>
                  </div>
                </FieldContainer>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default ArtistProfile;
