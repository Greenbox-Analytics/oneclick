import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import type { Tables } from "@/integrations/supabase/types";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Music, ArrowLeft, Camera, Edit, Save, X, Instagram, Youtube, MessageCircle, Mic2, Link as LinkIcon, Users, Music2, Trash2, CheckCircle, BookOpen, Plus, StickyNote } from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useQuery } from "@tanstack/react-query";
import NotesView from "@/components/notes/NotesView";
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

// Helper component for field containers (defined outside to prevent re-renders)
const FieldContainer = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <div className={`p-4 rounded-lg border border-border bg-card/50 hover:bg-card transition-colors ${className}`}>
    {children}
  </div>
);

const ArtistProfile = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const { toast } = useToast();
  const { user } = useAuth();
  const [hasContract, setHasContract] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [projects, setProjects] = useState<Tables<'projects'>[]>([]);

  const [formData, setFormData] = useState({
    name: "",
    email: "",
    bio: "",
    genres: [] as string[],
    avatar: "",
    social: {
      instagram: "",
      tiktok: "",
      youtube: "",
    },
    dsp: {
      spotify: "",
      appleMusic: "",
      soundcloud: "",
    },
    additional: {
      epk: "",
      pressKit: "",
      linktree: "",
    },
    customLinks: [] as { id: string; label: string; url: string }[],
    customSocialLinks: [] as { id: string; label: string; url: string }[],
    customDspLinks: [] as { id: string; label: string; url: string }[],
  });
  
  const [originalData, setOriginalData] = useState(formData);

  const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

  // Fetch TeamCard overlay for verified artists
  const teamcardQuery = useQuery({
    queryKey: ["artist-teamcard", id],
    queryFn: async () => {
      if (!user?.id || !id) return null;
      const res = await fetch(`${API_URL}/registry/artists/${id}/with-teamcard?user_id=${user.id}`);
      if (!res.ok) return null;
      return res.json();
    },
    enabled: !!user?.id && !!id,
  });

  const teamcard = teamcardQuery.data?.teamcard;
  const isVerified = teamcardQuery.data?.verified === true;

  // Display fields: prefer TeamCard data when verified
  const displayName = (isVerified && teamcard?.display_name) || formData.name;
  const displayBio = (isVerified && teamcard?.bio) || formData.bio;
  const displayAvatar = (isVerified && teamcard?.avatar_url) || formData.avatar;

  // Fetch artist data from Supabase
  useEffect(() => {
    const fetchArtist = async () => {
      if (!id) return;
      
      setIsLoading(true);
      const { data, error } = await supabase
        .from('artists')
        .select('*')
        .eq('id', id)
        .single();

      if (error) {
        toast({
          title: "Error",
          description: "Failed to load artist data",
          variant: "destructive",
        });
        setIsLoading(false);
        return;
      }

      if (data) {
        const artistData = {
          name: data.name,
          email: data.email,
          bio: data.bio || "",
          genres: data.genres || [],
          avatar: data.avatar_url || "",
          social: {
            instagram: data.social_instagram || "",
            tiktok: data.social_tiktok || "",
            youtube: data.social_youtube || "",
          },
          dsp: {
            spotify: data.dsp_spotify || "",
            appleMusic: data.dsp_apple_music || "",
            soundcloud: data.dsp_soundcloud || "",
          },
          additional: {
            epk: data.additional_epk || "",
            pressKit: data.additional_press_kit || "",
            linktree: data.additional_linktree || "",
          },
          customLinks: Array.isArray(data.custom_links) 
            ? data.custom_links.map((link: any) => ({
                id: link.id || Date.now().toString(),
                label: link.label || "",
                url: link.url || ""
              }))
            : [],
          customSocialLinks: Array.isArray(data.custom_social_links)
            ? data.custom_social_links.map((link: any) => ({
                id: link.id || Date.now().toString(),
                label: link.label || "",
                url: link.url || ""
              }))
            : [],
          customDspLinks: Array.isArray(data.custom_dsp_links)
            ? data.custom_dsp_links.map((link: any) => ({
                id: link.id || Date.now().toString(),
                label: link.label || "",
                url: link.url || ""
              }))
            : [],
        };
        setFormData(artistData);
        setOriginalData(artistData);
        setHasContract(data.has_contract || false);
      }
      setIsLoading(false);
    };

    const fetchProjects = async () => {
      if (!id) return;

      const { data, error } = await supabase
        .from('projects')
        .select('*')
        .eq('artist_id', id)
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Error fetching projects:', error);
        return;
      }

      setProjects(data || []);
    };

    fetchArtist();
    fetchProjects();
  }, [id, toast]);

  const handleContractUpload = () => {
    // TODO: Add backend contract upload logic
    setHasContract(true);
  };

  const handleSave = async () => {
    if (!id) return;

    const { error } = await supabase
      .from('artists')
      .update({
        name: formData.name,
        email: formData.email,
        bio: formData.bio,
        genres: formData.genres,
        avatar_url: formData.avatar,
        social_instagram: formData.social.instagram,
        social_tiktok: formData.social.tiktok,
        social_youtube: formData.social.youtube,
        dsp_spotify: formData.dsp.spotify,
        dsp_apple_music: formData.dsp.appleMusic,
        dsp_soundcloud: formData.dsp.soundcloud,
        additional_epk: formData.additional.epk,
        additional_press_kit: formData.additional.pressKit,
        additional_linktree: formData.additional.linktree,
        custom_links: formData.customLinks,
        custom_social_links: formData.customSocialLinks,
        custom_dsp_links: formData.customDspLinks,
      })
      .eq('id', id);

    if (error) {
      toast({
        title: "Error",
        description: "Failed to save artist data",
        variant: "destructive",
      });
      return;
    }

    toast({
      title: "Success",
      description: "Artist profile updated successfully",
    });
    setIsEditMode(false);
    setOriginalData(formData);
  };

  const handleCancel = () => {
    setFormData(originalData);
    setIsEditMode(false);
  };

  const handleDeleteArtist = async () => {
    if (!id) return;

    const { error } = await supabase
      .from('artists')
      .delete()
      .eq('id', id);

    if (error) {
      toast({
        title: "Error",
        description: "Failed to delete artist. Please try again.",
        variant: "destructive",
      });
    } else {
      toast({
        title: "Success",
        description: `${formData.name} has been deleted.`,
      });
      navigate("/artists");
    }
    setShowDeleteDialog(false);
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

  const addCustomSocialLink = () => {
    const newLink = {
      id: Date.now().toString(),
      label: "",
      url: ""
    };
    setFormData(prev => ({
      ...prev,
      customSocialLinks: [...prev.customSocialLinks, newLink]
    }));
  };

  const updateCustomSocialLink = (linkId: string, field: 'label' | 'url', value: string) => {
    setFormData(prev => ({
      ...prev,
      customSocialLinks: prev.customSocialLinks.map(link =>
        link.id === linkId ? { ...link, [field]: value } : link
      )
    }));
  };

  const removeCustomSocialLink = (linkId: string) => {
    setFormData(prev => ({
      ...prev,
      customSocialLinks: prev.customSocialLinks.filter(link => link.id !== linkId)
    }));
  };

  const addCustomDspLink = () => {
    const newLink = {
      id: Date.now().toString(),
      label: "",
      url: ""
    };
    setFormData(prev => ({
      ...prev,
      customDspLinks: [...prev.customDspLinks, newLink]
    }));
  };

  const updateCustomDspLink = (linkId: string, field: 'label' | 'url', value: string) => {
    setFormData(prev => ({
      ...prev,
      customDspLinks: prev.customDspLinks.map(link =>
        link.id === linkId ? { ...link, [field]: value } : link
      )
    }));
  };

  const removeCustomDspLink = (linkId: string) => {
    setFormData(prev => ({
      ...prev,
      customDspLinks: prev.customDspLinks.filter(link => link.id !== linkId)
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/10">
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
          <div className="flex gap-2 items-center">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/docs")}
              title="Documentation"
              className="text-muted-foreground hover:text-foreground"
            >
              <BookOpen className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={() => navigate("/dashboard")} title="Back to Dashboard">
              <ArrowLeft className="w-4 h-4" />
            </Button>
            {isEditMode && (
              <Button
                variant="outline"
                size="sm"
                className="text-destructive hover:bg-destructive hover:text-destructive-foreground"
                onClick={() => setShowDeleteDialog(true)}
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Delete
              </Button>
            )}
            <Button
              variant={isEditMode ? "outline" : "default"}
              size="sm"
              onClick={() => isEditMode ? handleCancel() : setIsEditMode(true)}
            >
              {isEditMode ? <><X className="w-4 h-4 mr-2" />Cancel</> : <><Edit className="w-4 h-4 mr-2" />Edit</>}
            </Button>
            {isEditMode && (
              <Button size="sm" onClick={handleSave}>
                <Save className="w-4 h-4 mr-2" />
                Save
              </Button>
            )}
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Profile Header */}
        <Card className="mb-6 border border-border shadow-sm">
          <CardContent className="pt-6">
            <div className="flex items-start gap-6">
              <div className="relative group">
                <Avatar className="w-28 h-28 ring-2 ring-primary/20">
                  <AvatarImage src={displayAvatar} alt={displayName} />
                  <AvatarFallback className="bg-gradient-to-br from-primary to-primary/70 text-primary-foreground text-3xl">
                    {displayName.charAt(0)}
                  </AvatarFallback>
                </Avatar>
                {isEditMode && (
                  <button className="absolute inset-0 bg-background/90 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all hover:scale-105">
                    <Camera className="w-7 h-7 text-primary" />
                  </button>
                )}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-3">
                  <h2 className="text-2xl font-bold text-foreground">{displayName}</h2>
                  {isVerified && (
                    <Badge className="bg-green-100 text-green-800 flex items-center gap-1">
                      <CheckCircle className="w-3 h-3" /> Verified
                    </Badge>
                  )}
                </div>
                <div className="flex flex-wrap gap-2 mb-3">
                  {formData.genres.map(genre => (
                    <Badge key={genre} variant="secondary" className="px-3 py-1 text-sm">{genre}</Badge>
                  ))}
                </div>
                <p className="text-muted-foreground text-sm line-clamp-2">{displayBio}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid gap-6">
          {/* Basic Information */}
          <Card className="border border-border shadow-sm hover:shadow-md transition-shadow overflow-hidden">
            <div className="h-0.5 bg-primary/40" />
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
                ) : formData.name ? (
                  <p className="text-sm font-medium text-foreground">{formData.name}</p>
                ) : (
                  <p className="text-sm text-muted-foreground/50 italic">Not set</p>
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
                ) : formData.email ? (
                  <p className="text-sm text-foreground">{formData.email}</p>
                ) : (
                  <p className="text-sm text-muted-foreground/50 italic">Not set</p>
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
                ) : formData.bio ? (
                  <p className="text-foreground leading-relaxed">{formData.bio}</p>
                ) : (
                  <p className="text-sm text-muted-foreground/50 italic">Not set</p>
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
                    {formData.genres.length > 0 ? (
                      formData.genres.map(genre => (
                        <Badge key={genre} variant="secondary" className="px-3 py-1">{genre}</Badge>
                      ))
                    ) : (
                      <p className="text-sm text-muted-foreground/50 italic">No genres added</p>
                    )}
                  </div>
                )}
              </FieldContainer>
            </CardContent>
          </Card>

          {/* Social Media */}
          <Card className="border border-border shadow-sm hover:shadow-md transition-shadow overflow-hidden">
            <div className="h-0.5 bg-pink-500/40" />
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
                    placeholder="https://instagram.com/username"
                    className="bg-background border-2 focus:border-pink-500 transition-colors"
                  />
                ) : formData.social.instagram ? (
                  <a href={formData.social.instagram} target="_blank" rel="noopener noreferrer" className="text-sm text-primary hover:underline break-all">
                    {formData.social.instagram}
                  </a>
                ) : (
                  <p className="text-sm text-muted-foreground/50 italic">Not set</p>
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
                    placeholder="https://tiktok.com/@username"
                    className="bg-background border-2 focus:border-blue-500 transition-colors"
                  />
                ) : formData.social.tiktok ? (
                  <a href={formData.social.tiktok} target="_blank" rel="noopener noreferrer" className="text-sm text-primary hover:underline break-all">
                    {formData.social.tiktok}
                  </a>
                ) : (
                  <p className="text-sm text-muted-foreground/50 italic">Not set</p>
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
                    placeholder="https://youtube.com/@channel"
                    className="bg-background border-2 focus:border-red-500 transition-colors"
                  />
                ) : formData.social.youtube ? (
                  <a href={formData.social.youtube} target="_blank" rel="noopener noreferrer" className="text-sm text-primary hover:underline break-all">
                    {formData.social.youtube}
                  </a>
                ) : (
                  <p className="text-sm text-muted-foreground/50 italic">Not set</p>
                )}
              </FieldContainer>

              {/* Custom Social Links Section */}
              {(formData.customSocialLinks.length > 0 || isEditMode) && (
                <div className="pt-4 border-t border-border">
                  <div className="flex items-center justify-between mb-4">
                    <Label className="text-sm font-semibold text-muted-foreground">Custom Social Links</Label>
                    {isEditMode && (
                      <Button 
                        type="button" 
                        variant="outline" 
                        size="sm"
                        onClick={addCustomSocialLink}
                        className="text-xs"
                      >
                        <Plus className="w-3 h-3 mr-1" />
                        Add Link
                      </Button>
                    )}
                  </div>
                  
                  <div className="space-y-3">
                    {formData.customSocialLinks.map((link) => (
                      <FieldContainer key={link.id} className="relative">
                        {isEditMode ? (
                          <div className="space-y-3">
                            <div className="flex items-center gap-2">
                              <Input 
                                value={link.label}
                                onChange={(e) => updateCustomSocialLink(link.id, 'label', e.target.value)}
                                placeholder="Platform Name (e.g., Twitter, Threads)"
                                className="bg-background border-2 focus:border-pink-500 transition-colors flex-1"
                              />
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                onClick={() => removeCustomSocialLink(link.id)}
                                className="text-destructive hover:text-destructive hover:bg-destructive/10"
                              >
                                <X className="w-4 h-4" />
                              </Button>
                            </div>
                            <Input 
                              value={link.url}
                              onChange={(e) => updateCustomSocialLink(link.id, 'url', e.target.value)}
                              placeholder="https://example.com/username"
                              className="bg-background border-2 focus:border-pink-500 transition-colors"
                            />
                          </div>
                        ) : (
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <LinkIcon className="w-4 h-4 text-pink-500" />
                              <Label className="text-sm font-semibold text-muted-foreground">{link.label}</Label>
                            </div>
                            {link.url ? (
                              <a href={link.url} target="_blank" rel="noopener noreferrer" className="text-sm text-primary hover:underline break-all">
                                {link.url}
                              </a>
                            ) : (
                              <p className="text-sm text-muted-foreground/50 italic">Not set</p>
                            )}
                          </div>
                        )}
                      </FieldContainer>
                    ))}

                    {formData.customSocialLinks.length === 0 && isEditMode && (
                      <p className="text-sm text-muted-foreground text-center py-4">
                        No custom social links added. Click "Add Link" to create one.
                      </p>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Streaming Platforms */}
          <Card className="border border-border shadow-sm hover:shadow-md transition-shadow overflow-hidden">
            <div className="h-0.5 bg-green-500/40" />
            <CardHeader className="bg-gradient-to-r from-green-500/5 to-transparent">
              <div className="flex items-center gap-2">
                <Mic2 className="w-5 h-5 text-green-500" />
                <CardTitle>Streaming Platforms</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-4 pt-6">
              <FieldContainer>
                <div className="flex items-center gap-2 mb-2">
                  <img src="/spotify.svg" alt="Spotify" className="w-4 h-4" />
                  <Label htmlFor="spotify" className="text-sm font-semibold text-muted-foreground">Spotify</Label>
                </div>
                {isEditMode ? (
                  <Input 
                    id="spotify"
                    value={formData.dsp.spotify}
                    onChange={(e) => updateNestedField('dsp', 'spotify', e.target.value)}
                    placeholder="https://open.spotify.com/artist/..."
                    className="bg-background border-2 focus:border-green-500 transition-colors"
                  />
                ) : formData.dsp.spotify ? (
                  <a href={formData.dsp.spotify} target="_blank" rel="noopener noreferrer" className="text-sm text-primary hover:underline break-all font-mono">
                    {formData.dsp.spotify}
                  </a>
                ) : (
                  <p className="text-sm text-muted-foreground/50 italic">Not set</p>
                )}
              </FieldContainer>

              <FieldContainer>
                <div className="flex items-center gap-2 mb-2">
                  <img src="/apple_music.png" alt="Apple Music" className="w-4 h-4" />
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
                ) : formData.dsp.appleMusic ? (
                  <a href={formData.dsp.appleMusic} target="_blank" rel="noopener noreferrer" className="text-sm text-primary hover:underline break-all">
                    {formData.dsp.appleMusic}
                  </a>
                ) : (
                  <p className="text-sm text-muted-foreground/50 italic">Not set</p>
                )}
              </FieldContainer>

              <FieldContainer>
                <div className="flex items-center gap-2 mb-2">
                  <img src="/soundcloud.png" alt="SoundCloud" className="w-4 h-4" />
                  <Label htmlFor="soundcloud" className="text-sm font-semibold text-muted-foreground">SoundCloud</Label>
                </div>
                {isEditMode ? (
                  <Input 
                    id="soundcloud"
                    value={formData.dsp.soundcloud}
                    onChange={(e) => updateNestedField('dsp', 'soundcloud', e.target.value)}
                    placeholder="https://soundcloud.com/artist"
                    className="bg-background border-2 focus:border-orange-500 transition-colors"
                  />
                ) : formData.dsp.soundcloud ? (
                  <a href={formData.dsp.soundcloud} target="_blank" rel="noopener noreferrer" className="text-sm text-primary hover:underline break-all">
                    {formData.dsp.soundcloud}
                  </a>
                ) : (
                  <p className="text-sm text-muted-foreground/50 italic">Not set</p>
                )}
              </FieldContainer>

              {/* Custom DSP Links Section */}
              {(formData.customDspLinks.length > 0 || isEditMode) && (
                <div className="pt-4 border-t border-border">
                  <div className="flex items-center justify-between mb-4">
                    <Label className="text-sm font-semibold text-muted-foreground">Custom DSP Links</Label>
                    {isEditMode && (
                      <Button 
                        type="button" 
                        variant="outline" 
                        size="sm"
                        onClick={addCustomDspLink}
                        className="text-xs"
                      >
                        <Plus className="w-3 h-3 mr-1" />
                        Add Link
                      </Button>
                    )}
                  </div>
                  
                  <div className="space-y-3">
                    {formData.customDspLinks.map((link) => (
                      <FieldContainer key={link.id} className="relative">
                        {isEditMode ? (
                          <div className="space-y-3">
                            <div className="flex items-center gap-2">
                              <Input 
                                value={link.label}
                                onChange={(e) => updateCustomDspLink(link.id, 'label', e.target.value)}
                                placeholder="Platform Name (e.g., Tidal, Deezer)"
                                className="bg-background border-2 focus:border-green-500 transition-colors flex-1"
                              />
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                onClick={() => removeCustomDspLink(link.id)}
                                className="text-destructive hover:text-destructive hover:bg-destructive/10"
                              >
                                <X className="w-4 h-4" />
                              </Button>
                            </div>
                            <Input 
                              value={link.url}
                              onChange={(e) => updateCustomDspLink(link.id, 'url', e.target.value)}
                              placeholder="https://example.com/artist/..."
                              className="bg-background border-2 focus:border-green-500 transition-colors"
                            />
                          </div>
                        ) : (
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <LinkIcon className="w-4 h-4 text-green-500" />
                              <Label className="text-sm font-semibold text-muted-foreground">{link.label}</Label>
                            </div>
                            {link.url ? (
                              <a href={link.url} target="_blank" rel="noopener noreferrer" className="text-sm text-primary hover:underline break-all">
                                {link.url}
                              </a>
                            ) : (
                              <p className="text-sm text-muted-foreground/50 italic">Not set</p>
                            )}
                          </div>
                        )}
                      </FieldContainer>
                    ))}

                    {formData.customDspLinks.length === 0 && isEditMode && (
                      <p className="text-sm text-muted-foreground text-center py-4">
                        No custom DSP links added. Click "Add Link" to create one.
                      </p>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Additional Links */}
          <Card className="border border-border shadow-sm hover:shadow-md transition-shadow overflow-hidden">
            <div className="h-0.5 bg-purple-500/40" />
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
                ) : formData.additional.epk ? (
                  <a href={formData.additional.epk} target="_blank" rel="noopener noreferrer" className="text-sm text-primary hover:underline break-all">
                    {formData.additional.epk}
                  </a>
                ) : (
                  <p className="text-sm text-muted-foreground/50 italic">Not set</p>
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
                ) : formData.additional.pressKit ? (
                  <a href={formData.additional.pressKit} target="_blank" rel="noopener noreferrer" className="text-sm text-primary hover:underline break-all">
                    {formData.additional.pressKit}
                  </a>
                ) : (
                  <p className="text-sm text-muted-foreground/50 italic">Not set</p>
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
                ) : formData.additional.linktree ? (
                  <a href={formData.additional.linktree} target="_blank" rel="noopener noreferrer" className="text-sm text-primary hover:underline break-all">
                    {formData.additional.linktree}
                  </a>
                ) : (
                  <p className="text-sm text-muted-foreground/50 italic">Not set</p>
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
                            {link.url ? (
                              <a href={link.url} target="_blank" rel="noopener noreferrer" className="text-sm text-primary hover:underline break-all">
                                {link.url}
                              </a>
                            ) : (
                              <p className="text-sm text-muted-foreground/50 italic">Not set</p>
                            )}
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

          {/* My Notes -- private to you */}
          <Card className="border border-border shadow-sm hover:shadow-md transition-shadow overflow-hidden">
            <div className="h-0.5 bg-amber-500/40" />
            <CardHeader className="bg-gradient-to-r from-amber-500/5 to-transparent">
              <div className="flex items-center gap-2">
                <StickyNote className="w-5 h-5 text-amber-500" />
                <CardTitle>My Notes</CardTitle>
              </div>
              <CardDescription>Private notes about this artist</CardDescription>
            </CardHeader>
            <CardContent className="pt-6">
              <NotesView scope={{ artistId: id }} />
            </CardContent>
          </Card>

        </div>
      </main>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Artist Profile?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete <strong>{formData.name}</strong> and all associated data. This action cannot be undone.
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

export default ArtistProfile;
