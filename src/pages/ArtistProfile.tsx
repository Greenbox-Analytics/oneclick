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
import { Music, Upload, FileText, ArrowLeft, Camera, Edit, Save, X, Instagram, Youtube, MessageCircle, Mic2, Link as LinkIcon, Users, Music2, Trash2, Folder, Plus } from "lucide-react";
import { useNavigate, useParams } from "react-router-dom";
import { ContractUploadModal } from "@/components/ContractUploadModal";
import { RoyaltyStatementUploadModal } from "@/components/RoyaltyStatementUploadModal";
import { useAuth } from "@/contexts/AuthContext";
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

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
  const [newProjectName, setNewProjectName] = useState("");
  const [showNewProject, setShowNewProject] = useState(false);
  const [projectFiles, setProjectFiles] = useState<Record<string, any[]>>({});
  const [uploadingFile, setUploadingFile] = useState<string | null>(null);
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [selectedFileType, setSelectedFileType] = useState<string | null>(null);
  const [fileToDelete, setFileToDelete] = useState<any>(null);
  const [contractUploadModalOpen, setContractUploadModalOpen] = useState(false);
  const [contractUploadProjectId, setContractUploadProjectId] = useState<string>("");
  const [royaltyStatementUploadModalOpen, setRoyaltyStatementUploadModalOpen] = useState(false);
  const [royaltyStatementUploadProjectId, setRoyaltyStatementUploadProjectId] = useState<string>("");

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

  // Fetch files for all projects
  useEffect(() => {
    const fetchAllProjectFiles = async () => {
      if (projects.length === 0) return;

      const filesPromises = projects.map(async (project) => {
        const { data, error } = await supabase
          .from('project_files')
          .select('*')
          .eq('project_id', project.id);

        if (error) {
          console.error('Error fetching files:', error);
          return { projectId: project.id, files: [] };
        }

        return { projectId: project.id, files: data || [] };
      });

      const results = await Promise.all(filesPromises);
      const filesMap: Record<string, any[]> = {};
      results.forEach(({ projectId, files }) => {
        filesMap[projectId] = files;
      });
      setProjectFiles(filesMap);
    };

    fetchAllProjectFiles();
  }, [projects]);

  const handleContractUploadClick = (projectId: string) => {
    setContractUploadProjectId(projectId);
    setContractUploadModalOpen(true);
  };

  const handleContractUploadComplete = () => {
    // Refresh project files after upload
    if (contractUploadProjectId) {
      fetchProjectFiles(contractUploadProjectId);
    }
  };

  const handleRoyaltyStatementUploadClick = (projectId: string) => {
    setRoyaltyStatementUploadProjectId(projectId);
    setRoyaltyStatementUploadModalOpen(true);
  };

  const handleRoyaltyStatementUploadComplete = () => {
    // Refresh project files after upload
    if (royaltyStatementUploadProjectId) {
      fetchProjectFiles(royaltyStatementUploadProjectId);
    }
  };

  const normalizeFileName = (name: string) => name.trim().toLowerCase();

  const fetchProjectFiles = async (projectId: string) => {
    const { data, error } = await supabase
      .from('project_files')
      .select('*')
      .eq('project_id', projectId);

    if (error) {
      console.error('Error fetching files:', error);
      return;
    }

    setProjectFiles(prev => ({
      ...prev,
      [projectId]: data || [],
    }));
  };

  const handleFileUpload = async (projectId: string, folderCategory: string, event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const projectFilesForValidation = projectFiles[projectId]
      ? projectFiles[projectId]
      : await (async () => {
          const { data, error } = await supabase
            .from('project_files')
            .select('file_name')
            .eq('project_id', projectId);

          if (error) {
            return [];
          }

          return data || [];
        })();

    const hasDuplicateInCache = projectFilesForValidation.some(
      existing => normalizeFileName(existing.file_name) === normalizeFileName(file.name)
    );

    if (hasDuplicateInCache) {
      toast({
        title: "Duplicate file name",
        description: `A file named "${file.name}" already exists in this project.`,
        variant: "destructive",
        duration: Number.POSITIVE_INFINITY,
      });
      event.target.value = "";
      return;
    }

    setUploadingFile(`${projectId}-${folderCategory}`);

    try {
      // Upload to Supabase storage
      const filePath = `${projectId}/${folderCategory}/${Date.now()}_${file.name}`;
      const { error: uploadError } = await supabase.storage
        .from('project-files')
        .upload(filePath, file);

      if (uploadError) throw uploadError;

      // Get public URL
      const { data: urlData } = supabase.storage
        .from('project-files')
        .getPublicUrl(filePath);

      // Create database record
      const { data, error: dbError } = await supabase
        .from('project_files')
        .insert({
          project_id: projectId,
          file_name: file.name,
          file_url: urlData.publicUrl,
          file_path: filePath,
          folder_category: folderCategory,
          file_size: file.size,
          file_type: file.type,
        })
        .select()
        .single();

      if (dbError) {
        const errorMessage = dbError.message?.toLowerCase() || "";
        if (errorMessage.includes("duplicate") || errorMessage.includes("unique")) {
          await supabase.storage.from('project-files').remove([filePath]);
          toast({
            title: "Duplicate file name",
            description: `A file named "${file.name}" already exists in this project.`,
            variant: "destructive",
            duration: Number.POSITIVE_INFINITY,
          });
          return;
        }
        throw dbError;
      }

      // Update local state
      setProjectFiles(prev => ({
        ...prev,
        [projectId]: [...(prev[projectId] || []), data],
      }));

      toast({
        title: "Success",
        description: "File uploaded successfully",
      });
    } catch (error: any) {
      console.error('Upload error:', error);
      toast({
        title: "Error",
        description: error.message || "Failed to upload file",
        variant: "destructive",
      });
    } finally {
      setUploadingFile(null);
      event.target.value = "";
    }
  };

  const handleFileView = async (file: any) => {
    try {
      const { data, error } = await supabase.storage
        .from('project-files')
        .createSignedUrl(file.file_path, 3600); // 1 hour expiry

      if (error) throw error;

      window.open(data.signedUrl, '_blank');
    } catch (error: any) {
      toast({
        title: "Error",
        description: "Failed to open file",
        variant: "destructive",
      });
    }
  };

  const handleFileDownload = async (file: any) => {
    try {
      const { data, error } = await supabase.storage
        .from('project-files')
        .download(file.file_path);

      if (error) throw error;

      const url = URL.createObjectURL(data);
      const a = document.createElement('a');
      a.href = url;
      a.download = file.file_name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error: any) {
      toast({
        title: "Error",
        description: "Failed to download file",
        variant: "destructive",
      });
    }
  };

  const confirmFileDelete = (file: any) => {
    setFileToDelete(file);
  };

  const handleFileDelete = async () => {
    if (!fileToDelete || !user) return;

    try {
      // If it's a contract file, also delete from vector database
      if (fileToDelete.folder_category === 'contract') {
        const formData = new FormData();
        formData.append("user_id", user.id);

        const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";
        const vectorDeleteResponse = await fetch(`${API_URL}/contracts/${fileToDelete.id}`, {
          method: "DELETE",
          body: formData,
        });

        if (!vectorDeleteResponse.ok) {
          const errorData = await vectorDeleteResponse.json();
          throw new Error(errorData.detail || "Failed to delete contract vectors");
        }

        // The backend already handles storage and database deletion for contracts
        // Update local state
        setProjectFiles(prev => ({
          ...prev,
          [fileToDelete.project_id]: (prev[fileToDelete.project_id] || []).filter(f => f.id !== fileToDelete.id),
        }));

        toast({
          title: "Success",
          description: "Contract deleted successfully",
        });
      } else {
        // For non-contract files, use the original deletion logic
        // Delete from storage
        const { error: storageError } = await supabase.storage
          .from('project-files')
          .remove([fileToDelete.file_path]);

        if (storageError) throw storageError;

        // Delete from database
        const { error: dbError } = await supabase
          .from('project_files')
          .delete()
          .eq('id', fileToDelete.id);

        if (dbError) throw dbError;

        // Update local state
        setProjectFiles(prev => ({
          ...prev,
          [fileToDelete.project_id]: (prev[fileToDelete.project_id] || []).filter(f => f.id !== fileToDelete.id),
        }));

        toast({
          title: "Success",
          description: "File deleted successfully",
        });
      }
    } catch (error: any) {
      console.error("Error deleting file:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to delete file",
        variant: "destructive",
      });
    } finally {
      setFileToDelete(null);
    }
  };

  const getFileCountByType = (projectId: string, folderCategory: string) => {
    const files = projectFiles[projectId] || [];
    return files.filter(f => f.folder_category === folderCategory).length;
  };

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

  const handleAddProject = async () => {
    if (!id || !newProjectName.trim()) return;

    const { data, error } = await supabase
      .from('projects')
      .insert({
        artist_id: id,
        name: newProjectName,
      })
      .select()
      .single();

    if (error) {
      console.error('Error creating project:', error);
      toast({
        title: "Error",
        description: "Failed to create project",
        variant: "destructive",
      });
      return;
    }

    setProjects([data, ...projects]);
    setNewProjectName("");
    setShowNewProject(false);
    toast({
      title: "Success",
      description: "Project created successfully",
    });
  };

  const handleDeleteProject = async (projectId: string) => {
    const { error } = await supabase
      .from('projects')
      .delete()
      .eq('id', projectId);

    if (error) {
      console.error('Error deleting project:', error);
      toast({
        title: "Error",
        description: "Failed to delete project",
        variant: "destructive",
      });
      return;
    }

    setProjects(projects.filter(p => p.id !== projectId));
    toast({
      title: "Success",
      description: "Project deleted successfully",
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/10">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div 
            className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate("/dashboard")}
          >
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center p-1.5">
              <Music className="w-full h-full text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              className="text-destructive hover:bg-destructive hover:text-destructive-foreground"
              onClick={() => setShowDeleteDialog(true)}
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Delete Artist
            </Button>
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
                    placeholder="https://instagram.com/username"
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
                    placeholder="https://tiktok.com/@username"
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
                    placeholder="https://youtube.com/@channel"
                    className="bg-background border-2 focus:border-red-500 transition-colors"
                  />
                ) : (
                  <p className="text-foreground text-lg">{formData.social.youtube}</p>
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
                            <p className="text-foreground text-lg break-all">{link.url}</p>
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
                ) : (
                  <p className="text-foreground text-lg font-mono text-sm">{formData.dsp.spotify}</p>
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
                ) : (
                  <p className="text-foreground text-lg break-all">{formData.dsp.appleMusic}</p>
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
                ) : (
                  <p className="text-foreground text-lg">{formData.dsp.soundcloud}</p>
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
                            <p className="text-foreground text-lg break-all">{link.url}</p>
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

          {/* Portfolio Section */}
          <Card className="border-2 shadow-md hover:shadow-lg transition-shadow">
            <CardHeader className="bg-gradient-to-r from-amber-500/5 to-transparent">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-amber-500" />
                  <CardTitle>Portfolio</CardTitle>
                </div>
                <Button
                  onClick={() => setShowNewProject(true)}
                  size="sm"
                  className="bg-amber-500 hover:bg-amber-600"
                >
                  <Plus className="w-4 h-4 mr-1" />
                  New Project
                </Button>
              </div>
              <CardDescription>Manage artist projects and files</CardDescription>
            </CardHeader>
            <CardContent className="pt-6">
              {showNewProject && (
                <div className="mb-6 p-4 border border-border rounded-lg bg-muted/30">
                  <div className="flex gap-2">
                    <Input
                      placeholder="Project name"
                      value={newProjectName}
                      onChange={(e) => setNewProjectName(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleAddProject()}
                    />
                    <Button onClick={handleAddProject} size="sm">
                      Add
                    </Button>
                    <Button onClick={() => {
                      setShowNewProject(false);
                      setNewProjectName("");
                    }} variant="ghost" size="sm">
                      Cancel
                    </Button>
                  </div>
                </div>
              )}

              <div className="space-y-4">
                {projects.length === 0 ? (
                  <p className="text-muted-foreground text-center py-8">
                    No projects yet. Create your first project to get started.
                  </p>
                ) : (
                  projects.map((project) => (
                    <Card key={project.id} className="border-border/50">
                      <CardHeader>
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-lg">{project.name}</CardTitle>
                          <Button
                            onClick={() => handleDeleteProject(project.id)}
                            variant="ghost"
                            size="sm"
                            className="text-destructive hover:text-destructive"
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 gap-3">
                          {[
                            { name: 'Contracts', color: 'amber', category: 'contract' },
                            { name: 'Split Sheets', color: 'blue', category: 'split_sheet' },
                            { name: 'Royalty Statements', color: 'green', category: 'royalty_statement' },
                            { name: 'Other Files', color: 'purple', category: 'other' },
                          ].map((folder) => {
                            const fileCount = getFileCountByType(project.id, folder.category);
                            const isUploading = uploadingFile === `${project.id}-${folder.category}`;
                            
                            return (
                              <div key={folder.category} className="relative">
                                {folder.category !== 'contract' && (
                                  <input
                                    type="file"
                                    id={`upload-${project.id}-${folder.category}`}
                                    className="hidden"
                                    onChange={(e) => handleFileUpload(project.id, folder.category, e)}
                                    disabled={isUploading}
                                  />
                                )}
                                <div 
                                  className="p-3 border border-border rounded-md hover:bg-muted/50 transition-colors cursor-pointer group"
                                  onClick={() => {
                                    setSelectedProject(project.id);
                                    setSelectedFileType(folder.category);
                                  }}
                                >
                                  <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center gap-2 text-sm font-medium">
                                      <Folder className={`w-4 h-4 text-${folder.color}-500`} />
                                      {folder.name}
                                    </div>
                                    <Button
                                      variant="ghost"
                                      size="icon"
                                      className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        if (folder.category === 'contract') {
                                          handleContractUploadClick(project.id);
                                        } else if (folder.category === 'royalty_statement') {
                                          handleRoyaltyStatementUploadClick(project.id);
                                        } else {
                                          document.getElementById(`upload-${project.id}-${folder.category}`)?.click();
                                        }
                                      }}
                                      disabled={isUploading}
                                    >
                                      {isUploading ? (
                                        <div className="w-3 h-3 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                                      ) : (
                                        <Upload className="w-3 h-3" />
                                      )}
                                    </Button>
                                  </div>
                                  <p className="text-xs text-muted-foreground">{fileCount} file{fileCount !== 1 ? 's' : ''}</p>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </CardContent>
                    </Card>
                  ))
                )}
              </div>
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

      {/* Files Viewer Dialog */}
      <Dialog open={selectedProject !== null && selectedFileType !== null} onOpenChange={(open) => {
        if (!open) {
          setSelectedProject(null);
          setSelectedFileType(null);
        }
      }}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Folder className="w-5 h-5" />
              {selectedFileType === 'contract' && 'Contracts'}
              {selectedFileType === 'split_sheet' && 'Split Sheets'}
              {selectedFileType === 'royalty_statement' && 'Royalty Statements'}
              {selectedFileType === 'other' && 'Other Files'}
            </DialogTitle>
            <DialogDescription>
              {projects.find(p => p.id === selectedProject)?.name}
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-2">
            {selectedProject && selectedFileType && projectFiles[selectedProject]?.filter(f => f.folder_category === selectedFileType).length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Folder className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>No files uploaded yet</p>
              </div>
            ) : (
              selectedProject && selectedFileType && projectFiles[selectedProject]?.filter(f => f.folder_category === selectedFileType).map((file) => (
                <div key={file.id} className="flex items-center justify-between p-3 border border-border rounded-lg hover:bg-muted/50 transition-colors">
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <FileText className="w-5 h-5 text-muted-foreground flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="font-medium truncate">{file.file_name}</p>
                      <p className="text-xs text-muted-foreground">
                        {file.file_size ? `${(file.file_size / 1024).toFixed(1)} KB` : 'Unknown size'} • {new Date(file.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleFileView(file)}
                      title="View file"
                    >
                      <FileText className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleFileDownload(file)}
                      title="Download file"
                    >
                      <Upload className="w-4 h-4 rotate-180" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => confirmFileDelete(file)}
                      className="text-destructive hover:text-destructive"
                      title="Delete file"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              ))
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* File Delete Confirmation Dialog */}
      <AlertDialog open={fileToDelete !== null} onOpenChange={(open) => {
        if (!open) setFileToDelete(null);
      }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete File?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete <strong>{fileToDelete?.file_name}</strong>. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleFileDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete File
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Contract Upload Modal */}
      {contractUploadProjectId && user && (
        <ContractUploadModal
          open={contractUploadModalOpen}
          onOpenChange={setContractUploadModalOpen}
          projectId={contractUploadProjectId}
          onUploadComplete={handleContractUploadComplete}
        />
      )}

      {/* Royalty Statement Upload Modal */}
      {royaltyStatementUploadProjectId && (
        <RoyaltyStatementUploadModal
          open={royaltyStatementUploadModalOpen}
          onOpenChange={setRoyaltyStatementUploadModalOpen}
          projectId={royaltyStatementUploadProjectId}
          onUploadComplete={handleRoyaltyStatementUploadComplete}
        />
      )}
    </div>
  );
};

export default ArtistProfile;
