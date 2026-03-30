import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useQuery } from "@tanstack/react-query";
import { useProjectAbout, useUpdateProjectAbout } from "@/hooks/useNotes";
import NotesEditor from "@/components/notes/NotesEditor";
import NotesView from "@/components/notes/NotesView";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Music, ArrowLeft, Loader2, StickyNote, Info } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";

const ProjectDetail = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const updateAbout = useUpdateProjectAbout();

  // Fetch project with artist info
  const projectQuery = useQuery({
    queryKey: ["project-detail", projectId],
    queryFn: async () => {
      if (!projectId) return null;
      const { data } = await supabase
        .from("projects")
        .select("*, artists(name, user_id)")
        .eq("id", projectId)
        .single();
      return data;
    },
    enabled: !!projectId,
  });

  const { data: aboutContent = [] } = useProjectAbout(projectId);

  const project = projectQuery.data;
  const isOwner = project?.artists?.user_id === user?.id;

  const handleAboutChange = (content: unknown[]) => {
    if (!projectId || !isOwner) return;
    updateAbout.mutate({ projectId, about_content: content });
  };

  if (projectQuery.isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!project) {
    return (
      <div className="min-h-screen bg-background flex flex-col items-center justify-center gap-4">
        <p className="text-muted-foreground">Project not found</p>
        <Button variant="outline" onClick={() => navigate("/portfolio")}>
          <ArrowLeft className="w-4 h-4 mr-2" /> Back to Portfolio
        </Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate("/dashboard")}>
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
          </div>
          <Button variant="outline" onClick={() => navigate("/portfolio")}>
            <ArrowLeft className="w-4 h-4 mr-2" /> Back to Portfolio
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-foreground">{project.name}</h2>
          <p className="text-muted-foreground">
            {project.artists?.name || "Unknown Artist"}
            {project.description && ` — ${project.description}`}
          </p>
        </div>

        <Tabs defaultValue="about">
          <TabsList className="mb-6">
            <TabsTrigger value="about" className="gap-1.5">
              <Info className="w-4 h-4" /> About
            </TabsTrigger>
            <TabsTrigger value="notes" className="gap-1.5">
              <StickyNote className="w-4 h-4" /> Notes
            </TabsTrigger>
          </TabsList>

          <TabsContent value="about">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">About This Project</h3>
                {!isOwner && (
                  <span className="text-xs text-muted-foreground">Read only</span>
                )}
              </div>
              <NotesEditor
                key={`about-${projectId}`}
                initialContent={aboutContent as unknown[]}
                onChange={isOwner ? handleAboutChange : undefined}
                editable={isOwner}
              />
            </div>
          </TabsContent>

          <TabsContent value="notes">
            {projectId && <NotesView scope={{ projectId }} />}
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default ProjectDetail;
