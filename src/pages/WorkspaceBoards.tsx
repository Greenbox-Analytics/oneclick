import { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Music, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { KanbanBoard } from "@/components/workspace/boards/KanbanBoard";
import { useAnalytics } from "@/hooks/useAnalytics";
import { useSmartBack } from "@/hooks/useSmartBack";

const WorkspaceBoards = () => {
  const navigate = useNavigate();
  const goBack = useSmartBack("/workspace?tab=boards");
  const { artistId } = useParams<{ artistId?: string }>();
  const { captureToolOpened } = useAnalytics();

  useEffect(() => {
    captureToolOpened("boards");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={goBack}
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <div
            className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate("/dashboard")}
          >
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <h2 className="text-3xl font-bold text-foreground mb-2">
            Project Board
          </h2>
          <p className="text-muted-foreground">
            {artistId
              ? "Manage tasks for this artist"
              : "Manage all your project tasks"}
          </p>
        </div>

        <KanbanBoard artistId={artistId} />
      </main>
    </div>
  );
};

export default WorkspaceBoards;
