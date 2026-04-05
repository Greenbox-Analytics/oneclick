import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Loader2 } from "lucide-react";
import { useCreateWork } from "@/hooks/useRegistry";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import { API_URL, apiFetch } from "@/lib/apiFetch";

interface AddWorkDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId: string;
  artistId: string;
}

const WORK_TYPES = [
  { value: "single", label: "Single" },
  { value: "ep_track", label: "EP Track" },
  { value: "album_track", label: "Album Track" },
  { value: "composition", label: "Composition" },
  { value: "other", label: "Other" },
];

export default function AddWorkDialog({ open, onOpenChange, projectId, artistId }: AddWorkDialogProps) {
  const { user } = useAuth();
  const [title, setTitle] = useState("");
  const [workType, setWorkType] = useState("single");
  const [customWorkType, setCustomWorkType] = useState("");
  const [isrc, setIsrc] = useState("");
  const [selectedAudioId, setSelectedAudioId] = useState<string>("");

  const createWork = useCreateWork();

  // Fetch audio files linked to this project
  const audioQuery = useQuery({
    queryKey: ["project-audio-for-dialog", projectId],
    queryFn: async () => {
      const { data } = await (supabase as any)
        .from("project_audio_links")
        .select("audio_file_id, audio_files(id, file_name)")
        .eq("project_id", projectId);
      return data || [];
    },
    enabled: open,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim()) return;

    createWork.mutate(
      {
        artist_id: artistId,
        project_id: projectId,
        title: title.trim(),
        work_type: workType,
        ...(workType === "other" && customWorkType.trim()
          ? { custom_work_type: customWorkType.trim() }
          : {}),
        ...(isrc.trim() ? { isrc: isrc.trim() } : {}),
      },
      {
        onSuccess: async (data: any) => {
          // Link audio file if one was selected
          if (selectedAudioId && data?.id && user?.id) {
            try {
              await apiFetch(
                `${API_URL}/registry/works/${data.id}/audio?audio_file_id=${selectedAudioId}&user_id=${user.id}`,
                { method: "POST" }
              );
            } catch {
              // Non-blocking: work was created, audio link failed silently
            }
          }
          setTitle("");
          setWorkType("single");
          setCustomWorkType("");
          setIsrc("");
          setSelectedAudioId("");
          onOpenChange(false);
        },
      }
    );
  };

  const audioFiles = (audioQuery.data || [])
    .map((link: any) => link.audio_files)
    .filter(Boolean);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Add Work</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="work-title">Title *</Label>
            <Input
              id="work-title"
              placeholder="Song or composition title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              required
            />
          </div>

          <div className="space-y-2">
            <Label>Work Type</Label>
            <Select value={workType} onValueChange={setWorkType}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {WORK_TYPES.map((t) => (
                  <SelectItem key={t.value} value={t.value}>
                    {t.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {workType === "other" && (
            <div className="space-y-2">
              <Label htmlFor="custom-type">Custom Type</Label>
              <Input
                id="custom-type"
                placeholder="e.g. Remix, Interlude..."
                value={customWorkType}
                onChange={(e) => setCustomWorkType(e.target.value)}
              />
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="work-isrc">ISRC (optional)</Label>
            <Input
              id="work-isrc"
              placeholder="e.g. USRC17607839"
              value={isrc}
              onChange={(e) => setIsrc(e.target.value)}
            />
          </div>

          {audioFiles.length > 0 && (
            <div className="space-y-2">
              <Label>Link Audio File</Label>
              <Select value={selectedAudioId} onValueChange={setSelectedAudioId}>
                <SelectTrigger>
                  <SelectValue placeholder="(None)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">(None)</SelectItem>
                  {audioFiles.map((af: any) => (
                    <SelectItem key={af.id} value={af.id}>
                      {af.file_name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={!title.trim() || createWork.isPending}>
              {createWork.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
              Add Work
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
