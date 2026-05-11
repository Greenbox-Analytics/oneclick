import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ArtistInfo } from "@/hooks/usePortfolioData";
import { NewArtistDialog } from "@/components/NewArtistDialog";
import { Plus } from "lucide-react";

const ADD_NEW_ARTIST_VALUE = "__add_new_artist__";

interface ProjectFormDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  project?: { id: string; name: string; description: string | null; artist_id: string } | null;
  artists: ArtistInfo[];
  defaultArtistId?: string;
  onSave: (data: { name: string; description: string; artist_id: string }) => Promise<void>;
}

export const ProjectFormDialog = ({
  open,
  onOpenChange,
  project,
  artists,
  defaultArtistId,
  onSave,
}: ProjectFormDialogProps) => {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [artistId, setArtistId] = useState("");
  const [saving, setSaving] = useState(false);
  const [newArtistDialogOpen, setNewArtistDialogOpen] = useState(false);

  const isEdit = !!project;

  const handleArtistChange = (value: string) => {
    if (value === ADD_NEW_ARTIST_VALUE) {
      setNewArtistDialogOpen(true);
      return;
    }
    setArtistId(value);
  };

  useEffect(() => {
    if (project) {
      setName(project.name);
      setDescription(project.description || "");
      setArtistId(project.artist_id);
    } else {
      setName("");
      setDescription("");
      setArtistId(defaultArtistId || "");
    }
  }, [project, open, defaultArtistId]);

  const handleSubmit = async () => {
    if (!name.trim() || !artistId) return;
    setSaving(true);
    try {
      await onSave({
        name: name.trim(),
        description: description.trim(),
        artist_id: artistId,
      });
      onOpenChange(false);
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{isEdit ? "Edit Project" : "Add Project"}</DialogTitle>
          <DialogDescription>
            {isEdit
              ? "Update project details"
              : "Create a new project for an artist"}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div>
            <Label htmlFor="project-artist">
              Artist <span className="text-green-500">*</span>
            </Label>
            <Select value={artistId} onValueChange={handleArtistChange} disabled={isEdit}>
              <SelectTrigger>
                <SelectValue placeholder="Select artist" />
              </SelectTrigger>
              <SelectContent>
                {artists.map((a) => (
                  <SelectItem key={a.id} value={a.id}>
                    {a.name}
                  </SelectItem>
                ))}
                {!isEdit && (
                  <>
                    {artists.length > 0 && (
                      <div className="my-1 h-px bg-border" role="separator" />
                    )}
                    <SelectItem value={ADD_NEW_ARTIST_VALUE} className="text-primary font-medium">
                      <span className="flex items-center gap-2">
                        <Plus className="w-4 h-4" />
                        Add new artist
                      </span>
                    </SelectItem>
                  </>
                )}
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label htmlFor="project-name">
              Project Name <span className="text-green-500">*</span>
            </Label>
            <Input
              id="project-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Album Title, Single Name"
            />
          </div>

          <div>
            <Label htmlFor="project-description">Description</Label>
            <Textarea
              id="project-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description..."
              rows={3}
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={!name.trim() || !artistId || saving}>
            {saving ? "Saving..." : isEdit ? "Save Changes" : "Add Project"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
    <NewArtistDialog
      open={newArtistDialogOpen}
      onOpenChange={setNewArtistDialogOpen}
      onCreated={(id) => setArtistId(id)}
    />
    </>
  );
};
