import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Loader2 } from "lucide-react";
import { useCreateWork } from "@/hooks/useRegistry";

interface AddWorkDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId: string;
  artistId: string;
}

const WORK_TYPES = ["Single", "EP Track", "Album Track", "Composition", "Other"];

export default function AddWorkDialog({ open, onOpenChange, projectId, artistId }: AddWorkDialogProps) {
  const [title, setTitle] = useState("");
  const [workType, setWorkType] = useState("Single");
  const [customWorkType, setCustomWorkType] = useState("");
  const [isrc, setIsrc] = useState("");

  const createWork = useCreateWork();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim()) return;

    const finalType = workType === "Other" ? customWorkType.trim() || "Other" : workType;

    createWork.mutate(
      {
        artist_id: artistId,
        project_id: projectId,
        title: title.trim(),
        work_type: finalType,
        ...(isrc.trim() ? { isrc: isrc.trim() } : {}),
      },
      {
        onSuccess: () => {
          setTitle("");
          setWorkType("Single");
          setCustomWorkType("");
          setIsrc("");
          onOpenChange(false);
        },
      }
    );
  };

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
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {workType === "Other" && (
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
