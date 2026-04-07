import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Plus, X } from "lucide-react";
import { getLabelColor } from "./labelColors";

interface TaskLabelsProps {
  labels: string[];
  setLabels: (labels: string[]) => void;
  saveField: (field: string, value: unknown) => void;
}

export function TaskLabels({ labels, setLabels, saveField }: TaskLabelsProps) {
  const [newLabel, setNewLabel] = useState("");

  const addLabel = () => {
    if (!newLabel.trim() || labels.includes(newLabel.trim())) return;
    const updated = [...labels, newLabel.trim()];
    setLabels(updated);
    setNewLabel("");
    saveField("labels", updated);
  };

  const removeLabel = (label: string) => {
    const updated = labels.filter((l) => l !== label);
    setLabels(updated);
    saveField("labels", updated);
  };

  return (
    <div className="space-y-2">
      <Label className="text-xs text-muted-foreground uppercase tracking-wider">
        Labels
      </Label>
      <div className="flex flex-wrap gap-1.5 mb-2">
        {labels.map((label) => {
          const lc = getLabelColor(label);
          return (
          <Badge
            key={label}
            variant="secondary"
            className={`gap-1 pr-1 ${lc.bg} ${lc.text}`}
          >
            {label}
            <button
              type="button"
              onClick={() => removeLabel(label)}
              className="hover:text-destructive"
            >
              <X className="h-3 w-3" />
            </button>
          </Badge>
          );
        })}
      </div>
      <div className="flex gap-2">
        <Input
          value={newLabel}
          onChange={(e) => setNewLabel(e.target.value)}
          placeholder="Add label..."
          className="h-8"
          onKeyDown={(e) => e.key === "Enter" && addLabel()}
        />
        <Button
          size="sm"
          variant="outline"
          onClick={addLabel}
          disabled={!newLabel.trim()}
          className="h-8"
        >
          <Plus className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  );
}
